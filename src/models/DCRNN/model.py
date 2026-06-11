import torch
import torch.nn as nn
import numpy as np

from utils import build_sparse_supports


# Diffusion Convolution GRU Cell 
class DCGRUCell(nn.Module):

    def __init__(self, input_dim, num_units, supports, max_diffusion_step, num_nodes):
        super().__init__()
        self._num_units     = num_units
        self._num_nodes     = num_nodes
        self._max_diff_step = max_diffusion_step
        self._supports      = supports                        # list of sparse tensors

        # number of diffusion matrices per support × steps  +  identity (k=0)
        num_supports = len(supports)
        num_matrices = num_supports * max_diffusion_step + 1  # +1 for x itself

        # input_size for gconv  =  (input_dim + hidden_dim)  (concatenated)
        input_and_hidden = input_dim + num_units

        # gate projection  (reset + update gates combined, output = 2 * num_units)
        self.W_gate = nn.Parameter(
            torch.empty(input_and_hidden * num_matrices, 2 * num_units)
        )
        self.b_gate = nn.Parameter(torch.ones(2 * num_units))   # bias_start=1 for gates

        # candidate projection
        self.W_cand = nn.Parameter(
            torch.empty(input_and_hidden * num_matrices, num_units)
        )
        self.b_cand = nn.Parameter(torch.zeros(num_units))

        nn.init.xavier_normal_(self.W_gate)
        nn.init.xavier_normal_(self.W_cand)

    # ------------------------------------------------------------------
    def _diffusion_conv(self, x, W, b):
        B, N, C = x.shape

        # x0 shape: (N, C*B), used for sparse mm
        x0 = x.permute(1, 2, 0).reshape(N, C * B)      # (N, C*B)
        diffused = [x0]                                  # k=0 : identity

        for support in self._supports:
            x1 = torch.sparse.mm(support, x0)           # (N, C*B)
            diffused.append(x1)
            for _ in range(2, self._max_diff_step + 1):
                x2 = 2 * torch.sparse.mm(support, x1) - x0
                diffused.append(x2)
                x1, x0 = x2, x1
            x0 = x.permute(1, 2, 0).reshape(N, C * B)  # reset x0 for next support

        # stack: (num_matrices, N, C*B)
        out = torch.stack(diffused, dim=0)
        out = out.reshape(len(diffused), N, C, B)
        out = out.permute(3, 1, 2, 0)                   # (B, N, C, num_matrices)
        out = out.reshape(B * N, C * len(diffused))      # (B*N, C*num_matrices)

        out = out @ W + b                                # (B*N, out_dim)
        return out.reshape(B, N * out.shape[-1] // (N // N), -1).reshape(B, -1)


    def _gconv(self, x, state, W, b):

        B = x.shape[0]
        N = self._num_nodes

        x_r     = x.reshape(B, N, -1)                   # (B, N, input_dim)
        state_r = state.reshape(B, N, -1)                # (B, N, num_units)
        x_s     = torch.cat([x_r, state_r], dim=-1)     # (B, N, input_dim+num_units)

        C = x_s.shape[-1]
        x0 = x_s.permute(1, 2, 0).reshape(N, C * B)    # (N, C*B)

        diffused = [x0]
        for support in self._supports:
            x1 = torch.sparse.mm(support, x0)
            diffused.append(x1)
            x0_orig = x0.clone()
            for _ in range(2, self._max_diff_step + 1):
                x2 = 2 * torch.sparse.mm(support, x1) - x0_orig
                diffused.append(x2)
                x1, x0_orig = x2, x1
            x0 = x_s.permute(1, 2, 0).reshape(N, C * B)   # reset for next support

        num_mat = len(diffused)
        out = torch.stack(diffused, dim=0)               # (num_mat, N, C*B)
        out = out.reshape(num_mat, N, C, B)
        out = out.permute(3, 1, 2, 0)                   # (B, N, C, num_mat)
        out = out.reshape(B * N, C * num_mat)            # (B*N, C*num_mat)

        out = out @ W + b                                # (B*N, out_dim)
        return out.reshape(B, -1)                        # (B, N*out_dim)

    def forward(self, x, hx):
        # reset & update gates 
        gate_out = torch.sigmoid(
            self._gconv(x, hx, self.W_gate, self.b_gate)
        )                                                # (B, N * 2*num_units)
        gate_out = gate_out.reshape(-1, self._num_nodes, 2 * self._num_units)
        r, u = gate_out[..., :self._num_units], gate_out[..., self._num_units:]
        r = r.reshape(x.shape[0], -1)                   # (B, N*num_units)
        u = u.reshape(x.shape[0], -1)

        # candidate
        c = torch.tanh(
            self._gconv(x, r * hx, self.W_cand, self.b_cand)
        )                                                # (B, N*num_units)

        # new state 
        new_h = u * hx + (1.0 - u) * c
        return new_h                                     # (B, N*num_units)


#Encoder
class DCRNNEncoder(nn.Module):
    def __init__(self, input_dim, num_units, supports, max_diffusion_step,
                 num_nodes, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.num_nodes  = num_nodes
        self.num_units  = num_units

        # first layer: input_dim -> num_units
        # subsequent: num_units -> num_units
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else num_units
            self.cells.append(
                DCGRUCell(in_dim, num_units, supports, max_diffusion_step, num_nodes)
            )

    def forward(self, x_seq):
        B = x_seq.shape[1]
        device = x_seq.device

        # initialise hidden states to zero
        hidden = [
            torch.zeros(B, self.num_nodes * self.num_units, device=device)
            for _ in range(self.num_layers)
        ]

        for t in range(x_seq.shape[0]):
            inp = x_seq[t]                      # (B, N*input_dim)
            for l, cell in enumerate(self.cells):
                hidden[l] = cell(inp, hidden[l])
                inp = hidden[l]                 # feed this layer's output to the next

        return torch.stack(hidden, dim=0)       # (num_layers, B, N*num_units)


#Decoder 
class DCRNNDecoder(nn.Module):
    def __init__(self, output_dim, num_units, supports, max_diffusion_step,
                 num_nodes, num_layers, horizon):
        super().__init__()
        self.num_layers = num_layers
        self.num_nodes  = num_nodes
        self.num_units  = num_units
        self.horizon    = horizon
        self.output_dim = output_dim

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = output_dim if i == 0 else num_units
            self.cells.append(
                DCGRUCell(in_dim, num_units, supports, max_diffusion_step, num_nodes)
            )

        # project hidden units -> output_dim per node
        self.projection = nn.Linear(num_units, output_dim)

    def forward(self, encoder_hidden, labels=None,
                use_curriculum=False, cl_threshold=None):
        B = encoder_hidden.shape[1]
        device = encoder_hidden.device

        hidden = [encoder_hidden[l] for l in range(self.num_layers)]

        # start token: all zeros  (GO symbol)
        dec_in = torch.zeros(B, self.num_nodes * self.output_dim, device=device)

        outputs = []
        for t in range(self.horizon):
            inp = dec_in
            for l, cell in enumerate(self.cells):
                hidden[l] = cell(inp, hidden[l])
                inp = hidden[l]

            # project last layer's hidden to output
            # inp shape: (B, N*num_units)
            proj_in = inp.reshape(B * self.num_nodes, self.num_units)
            dec_out = self.projection(proj_in)             # (B*N, output_dim)
            dec_out = dec_out.reshape(B, self.num_nodes * self.output_dim)
            outputs.append(dec_out)

            if use_curriculum and labels is not None and cl_threshold is not None:
                if np.random.rand() < cl_threshold:
                    dec_in = labels[t]
                else:
                    dec_in = dec_out.detach()
            else:
                dec_in = dec_out.detach()

        return torch.stack(outputs, dim=0)   # (horizon, B, N*output_dim)


#Full DCRNN Model
class DCRNNModel(nn.Module):

    def __init__(self, adj_mx, input_dim, output_dim, num_units,
                 num_rnn_layers, max_diffusion_step, horizon,
                 filter_type="dual_random_walk",
                 cl_decay_steps=1000,
                 use_curriculum_learning=True):
        super().__init__()
        self.horizon    = horizon
        self.num_nodes  = adj_mx.shape[0]
        self.output_dim = output_dim
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum = use_curriculum_learning

        # build transition matrices (sparse, on device)
        supports = build_sparse_supports(adj_mx, filter_type)

        self.encoder = DCRNNEncoder(
            input_dim, num_units, supports, max_diffusion_step,
            self.num_nodes, num_rnn_layers
        )
        self.decoder = DCRNNDecoder(
            output_dim, num_units, supports, max_diffusion_step,
            self.num_nodes, num_rnn_layers, horizon
        )

    def _cl_threshold(self, batches_seen):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps)
        )

    def forward(self, x, labels=None, batches_seen=None):
        B, T, N, F = x.shape

        # Reshape for encoder: (seq_len, B, N*F)
        x_enc = x.permute(1, 0, 2, 3).reshape(T, B, N * F)

        enc_hidden = self.encoder(x_enc)     # (num_layers, B, N*num_units)

        dec_labels = None
        if labels is not None and self.training:
            # labels: (B, horizon, N) -> (horizon, B, N*output_dim)
            dec_labels = labels.permute(1, 0, 2)          # (horizon, B, N)
            dec_labels = dec_labels.reshape(
                self.horizon, B, N * self.output_dim
            )

        cl_thresh = (self._cl_threshold(batches_seen)
                     if (batches_seen is not None and self.use_curriculum)
                     else None)

        dec_out = self.decoder(
            enc_hidden, dec_labels,
            use_curriculum=self.use_curriculum and self.training,
            cl_threshold=cl_thresh
        )
        # dec_out: (horizon, B, N*output_dim)
        dec_out = dec_out.reshape(self.horizon, B, N, self.output_dim)
        dec_out = dec_out.permute(1, 0, 2, 3)             # (B, horizon, N, output_dim)
        return dec_out[..., 0]                             # (B, horizon, N)
