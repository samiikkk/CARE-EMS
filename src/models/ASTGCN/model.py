import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import cheb_polynomial


# Spatial Attention 
class SpatialAttention(nn.Module):
   
    #Inputs / outputs in (B, N, F, T) convention.
    #Returns: (B, N, N)

    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(SpatialAttention, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))

    def forward(self, x):
        # x: (B, N, F, T)
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)   # (B, N, F, T)(T) -> (B, N, F) -> (B, N, T)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)         # (F)(B,N,F,T) -> (B,N,T) -> (B,T,N)
        product = torch.matmul(lhs, rhs)                          # (B, N, N)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B,N,N) -> (B,N,N)
        return F.softmax(S, dim=1)                                # (B, N, N)


# Temporal Attention 
class TemporalAttention(nn.Module):

    #Inputs / outputs in (B, N, F, T) convention.
    #Returns: (B, T, T)

    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(TemporalAttention, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

    def forward(self, x):
        # x: (B, N, F, T)
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # (B,N,F,T) -> (B,T,F,N)(N) -> (B,T,F) -> (B,T,N)
        rhs = torch.matmul(self.U3, x)                            # (F)(B,N,F,T) -> (B,N,T)
        product = torch.matmul(lhs, rhs)                          # (B,T,N)(B,N,T) -> (B,T,T)
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B,T,T)
        return F.softmax(E, dim=1)                                # (B, T, T)


# Chebyshev GCN with Spatial Attention 
class ChebGraphConvWithSAtt(nn.Module):

    #Uses pre-computed Chebyshev polynomial tensors.
    #Input / output: (B, N, F_in, T) -> (B, N, F_out, T)

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        super(ChebGraphConvWithSAtt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE))
            for _ in range(K)
        ])

    def forward(self, x, spatial_attention):
        # x: (B, N, F_in, T),  spatial_attention: (B, N, N)
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]              # (B, N, F_in)
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)
            for k in range(self.K):
                T_k = self.cheb_polynomials[k]                 # (N, N)
                T_k_with_at = T_k.mul(spatial_attention)       # (B, N, N)  element-wise
                theta_k = self.Theta[k]                        # (F_in, F_out)
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (B, N, F_in)
                output = output + rhs.matmul(theta_k)          # (B, N, F_out)
            outputs.append(output.unsqueeze(-1))               # (B, N, F_out, 1)
        return F.relu(torch.cat(outputs, dim=-1))              # (B, N, F_out, T)


# ASTGCN Block 
class ASTGCNBlock(nn.Module):

    #One ST block.  All tensors are (B, N, F, T) internally.
    #time_strides controls downsampling along T in the temporal conv.

    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
                 time_strides, cheb_polynomials, num_of_vertices, num_of_timesteps):
        super(ASTGCNBlock, self).__init__()
        self.TAt = TemporalAttention(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = SpatialAttention(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = ChebGraphConvWithSAtt(K, cheb_polynomials, in_channels, nb_chev_filter)
        # time conv: (B, F, N, T) -> (B, nb_time_filter, N, T//time_strides)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter,
                                   kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter,
                                       kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x):
        # x: (B, N, F_in, T)
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # 1. Temporal attention 
        temporal_At = self.TAt(x)                              # (B, T, T)
        x_TAt = torch.matmul(
            x.reshape(batch_size, -1, num_of_timesteps), temporal_At
        ).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)

        # 2. Spatial attention on the temporally-attended x
        spatial_At = self.SAt(x_TAt)                           # (B, N, N)

        # 3. Chebyshev GCN with spatial attention
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)        # (B, N, F_chev, T)

        # 4. Temporal conv 
        time_conv_output = self.time_conv(
            spatial_gcn.permute(0, 2, 1, 3)                    # (B, F_chev, N, T)
        )                                                       # (B, nb_time_filter, N, T')

        # 5. Residual shortcut
        x_residual = self.residual_conv(
            x.permute(0, 2, 1, 3)                              # (B, F_in, N, T)
        )                                                       # (B, nb_time_filter, N, T')

        # 6. Add + LayerNorm  
        x_residual = self.ln(
            F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)
            # (B, nb_time_filter, N, T') -> (B, T', N, nb_time_filter)
        ).permute(0, 2, 3, 1)
        # (B, T', N, nb_time_filter) -> (B, N, nb_time_filter, T')

        return x_residual                                       # (B, N, nb_time_filter, T')


# ASTGCN submodule 
class ASTGCN(nn.Module):

    #Stack of ASTGCN blocks + final prediction conv.
    #Expects input (B, N, F_in, T_in); returns (B, N, T_out).

    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter,
                 time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices):
        super(ASTGCN, self).__init__()

        self.BlockList = nn.ModuleList([
            ASTGCNBlock(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
                        time_strides, cheb_polynomials, num_of_vertices, len_input)
        ])
        self.BlockList.extend([
            ASTGCNBlock(DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter,
                        1, cheb_polynomials, num_of_vertices, len_input // time_strides)
            for _ in range(nb_block - 1)
        ])
        # final_conv: (B, T//stride, N, nb_time_filter) -> (B, num_for_predict, N, 1)
        self.final_conv = nn.Conv2d(int(len_input / time_strides), num_for_predict,
                                    kernel_size=(1, nb_time_filter))
        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, x):
        # x: (B, N, F_in, T_in)
        for block in self.BlockList:
            x = block(x)
        # x: (B, N, nb_time_filter, T//stride)
        output = self.final_conv(x.permute(0, 3, 1, 2))        # (B, T, N, F) -> (B, num_for_predict, N, 1)
        output = output[:, :, :, -1].permute(0, 2, 1)          # (B, N, num_for_predict)
        return output


def make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter,
               time_strides, L_tilde_np, num_for_predict, len_input, num_of_vertices):

    #Build pre-compute Chebyshev polynomials once from L_tilde (numpy),then pass them into the model

    cheb_polys = [
        torch.from_numpy(p).float().to(DEVICE)
        for p in cheb_polynomial(L_tilde_np, K)
    ]
    model = ASTGCN(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter,
                   time_strides, cheb_polys, num_for_predict, len_input, num_of_vertices)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model
