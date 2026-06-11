import pandas as pd
import numpy as np
import itertools
import json

with open('categories.json', 'r') as f:
    categories = json.load(f)

def fill_missing_ZIP(df):

    df['ZIPCODE'] = df.groupby(['BOROUGH', 'INCIDENT_DISPATCH_AREA'])['ZIPCODE'].transform(
        lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x
    )
    return df


def aggregate(df):

    #Data aggregation for indicator columns on hourly basis for each zip code
    def get_proportion(x):
        return x.value_counts(normalize=True).get('Y', 0)


    indicator_cols = [
     "SPECIAL_EVENT_INDICATOR",
    ]

    categories_cols = [
        'Environmental and Poisoning Emergencies',
        'Mass Casualty or Public Incidents', 'Medical Emergencies', 'Other',
        'Trauma-Related Incidents']


    agg = {
        "INITIAL_SEVERITY_LEVEL_CODE": "median",
        # "FINAL_SEVERITY_LEVEL_CODE": "median",
        # "DISPATCH_RESPONSE_SECONDS_QY": "median",
        # "INCIDENT_RESPONSE_SECONDS_QY": "median",
        # "INCIDENT_TRAVEL_TM_SECONDS_QY": "median",
        **{c: get_proportion for c in indicator_cols},
        **{c: "sum" for c in categories_cols},
        'INCIDENT_DISPATCH_AREA': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
        "BOROUGH": "first",
    }

    grouped = (
    df.groupby(['date', 'hour', 'ZIPCODE'])
    .agg(agg)
    .reset_index()
    )
    print("Aggregated data on an hourly basis for each ZIPCODE.", flush=True)

    grouped['call_count'] = grouped['Medical Emergencies'] + grouped['Trauma-Related Incidents'] + grouped['Mass Casualty or Public Incidents'] + grouped['Environmental and Poisoning Emergencies'] + grouped['Other']
    print("Calculated call_count for each group.")

    grouped = grouped.sort_values(['ZIPCODE', 'date', 'hour']).reset_index(drop=True)
    print("Sorted grouped data by ZIPCODE, date, and hour.")

    return grouped



def incident_type_categorization(x):
    return next((cat for cat, codes in categories.items() if x in codes), 'no-category')


df =  pd.read_csv('EMS_Incident_Dispatch_Data_After_2017.csv')
print(f"Initial data shape: {df.shape}", flush=True)

# Drop unnecessary columns
df = df[['INCIDENT_DATETIME', 'INITIAL_CALL_TYPE', 'INITIAL_SEVERITY_LEVEL_CODE', 'BOROUGH', 'INCIDENT_DISPATCH_AREA', 'ZIPCODE', 'SPECIAL_EVENT_INDICATOR']]
print("Dropped unnecessary columns.", flush=True)

# Filling missing zip code with mode values within each BOROUGH and INCIDENT_DISPATCH_AREA group
df = fill_missing_ZIP(df)
print('Successfully filled missing zip codes with mode values within each BOROUGH and INCIDENT_DISPATCH_AREA group.', flush=True)

# Drop rows with missing ZIPCODE
df.dropna(subset=['ZIPCODE'], inplace=True)
print('Dropped rows with missing ZIPCODE.', flush=True)


# Keeping only rows with valid 5-digit ZIP codes
df['ZIPCODE'] = (
    df['ZIPCODE']
    .astype(str)              # Convert to string
    .str.split('.').str[0]    # Remove '.0' 
    .str.strip()              # Remove spaces
)
df = df[df['ZIPCODE'].str.match(r'^\d{5}$')].reset_index(drop=True)
print("Filtered to keep only valid 5-digit ZIPCODEs.", flush=True)


# Convert INCIDENT_DATETIME to datetime and extract date and hour
df['INCIDENT_DATETIME'] = pd.to_datetime(df['INCIDENT_DATETIME'], errors='coerce')
df['date'] = df['INCIDENT_DATETIME'].dt.date
df['hour'] = df['INCIDENT_DATETIME'].dt.hour
df.drop(columns=['INCIDENT_DATETIME'], inplace=True)
print("Extracted date and hour from INCIDENT_DATETIME.", flush=True)

# #Since 'INCIDENT_RESPONSE_SECONDS_QY' and 'INCIDENT_TRAVEL_TM_SECONDS_QY' have many null values, the aggregated hourly records had lots of null values for some hours, so let's first interpolate that particular value. Firstly, the dataset is sorted according to zip code, date and hour then a linear interpolation is done to presever temporal sequence.
# df = df.sort_values(['ZIPCODE', 'date', 'hour']).reset_index(drop=True)
# df['INCIDENT_RESPONSE_SECONDS_QY'] = df.groupby('ZIPCODE')['INCIDENT_RESPONSE_SECONDS_QY'].transform(lambda x: x.interpolate(limit_direction='both'))
# df['INCIDENT_TRAVEL_TM_SECONDS_QY'] = df.groupby('ZIPCODE')['INCIDENT_TRAVEL_TM_SECONDS_QY'].transform(lambda x: x.interpolate(limit_direction='both'))
# print("Linear interpolation done on INCIDENT_RESPONSE_SECONDS_QY and INCIDENT_TRAVEL_TM_SECONDS_QY.", flush=True)

#Using INITIAL_CALL_TYPE column to categorize the samples into 5 categories (medical, trauma and so on)

entries_in_categories = list(itertools.chain.from_iterable(categories.values()))
df = df[df['INITIAL_CALL_TYPE'].map(lambda x: True if x in entries_in_categories else False)]


df['INITIAL_CALL_TYPE'] = df['INITIAL_CALL_TYPE'].map(incident_type_categorization)

df = pd.concat([df, pd.get_dummies(df['INITIAL_CALL_TYPE'], dtype=int)], axis=1)
df.drop(columns=['INITIAL_CALL_TYPE'], inplace=True)
print("Categorized INITIAL_CALL_TYPE into broader categories.", flush=True)

aggregated_df = aggregate(df)
aggregated_df.to_csv('aggregated_data.csv', index=False)
print("Saved the aggregated data to 'aggregated_data.csv'.", flush=True)