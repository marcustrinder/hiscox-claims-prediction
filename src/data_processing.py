import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def collect_from_database(query: str) -> pd.DataFrame:
    # Simulating data collection from a database
    n_rows = 10_000
    n_features = 16
    features, labels = make_classification(
        n_samples=n_rows,
        n_features=n_features,
        n_informative=7,
        n_redundant=4,
        n_repeated=3,
        n_classes=2,
        class_sep=1.2,
        flip_y=0.035,
        weights=[0.85, 0.15],
        random_state=1889,
    )
    df = pd.DataFrame(features, columns=[f'numeric_{i+1}' for i in range(n_features)])
    df.insert(value=labels, loc=0, column='claim_status')
    # Simulating data processing
    df['age'] = MinMaxScaler(feature_range=(18, 95)).fit_transform(df['numeric_1'].values[:, None]).astype('int')
    df['height_cm'] = MinMaxScaler(feature_range=(140, 210)).fit_transform(df['numeric_2'].values[:, None]).astype('int')
    df['weight_kg'] = MinMaxScaler(feature_range=(45, 125)).fit_transform(df['numeric_3'].values[:, None]).astype('int')
    df['income'] = MinMaxScaler(feature_range=(0, 250_000)).fit_transform(df['numeric_4'].values[:, None]).astype('int')
    # Continue processing for other columns...
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['family_history_3', 'employment_type'])
    non_numerical = ['gender', 'marital_status', 'occupation', 'location', 'prev_claim_rejected', 'known_health_conditions', 'uk_residence', 'family_history_1', 'family_history_2', 'family_history_4', 'family_history_5', 'product_var_1', 'product_var_2', 'product_var_3', 'health_status', 'driving_record', 'previous_claim_rate', 'education_level', 'income level', 'n_dependents']
    for column in non_numerical:
        df[column] = df[column].astype('category')
    return df
