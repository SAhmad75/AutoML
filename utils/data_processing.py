# utils/data_processing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def fill_missing_values(data, strategy='mean'):
    """
    Fill missing values in the dataset based on the specified strategy.

    Parameters:
    - data: DataFrame containing the dataset.
    - strategy: Strategy for filling missing values (e.g., 'mean', 'median', 'mode').

    Returns:
    - DataFrame with missing values filled.
    """
    if strategy == 'mean':
        for column in data.select_dtypes(include=['float64', 'int64']).columns:
            data[column] = data[column].fillna(data[column].mean())
    elif strategy == 'median':
        for column in data.select_dtypes(include=['float64', 'int64']).columns:
            data[column] = data[column].fillna(data[column].median())
    elif strategy == 'mode':
        for column in data.select_dtypes(include=['float64', 'int64']).columns:
            data[column] = data[column].fillna(data[column].mode()[0])
    else:
        raise ValueError("Strategy must be 'mean', 'median', or 'mode'")
    return data

def label_encode(data):
    """
    Encodes categorical columns using label encoding.

    Parameters:
    - data: DataFrame containing the dataset.

    Returns:
    - DataFrame with categorical columns encoded.
    """
    # Identify categorical columns (columns with object or string data type)
    categorical_columns = data.select_dtypes(include=['object']).columns

    # Apply label encoding to these categorical columns
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        data[column] = label_encoder.fit_transform(data[column].astype(str))

    return data

def one_hot_encode(data):
    """
    Encodes categorical columns using one-hot encoding.

    Parameters:
    - data: DataFrame containing the dataset.

    Returns:
    - DataFrame with categorical columns one-hot encoded.
    """
    return pd.get_dummies(data)
