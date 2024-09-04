# utils/model_training.py
from sklearn.model_selection import train_test_split


def split_data(data, target_column, test_size=0.2):
    """
    Splits the dataset into training and testing sets.

    Parameters:
    - data: pandas DataFrame, the dataset to split
    - target_column: str, the name of the target column
    - test_size: float, the proportion of the dataset to include in the test split

    Returns:
    - X_train, X_test, y_train, y_test: split datasets
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score


def train_model(model, X_train, y_train):
    """
    Train the model and return the trained model.

    Parameters:
    - model: The model to be trained.
    - X_train: Training features.
    - y_train: Training labels/target.

    Returns:
    - Trained model.
    """
    if not hasattr(model, 'fit'):
        raise TypeError("The provided model does not have a 'fit' method.")

    model.fit(X_train, y_train)
    return model


