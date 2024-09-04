# utils/model_evaluation.py
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluates a classification model using common metrics.

    Parameters:
    - model: sklearn estimator, the trained classification model
    - X_test: pandas DataFrame, the test data features
    - y_test: pandas Series, the test data labels

    Returns:
    - metrics: dict, a dictionary containing evaluation metrics
    """
    predictions = model.predict(X_test)

    # Calculate confusion matrix and accuracy
    conf_matrix = confusion_matrix(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)

    # Get classification report and convert it to DataFrame
    class_report = classification_report(y_test, predictions, output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()

    # Organize metrics
    metrics = {
        "Confusion Matrix": conf_matrix,
        "Classification Report": class_report_df,
        "Accuracy": accuracy
    }

    return metrics

def evaluate_regression_model(model, X_test, y_test):
    """
    Evaluates a regression model using common metrics.

    Parameters:
    - model: sklearn estimator, the trained regression model
    - X_test: pandas DataFrame, the test data features
    - y_test: pandas Series, the test data labels

    Returns:
    - metrics: dict, a dictionary containing evaluation metrics
    """
    predictions = model.predict(X_test)
    metrics = {
        "Mean Squared Error": mean_squared_error(y_test, predictions),
        "R2 Score": r2_score(y_test, predictions)
    }
    return metrics
