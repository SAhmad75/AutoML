import streamlit as st
import pandas as pd
import joblib
import os
import io
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

# Import utility functions
from utils.data_processing import label_encode, one_hot_encode, fill_missing_values
from utils.model_training import split_data, train_model
from utils.model_evaluation import evaluate_classification_model, evaluate_regression_model
from utils.hyperparameter_tuning import hyperparameter_tuning

# Define hyperparameter grids for each model
hyperparameter_grids = {
    "Random Forest": {
        "n_estimators": [50, 100, 150],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5, 10]
    },
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l1", "l2"]
    },
    "Support Vector Classifier": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": [0.001, 0.01, 0.1, 1]
    },
    "XGBoost Classifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0]
    },
    "Random Forest Regressor": {
        "n_estimators": [50, 100, 150],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5, 10]
    },
    "Support Vector Regressor": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": [0.001, 0.01, 0.1, 1]
    },
    "XGBoost Regressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0]
    }
}
# Initialize session state for storing model metrics and paths
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'model_paths' not in st.session_state:
    st.session_state.model_paths = {}
if 'previous_file' not in st.session_state:
    st.session_state.previous_file = None

with st.sidebar:
    st.image("Pic.jpg")
    st.title("Athena AutoML")
    st.info("Building Automated ML Pipelines using Streamlit")

# Define directories for saving models and results
base_directory = os.path.dirname(__file__)
models_directory = os.path.join(base_directory, "models")
results_directory = os.path.join(base_directory, "results")

# Ensure directories exist
os.makedirs(models_directory, exist_ok=True)
os.makedirs(results_directory, exist_ok=True)

# Step 1: Upload a Dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

data = None
dataset_name = None

# Check if a new file has been uploaded
if uploaded_file is not None:
    # Check if this is a new file or different from the previous one
    if st.session_state.previous_file is None or st.session_state.previous_file != uploaded_file.name:
        # Reset session state for a fresh start
        st.session_state.model_metrics = {}
        st.session_state.model_paths = {}
        st.session_state.previous_file = uploaded_file.name

    data = pd.read_csv(uploaded_file)
    dataset_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]  # Extract dataset name without extension
    st.write("Dataset Preview:", data.head())

    # Step 2: Configure Model
    columns = data.columns.tolist()
    target_column = st.selectbox("Select the target column", columns)
    drop_columns = st.multiselect("Select columns to drop", columns)
    ordinal_columns = st.multiselect("Select ordinal columns", columns)

    # Apply changes (dropping columns)
    data = data.drop(columns=drop_columns)
    st.write("Updated Dataset:", data.head())

    # Apply Filtering Conditions
    st.subheader("Filter Dataset")

    # Create a dictionary to store filter conditions
    filters = {}

    # Filter columns
    filter_columns = st.multiselect("Select columns to filter", options=data.columns.tolist(), default=[])

    for column in filter_columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            min_val, max_val = st.slider(f"Filter {column} values", float(data[column].min()),
                                         float(data[column].max()),
                                         (float(data[column].min()), float(data[column].max())))
            filters[column] = (min_val, max_val)
        else:
            unique_values = data[column].unique()
            selected_values = st.multiselect(f"Filter {column} values", unique_values.tolist(),
                                             default=unique_values.tolist())
            filters[column] = selected_values

    # Apply filters only to selected columns
    filtered_data = data.copy()
    for column, condition in filters.items():
        if pd.api.types.is_numeric_dtype(filtered_data[column]):
            filtered_data = filtered_data[
                (filtered_data[column] >= condition[0]) & (filtered_data[column] <= condition[1])]
        else:
            filtered_data = filtered_data[filtered_data[column].isin(condition)]

    st.write("Filtered Dataset Preview:", filtered_data.head())

    # Step 4: Handle Missing Data
    fill_strategy = st.selectbox("Choose missing value fill strategy", ["mean", "median", "mode"])
    filtered_data = fill_missing_values(filtered_data, strategy=fill_strategy)
    st.write("Data after filling missing values:", filtered_data.head())

    # Encode categorical features option
    # Streamlit interface for encoding options
    st.subheader("Categorical Feature Encoding")

    # Option to encode categorical features
    encode_features = st.checkbox("Encode Categorical Features")

    if encode_features:
        # Option to choose encoding method
        encoding_method = st.selectbox("Select Encoding Method", ["One-Hot Encoding", "Label Encoding"])

        if encoding_method == "One-Hot Encoding":
            filtered_data = one_hot_encode(filtered_data)
            st.write("Data after one-hot encoding:", filtered_data.head())
        elif encoding_method == "Label Encoding":
            filtered_data = label_encode(filtered_data)
            st.write("Data after label encoding:", filtered_data.head())
    else:
        st.write("No encoding applied to categorical features.")

    # Step 5: Split Dataset
    train_size = st.slider("Select training set size (%)", 50, 90, 80) / 100
    X_train, X_test, y_train, y_test = split_data(filtered_data, target_column, test_size=1 - train_size)
    st.write("Training Set Size:", len(X_train))
    st.write("Testing Set Size:", len(X_test))

    # Step 6: Model Type Selection
    model_type = st.selectbox("Select Task Type", ["Classification", "Regression"]).lower()  # Convert to lowercase

    # Step 7: Choose Models
    model_options = {
        "classification": {
            "Random Forest": RandomForestClassifier,
            "Logistic Regression": LogisticRegression,
            "Support Vector Classifier": SVC,
            "XGBoost Classifier": XGBClassifier
        },
        "regression": {
            "Random Forest Regressor": RandomForestRegressor,
            "Linear Regression": LinearRegression,
            "Support Vector Regressor": SVR,
            "XGBoost Regressor": XGBRegressor
        }
    }

    selected_model_name = st.selectbox("Select a model", list(model_options[model_type].keys()))
    model_class = model_options[model_type][selected_model_name]

    # Hyperparameter tuning section

    user_params = {}  # Initialize user_params here

    if st.checkbox("Enable Hyperparameter Tuning"):
        if selected_model_name in hyperparameter_grids:
            st.subheader("Hyperparameter Tuning")
            params = hyperparameter_grids[selected_model_name]
            user_params = {}

            for param, values in params.items():
                if isinstance(values[0], (int, float)):
                    # Ensure consistent slider values
                    min_val = min(values)
                    max_val = max(values)

                    # Adjust types to be consistent
                    if isinstance(min_val, float) or isinstance(max_val, float):
                        min_val = float(min_val)
                        max_val = float(max_val)
                        step_val = (max_val - min_val) / 10.0  # Float step
                    else:
                        min_val = int(min_val)
                        max_val = int(max_val)
                        step_val = max(1, (max_val - min_val) // 10)  # Integer step

                    user_params[param] = st.slider(
                        f"{param}",
                        min_value=min_val,
                        max_value=max_val,
                        value=min_val,
                        step=step_val
                    )
                else:
                    # For categorical parameters, use a select box
                    user_params[param] = st.selectbox(f"{param}", options=values)
        else:
            st.write("No hyperparameters available for tuning with this model.")

    # Model training and hyperparameter tuning
    if st.button("Train Model"):
        if hyperparameter_grids[selected_model_name]:
            param_grid = {k: [v] for k, v in user_params.items()}
            trained_model = hyperparameter_tuning(model_class, param_grid, filtered_data, target_column)
        else:
            model = model_class()
            trained_model = train_model(model, X_train, y_train)

        # Evaluate Model
        if model_type == "classification":
            metrics = evaluate_classification_model(trained_model, X_test, y_test)
            st.write("Confusion Matrix:")
            st.write(metrics["Confusion Matrix"])

            st.write("### Classification Report")
            st.dataframe(metrics["Classification Report"].style.highlight_max(axis=0))

            st.write("Accuracy:", metrics["Accuracy"])

            # Extract only the relevant metrics for comparison
            classification_report_df = metrics["Classification Report"]
            accuracy = metrics["Accuracy"]
            precision = classification_report_df.loc["weighted avg", "precision"]
            recall = classification_report_df.loc["weighted avg", "recall"]
            f1_score = classification_report_df.loc["weighted avg", "f1-score"]

            # Store these metrics for comparison
            metrics_flat = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1_score
            }
            st.session_state.model_metrics[selected_model_name] = metrics_flat
            st.session_state.model_paths[selected_model_name] = os.path.join(models_directory,
                                                                            f"{dataset_name}_{selected_model_name}_model.pkl")

        else:
            metrics = evaluate_regression_model(trained_model, X_test, y_test)
            st.write("Mean Squared Error:", metrics["Mean Squared Error"])
            st.write("R2 Score:", metrics["R2 Score"])

            # Convert metrics to a simple format for saving and comparison
            metrics_flat = {
                "Mean Squared Error": metrics["Mean Squared Error"],
                "R2 Score": metrics["R2 Score"]
            }
            st.session_state.model_metrics[selected_model_name] = metrics_flat
            st.session_state.model_paths[selected_model_name] = os.path.join(models_directory,
                                                                            f"{dataset_name}_{selected_model_name}_model.pkl")

        # Save the trained model to a file in the models directory
        model_filename = st.session_state.model_paths[selected_model_name]
        joblib.dump(trained_model, model_filename)
        st.write(f"Model saved to {model_filename}")

        # Convert the trained model to a binary stream
        model_io = io.BytesIO()
        joblib.dump(trained_model, model_io)
        model_io.seek(0)  # Rewind the file-like object

        # Provide a download button for the model file
        st.download_button(
            label="Download Model",
            data=model_io,
            file_name=f"{dataset_name}_{selected_model_name}_model.pkl",
            mime="application/octet-stream"
        )

    # Step 10: Compare Models
    if st.session_state.model_metrics:
        st.subheader("Compare Models")

        # Convert metrics to a DataFrame for comparison
        # Only display accuracy, precision, recall, and F1 score for classification models
        metrics_df = pd.DataFrame(st.session_state.model_metrics).T
        metrics_df.index.name = 'Model'
        st.dataframe(metrics_df)

        # Step 11: Select and Save the Best Model
        best_model_name = st.selectbox("Select the best model based on metrics", options=st.session_state.model_metrics.keys())

        if st.button("Save Selected Model"):
            if best_model_name:
                # Delete all other models
                for model_name, model_path in st.session_state.model_paths.items():
                    if model_name != best_model_name:
                        os.remove(model_path)
                        st.write(f"Deleted model: {model_path}")

                # Save the best model
                best_model_path = st.session_state.model_paths[best_model_name]
                st.write(f"Best model saved to {best_model_path}")

                # Provide a download button for the best model
                with open(best_model_path, 'rb') as file:
                    st.download_button(
                        label="Download Best Model",
                        data=file,
                        file_name=os.path.basename(best_model_path),
                        mime="application/octet-stream"
                    )
            else:
                st.warning("No model selected for saving.")
    else:
        st.warning("Please upload a CSV file to start the AutoML process.")
