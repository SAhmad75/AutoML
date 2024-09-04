# Athena AutoML

## Overview

Athena AutoML is a web application for building automated machine learning pipelines using Streamlit. It allows users to upload datasets, select and train models, perform hyperparameter tuning, and evaluate model performance.

## Features

- Upload and preprocess datasets
- Select and train various machine learning models
- Choose whether you want to do encoding 
- Perform hyperparameter tuning
- Evaluate and compare model performance
- Download and save trained models

## Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    ```

2. Navigate to the project directory:
    ```bash
    cd AutoML
    ```

3. Set up a virtual environment (optional but recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

4. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Open the local URL in your browser (e.g., `http://localhost:8501`).

## Encoding 

To enable encoding, check the relevant checkbox in the app and configure the required columns.

## Hyperparameter Tuning

To enable hyperparameter tuning, check the relevant checkbox in the app and configure the parameters as needed.



## Contributing

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature-branch
    ```
3. Commit your changes:
    ```bash
    git commit -am "Add new feature"
    ```
4. Push to the branch:
    ```bash
    git push origin feature-branch
    ```
5. Open a pull request on GitHub.

## License

Specify the license under which the project is distributed.

