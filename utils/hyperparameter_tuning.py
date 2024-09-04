from sklearn.model_selection import GridSearchCV


def hyperparameter_tuning(model_class, param_grid, train_data, target_column):
    """
    Perform hyperparameter tuning using GridSearchCV.

    :param model_class: The model class to be used (e.g., RandomForestClassifier).
    :param param_grid: Dictionary with parameters names as keys and lists of parameter settings to try as values.
    :param train_data: DataFrame containing the training data.
    :param target_column: The name of the target column.
    :return: The best estimator found by GridSearchCV.
    """
    model = model_class()
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
