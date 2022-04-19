from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.01,
                                       max_features=3, max_depth=3, random_state=0)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X, return_prob=False):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    if return_prob:
        preds = model.predict_proba(X)[:, 1]
    else:
        preds = model.predict(X)

    return preds


def slice_inference(df, X, y, feature, model):
    """ Function for calculating descriptive stats on slices of the dataset."""
    for cls in df[feature].unique():
        idx = (df[feature] == cls)
        idx_preds = inference(model, X[idx], return_prob=False)
        precision, recall, fbeta = compute_model_metrics(y[idx], idx_preds)
        print(f"Feature Class: {cls}")
        print(f"{feature} precision: {precision:.4f}")
        print(f"{feature} recall: {recall:.4f}")
        print(f"{feature} fbeta: {fbeta:.4f}")
        print("----------------")
