import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def load_data(sample_path):
    """
    Load dataset

    Parameters:
    sample_path(str): path to the file

    Returns:
    pd.DataFrame 
    """
    return pd.read_csv(sample_path)


def split_data(df, target_col, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    
    Parameters:
    df(Dataframe): dataframe
    target_col(str): col of the dataframe
    test_size(float): % of the test 

    Returns:
    X_train, y_train, X_test, y_test

    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance

    Parameters:
    model(Pipeline): the model create within the pipeline
    X_test(pd.DataFrame): X_test data
    y_test(pd.DataFrame): y_test data

    Returns:
    metrics(dict): The defined metrics for the model passed 
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    }
    
    if y_proba is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_test, y_proba)
    
    return metrics


def create_pipeline(classifier):
    """
    Create preprocessing and modeling pipeline with imputation

    Parameters:
    classifier(sklearn.model): A ML model

    Returns:
    The Pipeline created containing the model 
    """
    
     # 1. Imputación solo para la columna problemática
    preprocessor = ColumnTransformer(
        transformers=[
            ('imputer', SimpleImputer(strategy='median'), ['Bare.nuclei']) # Columna con nulos
        ],
        remainder='passthrough'  # Pasa el resto de columnas sin cambios
    )
    
    # 2. Pipeline completa
    return Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),  # Escala todas las columnas
        ('classifier', classifier)
    ])