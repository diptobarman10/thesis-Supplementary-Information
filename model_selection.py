import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def encode_features(df):
    """Encodes categorical features using OneHotEncoder."""
    categorical_cols = ['political_ideology', 'gender', 'education', 'topic', 'sentiment']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    df = pd.concat([df.drop(categorical_cols, axis=1).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    return df

def prepare_data(interactions):
    """Prepares data for model training and evaluation."""
    interactions_encoded = encode_features(interactions)
    X = interactions_encoded.drop(['user_id', 'content_id', 'susceptible', 'is_misinformation', 'time_step'], axis=1)
    y = interactions_encoded['susceptible']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Trains and evaluates a model, returning accuracy and AUC scores."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    return accuracy, auc

def select_best_model(interactions):
    """Selects the best model based on AUC score."""
    X_train, X_test, y_train, y_test = prepare_data(interactions)
    
    models = {
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LightGBM': LGBMClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(verbose=0, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        accuracy, auc = evaluate_model(model, X_train, X_test, y_train, y_test)
        results[name] = {'accuracy': accuracy, 'auc': auc, 'model': model}
        print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    
    best_model_name = max(results, key=lambda x: results[x]['auc'])
    best_model = results[best_model_name]['model']
    best_auc = results[best_model_name]['auc']
    
    print(f"\nBest model: {best_model_name} with AUC: {best_auc:.4f}")
    
    return best_model

# Example usage:
if __name__ == "__main__":
    # Assuming you have a function to generate interactions
    from pai_simulation_27th_september_draft import generate_user_profiles, generate_content_items, simulate_interactions
    
    num_users = 100
    num_content = 100
    time_steps = 5
    
    user_profiles = generate_user_profiles(num_users)
    content_items = generate_content_items(num_content)
    interactions = simulate_interactions(user_profiles, content_items, time_steps)
    
    best_model = select_best_model(interactions)
    
    # You can now use this best_model in your main simulation