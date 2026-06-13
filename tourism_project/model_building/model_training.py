import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib


# Define paths for loading the prepared data
train_file_path = "tourism_project/model_building/data_prepared/train_processed.csv"
test_file_path = "tourism_project/model_building/data_prepared/test_processed.csv"

# Load the prepared data
train_df_processed = pd.read_csv(train_file_path)
test_df_processed = pd.read_csv(test_file_path)

print(f"Prepared training data loaded from: {train_file_path}")
print(f"Prepared testing data loaded from: {test_file_path}")

# Separate features (X) and target (y)
X_train = train_df_processed.drop('ProdTaken', axis=1)
y_train = train_df_processed['ProdTaken']
X_test = test_df_processed.drop('ProdTaken', axis=1)
y_test = test_df_processed['ProdTaken']

print("Data split into features and target.")

# Start an MLflow run
with mlflow.start_run():
    # Define model parameters
    n_estimators = 100
    max_depth = 10
    random_state = 42

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("model_type", "RandomForestClassifier")

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    print(f"Model Evaluation:\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}\nROC-AUC: {roc_auc:.4f}")

    # Log the model
    mlflow.sklearn.log_model(model, "random_forest_model", registered_model_name="TourismProductPredictor")
    print("Model logged and registered with MLflow.")

print("Model training and tracking script finished.")
# save the model locally
model_path="tourism_model_v1.joblib"
joblib.dump(model, model_path)
print("Model saved to:", model_path)

# log the model artifact
mlflow.log_artifact(model_path, "model")
print("Model artifact logged at: {model_path}")

# Upload to Hugging face
repo_id = "LalithaRB/tourism_project_lb"
repo_type = "model"

try:
  api.repo_info(repo_id=repo_id, repo_type=repo_type)
  print(f"Space '{repo_id}' already exists.")
except RepositoryNotFoundError:
          print(f"Space '{repo_id}' not found. Creating new space")
          create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
          print(f"Space '{repo_id}' created successfully.")
api.upload_file(
    path_or_fileobj="tourism_model_v1.joblib",
    path_in_repo="tourism_model_v1.joblib",
    repo_id=repo_id,
    repo_type=repo_type
)
