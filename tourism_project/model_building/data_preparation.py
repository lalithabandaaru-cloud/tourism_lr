import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from huggingface_hub import HfApi
import joblib

# Ensure necessary directories exist for processed data and preprocessor
os.makedirs("tourism_project/model_building/data_prepared", exist_ok=True)
os.makedirs("tourism_project/model_building", exist_ok=True) # Ensure this directory exists for preprocessor

api = HfApi(token=os.getenv("MLOps-tourism"))
DATASET = "hf://datasets/LalithaRB/tourism_project_lb/tourism.csv"
df = pd.read_csv(DATASET)
print("Dataset loaded successfully")

# Separate features and target
# Dropping 'CustomerID' as it's an identifier and not a predictive feature
X = df.drop(['ProdTaken', 'CustomerID'], axis=1)
y = df['ProdTaken']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Identify categorical and numerical columns for preprocessing
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Impute missing numerical values with median
    ('scaler', StandardScaler()) # Scale numerical features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Impute missing categorical values with most frequent
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-hot encode categorical features
])

# Create a preprocessor using ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough' # Keep any other columns that were not transformed
)

# Fit and transform the training data, then transform the test data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save the preprocessor
preprocessor_path = "tourism_project/model_building/preprocessor.joblib"
joblib.dump(preprocessor, preprocessor_path)
print(f"Preprocessor saved to: {preprocessor_path}")

# Get feature names after one-hot encoding
feature_names = preprocessor.get_feature_names_out()

# Convert processed arrays back to DataFrames
X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)

# Combine processed features with the target variable
train_df_processed = pd.concat([X_train_processed_df, y_train.reset_index(drop=True)], axis=1)
test_df_processed = pd.concat([X_test_processed_df, y_test.reset_index(drop=True)], axis=1)

# Define paths for saving the prepared data
train_file_path = "tourism_project/model_building/data_prepared/train_processed.csv"
test_file_path = "tourism_project/model_building/data_prepared/test_processed.csv"

# Save the prepared data to CSV files
train_df_processed.to_csv(train_file_path, index=False)
test_df_processed.to_csv(test_file_path, index=False)
print(f"Prepared training data saved to: {train_file_path}")
print(f"Prepared testing data saved to: {test_file_path}")

# Upload prepared data to Hugging Face Hub
repo_id = "LalithaRB/tourism_project_lb"
repo_type = "dataset"
folder_to_upload = "tourism_project/model_building/data_prepared"

# Create a dummy README.md for the dataset if it doesn't exist, which is good practice for HF datasets
readme_path = os.path.join(folder_to_upload, "README.md")
if not os.path.exists(readme_path):
    with open(readme_path, "w") as f:
        f.write("---\n")
        f.write(f"tags:\n- tabular\n- classification\n---\n")
        f.write(f"# Tourism Project Prepared Data\n\n")
        f.write(f"This dataset contains the preprocessed training and testing data for the Tourism Project.\n")

try:
    api.upload_folder(
        folder_path=os.path.abspath(folder_to_upload),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message="Add processed training and testing data"
    )
    print(f"Prepared data uploaded successfully to Hugging Face Hub under '{repo_id}'")
except Exception as e:
    print(f"Error uploading prepared data to Hugging Face Hub: {e}")

# Upload the preprocessor as well
try:
    api.upload_file(
        path_or_fileobj=os.path.abspath(preprocessor_path),
        path_in_repo="preprocessor.joblib", # Name in the HF repo
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message="Add preprocessor.joblib"
    )
    print(f"Preprocessor uploaded successfully to Hugging Face Hub under '{repo_id}'")
except Exception as e:
    print(f"Error uploading preprocessor to Hugging Face Hub: {e}")


print("Data preparation and upload script finished.")
