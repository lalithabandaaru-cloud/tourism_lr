from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os
repo_id="LalithaRB/tourism_project_lb"
repo_type="dataset"

#initialize API client
api=HfApi(token=os.getenv("MLOps-tourism"))

#check if repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repository '{repo_id}' already exists.")
except RepositoryNotFoundError:
          print(f"Dataset repository '{repo_id}' not found. Creating new repository.")
          create_repo(repo_id=repo_id, repo_type=repo_type,private=False)
          print(f"Dataset repository '{repo_id}' created successfully.")

# Upload the data folder. Ensure tourism.csv is inside tourism_project/data
# The instructions prior to this cell mentioned uploading tourism.csv into 'tourism_project/data'.
# So, we should upload from 'tourism_project/data' here.
try:
    api.upload_folder(
        folder_path=os.path.abspath("tourism_project/data"),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message="Add initial tourism.csv data"
    )
    print(f"Data uploaded successfully to Hugging Face Hub under '{repo_id}'")
except Exception as e:
    print(f"Error uploading data to Hugging Face Hub: {e}")
