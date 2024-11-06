import kagglehub
import os

# Specify the target directory where you want to save the dataset
target_directory = 'data/'

# Ensure the target directory exists
os.makedirs(target_directory, exist_ok=True)

# Download the dataset
path = kagglehub.dataset_download("chenxaoyu/modelnet-normal-resampled")

# Move the downloaded files to the target directory
os.rename(path, target_directory)
