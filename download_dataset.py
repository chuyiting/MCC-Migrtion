import kagglehub
import os
import shutil

# # Specify the target directory where you want to save the dataset
# target_directory = 'data/'

# # Ensure the target directory exists
# os.makedirs(target_directory, exist_ok=True)

# Download the dataset
path = kagglehub.dataset_download("chenxaoyu/modelnet-normal-resampled")

# Move the downloaded files to the target directory
destination_dir = os.path.expanduser("data")

# Move the downloaded folder to the new location and rename it to 'data'
shutil.move(path, destination_dir)

print(f"Dataset moved and renamed to {destination_dir}")
