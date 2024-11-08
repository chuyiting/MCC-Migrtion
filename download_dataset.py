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
items = os.listdir(path)
folder = os.path.join(path, items[0])

# Move the downloaded folder to the new location and rename it to 'data'
current_directory = os.getcwd()
shutil.move(folder, current_directory)
os.rename(os.path.join(current_directory, os.path.basename(folder)), os.path.join(current_directory, 'data'))

