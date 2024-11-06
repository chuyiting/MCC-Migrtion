import kagglehub

target_directory = 'data/'

# Download latest version
path = kagglehub.dataset_download("chenxaoyu/modelnet-normal-resampled", path=target_directory)

print("Path to dataset files:", path)