import os
import tarfile

import wget


def download_physio_data(data_dir, dataset_tar_path, extracted_dir_path):
    """
    Checks if the PhysioNet Challenge 2012 dataset is already downloaded and extracted.
    If not present, it downloads and extracts the dataset. You can change the download path
    or save location by modifying the 'data/' directory path in the code.
    """
    # Ensure the data directory exists, create if it does not
    os.makedirs(data_dir, exist_ok=True)

    # Check if the dataset tar file is already downloaded
    if not os.path.exists(dataset_tar_path):
        # URL to download the dataset
        url = "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download"
        print("Downloading dataset...")
        wget.download(url, out=data_dir)
    else:
        print("Dataset tar file already downloaded.")

    # Check if the dataset is already extracted
    if not os.path.exists(extracted_dir_path):
        print("Extracting dataset...")
        with tarfile.open(dataset_tar_path, "r:gz") as t:
            t.extractall(path=extracted_dir_path)
    else:
        print("Dataset already extracted.")
