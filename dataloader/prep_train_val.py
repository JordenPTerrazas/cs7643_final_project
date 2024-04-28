import os
import json
import random

def create_mapping(clean_dir, noisy_dir):
    """
    Create a mapping from nosied file to clean file (e.g feature to target).
    
    Args: 
        clean_dir (str): Path to clean directory
        noisy_dir (str): Path to noisy directory

    Returns:
        mapping (dict): Mapping from noisy file to clean file 
    """
    clean_list = os.listdir(clean_dir)
    noisy_list = os.listdir(noisy_dir)
    
    mapping = {}
    
    # Ignore the poor time complexity :)
    for noisy in noisy_list:
        noisy_fileid = noisy.split("fileid_")[-1].split(".")[0]
        for clean in clean_list:
            clean_fileid = clean.split("fileid_")[-1].split(".")[0]
            if noisy_fileid == clean_fileid:
                mapping[noisy] = clean
                break
    
    return mapping

if __name__ == "__main__":
    split_perc = 0.8
    dataset_dir = "data/datasets/DNS_subset_10"
    clean_dir = f"{dataset_dir}/clean"
    noisy_dir = f"{dataset_dir}/noisy"
    
    # Create the mapping if none exists
    if os.path.exists(f"{dataset_dir}/data_mapping.json"):
        print("Mapping already exists, loading it in!")

        # Read existing mapping
        with open(f"{dataset_dir}/data_mapping.json", "r") as f:
            data_mapping = json.load(f)
    else:
        print("Creating data mapping! This may take a minute...")
        data_mapping = create_mapping(clean_dir=clean_dir, noisy_dir=noisy_dir)
    
        # Save the data_mapping as a JSON file
        with open(f"{dataset_dir}/data_mapping.json", "w") as f:
            json.dump(data_mapping, f)

    # Split the data into train and val
    data_mapping = list(data_mapping.items())
    random.shuffle(data_mapping)
    split_idx = int(len(data_mapping) * split_perc)
    train_mapping = data_mapping[:split_idx]
    val_mapping = data_mapping[split_idx:]

    # Make the train and val directories
    train_dir = f"{dataset_dir}/train"
    val_dir = f"{dataset_dir}/val"

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        os.makedirs(f"{train_dir}/clean")
        os.makedirs(f"{train_dir}/noisy")

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
        os.makedirs(f"{val_dir}/clean")
        os.makedirs(f"{val_dir}/noisy")

    # Move the files over
    for idx, (noisy, clean) in enumerate(train_mapping):
        os.rename(f"{clean_dir}/{clean}", f"{train_dir}/clean/{clean}")
        os.rename(f"{noisy_dir}/{noisy}", f"{train_dir}/noisy/{noisy}")

    for idx, (noisy, clean) in enumerate(val_mapping):
        os.rename(f"{clean_dir}/{clean}", f"{val_dir}/clean/{clean}")
        os.rename(f"{noisy_dir}/{noisy}", f"{val_dir}/noisy/{noisy}")
    
    # clean_list = os.listdir("data/datasets/DNS_subset_10/clean")
    # noisy_list = os.listdir("data/datasets/DNS_subset_10/noisy")

    # for idx, (clean, noisy) in enumerate(zip(clean_list, noisy_list)):
    #     if idx == 5:
    #         break
    #     #print(clean, noisy)
    #     clean_fileid = clean.split("fileid_")[-1].split(".")[0]
    #     noisy_fileid = noisy.split("fileid_")[-1].split(".")[0]
    #     print(clean_fileid, noisy_fileid)