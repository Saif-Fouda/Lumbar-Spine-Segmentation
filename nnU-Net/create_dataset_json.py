import json
import os

# Define the base directory
base_dir = '/Users/saiffouda/WorkingSpace_code/DL/Project/Structure/nnUNet_raw/Dataset001_LumbarSpine'

# Define the dataset information
dataset = {
    "name": "Lumbar Spine Segmentation",
    "description": "Segmentation of vertebrae, intervertebral discs, and spinal canal in lumbar spine MRI",
    "reference": "Lumbar SPIDER Challenge",
    "licence": "CC-BY-SA 4.0",
    "release": "0.0",
    "tensorImageSize": "4D",
    "modality": {
        "0": "MRI_T1",
        "1": "MRI_T2"
    },
    "labels": {
        "0": "background",
        "1": "vertebrae",
        "2": "intervertebral discs",
        "3": "spinal canal"
    },
    "channel_names": {
        "0": "T1",
        "1": "T2"
    },
    "numTraining": 10,
    "numTest": 3,
    "file_ending": ".nii.gz",
    "training": [],
    "test": []
}

# Define paths for images and labels
imagesTr_path = os.path.join(base_dir, 'imagesTr')
labelsTr_path = os.path.join(base_dir, 'labelsTr')
imagesTs_path = os.path.join(base_dir, 'imagesTs')
labelsTs_path = os.path.join(base_dir, 'labelsTs')

# Create training entries
for i in range(1, 11):
    case = f"{i:03}"
    training_entry = {
        "image": [
            os.path.join(imagesTr_path, f"BRATS_{case}_0000.nii.gz"),
            os.path.join(imagesTr_path, f"BRATS_{case}_0001.nii.gz")
        ],
        "label": os.path.join(labelsTr_path, f"BRATS_{case}.nii.gz")
    }
    dataset["training"].append(training_entry)

# Create test entries
for i in range(11, 14):
    case = f"{i:03}"
    test_entry = {
        "image": [
            os.path.join(imagesTs_path, f"BRATS_{case}_0000.nii.gz"),
            os.path.join(imagesTs_path, f"BRATS_{case}_0001.nii.gz"),
            os.path.join(imagesTs_path, f"BRATS_{case}_0002.nii.gz")
        ],
        "label": os.path.join(labelsTs_path, f"BRATS_{case}.nii.gz")
    }
    dataset["test"].append(test_entry)

# Save the dataset.json file
with open(os.path.join(base_dir, 'dataset.json'), 'w') as f:
    json.dump(dataset, f, indent=4)

print("dataset.json created successfully.")
