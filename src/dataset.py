# Import libraries
import os
import itk
import SimpleITK as sitk
import nibabel as nib
from nibabel.orientations import axcodes2ornt, apply_orientation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import torchio as tio
from torch.utils.data import DataLoader, Dataset
import torch

# Constants for the dataset

# Manufacturer
SIEMENS = "SIEMENS"
PHILIPS_HEALTHCARE = "Philips Healthcare"
PHILIPS_MEDICAL_SYSTEM = "Philips Medical Systems"

# target spacing
# TARGET_SPACING = (0.8, 0.8, 0.8)
# target shape
TARGET_SHAPE = (128, 128, 128)

# Preprocess the data

# Load the data and convert to nii format for easy visualization
# return the image and the array

def load_convert_mha(file_path, file_ext=".mha"):
    # Adding the file extension
    file_path = file_path + file_ext

    # Set the nii file path
    out_file = file_path.replace('.mha', '.nii')
    nii_file = ""

    # Load the image
    if os.path.exists(file_path):
        mha_file = itk.imread(file_path)

    # Convert mha to nii file
    if not os.path.exists(out_file):
        nii_file = itk.imwrite(mha_file, out_file)
    else:
        nii_file = out_file.split("/")[-1]

    # Load the nii file and get the image and array
    nii_array = np.asanyarray(nib.load(out_file).get_fdata())

    # print nii_array.shape
    print("load and convert->",nii_file, nii_array.shape)

    return nii_file, nii_array


def extract_sagittal_view(nii_file,img_array):
    # Load the nii file
    data = nib.load(nii_file).get_fdata()

    # get affine matrix
    affine = nib.load(nii_file).affine
    # print("Affine matrix:", affine)

    # Check current orientation of the image it could be LPS (Left, Posterior, Superior), PIR (Posterior, Inferior, Right)
    current_orientation = nib.aff2axcodes(affine)
    print("Current orientation:", current_orientation)

    # Set the target  orientation RAS (Right, Anterior, Superior)
    target_orientation = ('R', 'A', 'S')

    # Compute the transformation needed to go from the current to the target orientation
    ornt = axcodes2ornt(current_orientation)
    target_ornt = axcodes2ornt(target_orientation)
    transformation_matrix = nib.orientations.ornt_transform(ornt, target_ornt)

    # Apply the transformation to reorient the data
    reoriented_data = apply_orientation(data, transformation_matrix)
    print("Reoriented_data = ",reoriented_data.shape)

    # get the number of slices in x-axis
    mean_x_axis= (reoriented_data.shape[0]-1)//2

    # print the number of slices in x-axis
    # print(img_array.shape[0])

    # get the sagittal view and rotate the image 90 degrees
    sagittal_view = reoriented_data[mean_x_axis, :, :]
    sagittal_view = np.rot90(sagittal_view, k= 1)
    sagittal_view = np.flip(sagittal_view, axis=1)

    # return the shape of the sagittal view
    return sagittal_view

# Find the maximum dimensions of the images and add the sagittal view to the list
# for the demo we will limit the loop for 50 images, Max dimensions for file (58_t2.mha)
# return the maximum dimensions

def find_max_dimensions(dataset_path, file_list):
    max_dim = (0, 0)
    sagittal_view = np.zeros((0, 0))
    sagittal_view_list = {}

    for file_name in file_list:
        # Load the image and convert to nii
        nii_file, nii_array = load_convert_mha(os.path.join(dataset_path, file_name))

        sagittal_view = extract_sagittal_view(nii_file= os.path.join(dataset_path, nii_file), img_array= nii_array)

        # Add the sagittal view to to dict with the filename as the key
        sagittal_view_list[nii_file] = sagittal_view

        print(nii_file, sagittal_view.shape)

        if sagittal_view.shape > max_dim:
            max_dim = sagittal_view.shape

    return max_dim, sagittal_view_list


def read_csv_file(file_path):
    # Read the csv file
    df = pd.read_csv(file_path)
    return df

def filter_data(df, dataset):
    # Filter the df based on the keyword
    df = df[df['Manufacturer'] == dataset]
    return df

def split_data(df):
    # Split the data into train and test
    train_df = df[df['subset'] == "training"]
    test_df = df[df['subset'] == "validation"]
    return train_df, test_df

# load csv file based on manufacturer, filer and split data
def load_csv_file(file_path, dataset):
    # Read the csv file
    df = read_csv_file(file_path)

    # Filter the data based on the dataset
    df = filter_data(df, dataset)

    # Split the data into train and test
    train_df, test_df = split_data(df)

    return train_df, test_df

def drop_nan_columns(df):
    # Drop columns with NaN values
    df = df.dropna(axis=1)
    return df

def resample_image_sitk(dataset_path, file_list, target_shape):
    resampled_images = []
    for i, nii_file in enumerate(file_list):
        if i == 15:
            break
        
        # Set the output path
        output_path = f"{nii_file}_sitk_{i}.nii"
        
        # Check if the file already exists
        if check_file_exists(dataset_path, output_path):
            resampled_images.append(os.path.join(dataset_path, output_path))
        else:
            # Read the image and get the original spacing and size
            image, original_spacing, original_size = sitk_read_image(os.path.join(dataset_path, nii_file))
            # print(f'Original spacing: {original_spacing}')
            # print(f'Original size: {original_size}')
            
            # Calculate new spacing
            new_spacing = [original_spacing[i] * (original_size[i] / target_shape[i]) for i in range(len(target_shape))]
            # print(f'New spacing: {new_spacing}')

            # Center the resampled image
            original_origin = image.GetOrigin()
            original_center = [original_origin[i] + original_spacing[i] * original_size[i] / 2.0 for i in range(3)]
            new_origin = [original_center[i] - new_spacing[i] * target_shape[i] / 2.0 for i in range(3)]
            
            # Create a resampler
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetSize(target_shape)
            resampler.SetOutputOrigin(new_origin)
            resampler.SetInterpolator(sitk.sitkBSpline5)  # Use B-Spline interpolation for high quality

            # Set the default pixel value to zero (background)
            resampler.SetDefaultPixelValue(0)
            
            # Resample the image
            resampled_image = resampler.Execute(image)
            
            # Normalize the voxel values to the range (0, 255)
            normalized_image = normalize_image(resampled_image)
            
            # save new nii file
            sitk.WriteImage(normalized_image, os.path.join(dataset_path, output_path))
            
            # save the file path to the list
            resampled_images.append(os.path.join(dataset_path, output_path))

    return resampled_images

def check_file_exists(dataset_path, output_path):
    if os.path.exists(os.path.join(dataset_path, output_path)):
        return True
    return False
    
def sitk_read_image(file_path, file_ext=".nii"):
    file_path = file_path + file_ext
    # Read the image
    image = sitk.ReadImage(file_path)
    
    # Cast the image to float32
    image = sitk.Cast(image, sitk.sitkFloat32)
    
    # Get the spacing and size of the image
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    return image, original_spacing, original_size

# Normalize the voxel values to the range (0, 255)
def normalize_image(image):
    image_array = sitk.GetArrayFromImage(image)
    
    # Find minimum and maximum values in the image
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    
    # Shift the range to start from 0
    image_array = image_array - min_val
    
    # Scale to the range (0, 255)
    if max_val != min_val:  # Avoid division by zero
        image_array = (image_array / (max_val - min_val)) * 255.0
    
    # Clip values to ensure they are within (0, 255)
    image_array = np.clip(image_array, 0, 255)
    
    # Convert back to SimpleITK image
    normalized_image = sitk.GetImageFromArray(image_array.astype(np.uint8))
    normalized_image.SetSpacing(image.GetSpacing())
    normalized_image.SetOrigin(image.GetOrigin())
    normalized_image.SetDirection(image.GetDirection())
    
    return normalized_image