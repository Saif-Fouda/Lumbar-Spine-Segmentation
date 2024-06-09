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
import torch

# Constants for the dataset

# Manufacturer
SIEMENS = "SIEMENS"
PHILIPS_HEALTHCARE = "Philips Healthcare"
PHILIPS_MEDICAL_SYSTEM = "Philips Medical Systems"

# target spacing
TARGET_SPACING = (1.0, 1.0, 1.0)
# target shape
TARGET_SHAPE = (256, 256, 256)

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

def resample_image_sitk(dataset_path, file_list, target_shape, target_spacing):
    resampled_images = {}
    for i, nii_file in enumerate(file_list):
        if i == 3:
            break
        # get affine matrix
        image = sitk_read_image(os.path.join(dataset_path, nii_file))
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(target_shape)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)
        
        resampled_data = resampler.Execute(image)
        
        # save new nii file
        output_path = f"{nii_file}_sitk_{i}.nii"
        sitk.WriteImage(resampled_data, os.path.join(dataset_path, output_path))

    return resampled_images
def resample_images(dataset_path, file_list, target_shape, target_spacing):
    resampled_images = {}
    for i, nii_file in enumerate(file_list):
        if i == 3:
            break
        # get affine matrix
        image, original_affine = get_affine(os.path.join(dataset_path, nii_file))

        original_spacing = np.sqrt(np.sum(original_affine[:3, :3]**2, axis=0))

        # resample the image
        resampled_data = resample_image(image= image, target_shape= target_shape,
                                         target_spacing= target_spacing, original_affine= original_affine)

        # update the affine matrix
        new_affine = update_affine(original_affine= original_affine,
                                   original_spacing= original_spacing,
                                   target_spacing= target_spacing)

        resampled_image = nib.Nifti1Image(resampled_data, new_affine)
        # save new nii file
        output_path = f"{nii_file}_{i}.nii"
        nib.save(resampled_image, os.path.join(dataset_path, output_path))
        
        # save the resampled image to the dict
        resampled_images[output_path] = resampled_data
        # save_nii(data= resampled_image, affine= new_affine, output_path= os.path.join(dataset_path,output_path))

    return resampled_images

def resample_image(image, target_shape, original_affine, target_spacing, file_ext=".nii"):
    original_spacing = np.sqrt(np.sum(original_affine[:3, :3]**2, axis=0))
    zoom_factors = np.array(target_shape) / np.array(image.shape)
    resampled_data = zoom(image, zoom_factors, order=1)  # Linear interpolation

    # resample using torchio
    # resample_transform = tio.Resample(target_spacing)
    # resampled_image = resample_transform(image)

    # resize_transform = tio.Resize(target_shape)
    # resized_image = resize_transform(resampled_image)
    return resampled_data

def update_affine(original_affine, original_spacing, target_spacing):
    new_affine = original_affine.copy()
    for i in range(3):
        new_affine[i, i] = target_spacing[i]
    return new_affine

def save_nii(data, affine, output_path):
    new_img = nib.Nifti1Image(data, affine)
    nib.save(new_img, output_path)
    print(f"Saved resampled image to {output_path}")

def get_affine(file_path, file_ext=".nii"):
    file_path = file_path + file_ext
    image = nib.load(file_path).get_fdata()
    affine = nib.load(file_path).affine
    return image, affine

def sitk_read_image(file_path, file_ext=".nii"):
    file_path = file_path + file_ext
    image = sitk.ReadImage(file_path)
    return image