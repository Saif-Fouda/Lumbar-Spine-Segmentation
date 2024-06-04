# Import libraries
import os
import itk
import SimpleITK as sitk
import nibabel as nib
from nibabel.orientations import axcodes2ornt, apply_orientation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Preprocess the data

# Load the data and convert to nii format for easy visualization
# return the image and the array

def load_convert_mha(file_path):

    # Set the nii file path
    out_file = file_path.replace('.mha', '.nii')
    nii_file = ""
    # Load the image
    mha_file = itk.imread(file_path)

    # Convert mha to nii file
    if not os.path.exists(out_file):
        nii_file = itk.imwrite(mha_file, out_file)
    
    # Load the nii file and get the image and array
    arr = np.asanyarray(nib.load(out_file).get_fdata())

    # print arr.shape
    # print("load and convert->",nii_file, arr.shape)
    return nii_file, arr


def extract_sagittal_view(file_name, nii_file,img_array):
    # Load the nii file
    data = nib.load(nii_file).get_fdata()
    print(file_name)

    # get affine matrix
    affine = nib.load(nii_file).affine
    print("Affine matrix:", affine)

    # Check current orientation of the image it could be LPS (Left, Posterior, Superior), PIR (Posterior, Inferior, Right)
    current_orientation = nib.aff2axcodes(affine)
    # print("Current orientation:", current_orientation)

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

def find_max_dimensions(base_path, dataset):
    max_dim = (0, 0)
    sagittal_view = np.zeros((0, 0))
    sagittal_view_list = {}
    itr = 0

    for filename in os.listdir(os.path.join(base_path, dataset)):
        if itr == 10:
            break
        # Load the image and convert to nii
        img, img_array = load_convert_mha(os.path.join(base_path, dataset, filename))

        if filename.endswith('.nii'):
            itr += 1
            sagittal_view = extract_sagittal_view(file_name = filename, nii_file= os.path.join(base_path, dataset, filename), img_array= img_array)

            # Add the sagittal view to to dict with the filename as the key
            sagittal_view_list[filename] = sagittal_view

            # print(filename, sagittal_view.shape)

        if sagittal_view.shape > max_dim:
            max_dim = sagittal_view.shape
    return max_dim, sagittal_view_list


def pad_image(img_array, max_dim):
    padding = [(0, max(0, max_dim[i] - img_array.shape[i])) for i in range(2)]
    padded_img = np.pad(img_array, pad_width=padding, mode='constant', constant_values=0)
    return padded_img