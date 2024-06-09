from dataset import *

if __name__ == '__main__':
    # Load the csv file
    # Set the paths
    base_path = '../inputs/'
    overview_path = os.path.join(base_path, 'overview.csv')
    image_path = os.path.join(base_path, 'images')
    mask_path = os.path.join(base_path, 'masks')
    sagittal_view_list = {}
    mask_list = {}


    # load data from the dataset with manufacturer and model
    train_df, validation_df = load_csv_file(overview_path, PHILIPS_HEALTHCARE)

    # drop columns with NaN values
    train_df = drop_nan_columns(train_df)
    validation_df = drop_nan_columns(validation_df)

    print('Train data:', train_df)
    print('Validation data:', validation_df)