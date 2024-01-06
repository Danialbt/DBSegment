import os
import numpy as np
import pandas as pd
import nibabel as nib

def mutual_information(hgram):
    """ Mutual information for joint histogram"""
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

# Define the folder path containing the input files
folder_path = '/mnt/lscratch/users/dbajgiran/V3/registeration/'

# Define the file pairs to compare
#file_pairs = [
#    ('pat_0303_0000.nii.gz', 'pat_0303_0001.nii.gz'),
#    ('pat_0304_0000.nii.gz', 'pat_0304_0001.nii.gz'),
#    ('pat_0305_0000.nii.gz', 'pat_0305_0001.nii.gz')
#]
file_pairs = []

for i in range(101, 749):
    if i != 485 and i != 563 and i != 610 and i != 621 and i != 683 and i != 686 and i != 710 and i != 713 and i != 718 and i != 728 and i != 735 and i != 744:
        t1_file = f'pat_{i:04d}_0000.nii.gz'
        t2_file = f'pat_{i:04d}_0001.nii.gz'
        file_pairs.append((t1_file, t2_file))


results = []

for t1_file, t2_file in file_pairs:
    # Construct the file paths
    t1_path = os.path.join(folder_path, t1_file)
    t2_path = os.path.join(folder_path, t2_file)

    # Load the NIfTI files
    t1_img = nib.load(t1_path)
    t2_img = nib.load(t2_path)

    t1_data = t1_img.get_fdata()
    t2_data = t2_img.get_fdata()

    t1_slice = t1_data[:, :, 101]
    t2_slice = t2_data[:, :, 101]

    hist_2d, x_edges, y_edges = np.histogram2d(t1_slice.ravel(), t2_slice.ravel(), bins=50)

    mutual_info = mutual_information(hist_2d)

    # Append the results to the list
    results.append({'T1 Image': t1_file, 'T2 Image': t2_file, 'Mutual Information': mutual_info})

# Create a DataFrame from the results list
results_df = pd.DataFrame(results)

# Save the results DataFrame to an Excel file
results_df.to_excel('mutual_information_results.xlsx', index=False)
