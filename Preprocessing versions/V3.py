import optparse
import sys
import numpy as np
import nibabel as nib
import os
from nibabel.freesurfer.mghformat import MGHHeader
from scipy.ndimage import affine_transform
from numpy.linalg import inv
import glob

def options_parse():
    """
        Command line option parser
        """
    parser = optparse.OptionParser(version='$Id: conform_v3.py, v3 $')
    parser.add_option('--input', '-i', dest='input')
    parser.add_option('--output', '-o', dest='output')
    (fin_options, args) = parser.parse_args()
    return fin_options

def correct_header(input, output):
    img1 = nib.load(input)
    corr_affine = img1.get_qform()
    img1.set_sform(corr_affine)
    img1.update_header()
    img1.set_data_dtype(img1.get_data_dtype())
    nib.save(img1, output)

def correct_num_col(input, output):
    img1 = nib.load(input)
    #my_mr = os.path.join(path_in, case_name)
    #mr = nib.load(my_mr)
    aff = img1.get_qform()
    mr_mat = img1.get_fdata()
    mr_shape = mr_mat.shape
    if len(mr_shape) > 3:
        l0 = mr_mat.shape[0]
        l1 = mr_mat.shape[1]
        l2 = mr_mat.shape[2]
        l3 = mr_mat.shape[3]
        new_mr = mr_mat.reshape(l0,l1,l2)
        new_mr = nib.Nifti1Image(new_mr, affine = aff)
        new_mr.set_data_dtype(img1.get_data_dtype())
        nib.save(new_mr, output)
    #   nib.save(new_mr, output)
    #else:
    #   new_mr = nib.Nifti1Image(mr_mat, affine = aff)
    #   new_mr.to_filename(output)
    #   nib.save(new_mr, output)
def conform_v3(input, output):
    img = nib.load(input)
    h1 = MGHHeader.from_header(img)
    
    #x1, y1, z1=img.shape[:3]
    
    h1.set_data_shape([256, 256, 256])
    
    #sx, sy, sz=img.header.get_zooms()
    h1.set_zooms([1, 1, 1])
    
    h1['Mdc'] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    h1['Pxyz_c'] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]
    
    ras2ras=np.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    vox2vox=inv(h1.get_affine())@ ras2ras @ img.affine
    
    new_data = affine_transform(img.get_fdata(), inv(vox2vox), output_shape=h1.get_data_shape(), order=1)
    new_img = nib.MGHImage(new_data, h1.get_affine(), h1)
    
    new_img.set_data_dtype(img.get_data_dtype()) 
    #new_img.set_data_dtype(np.uint8)
    nib.save(new_img, output)

if __name__ == "__main__":
    # Command Line options are error checking done here
    #export nnUNet_raw_data_base="/work/projects/mri_seg/base_nnu"
    #export nnUNet_preprocessed="/work/projects/mri_seg/base_nnu/nnUNet_preprocessed"
    #export RESULTS_FOLDER="/work/projects/mri_seg/base_nnu/nnUNet_trained_models"
    options = options_parse()
    os.chdir(options.input)
    directory = '/mnt/lscratch/users/dbajgiran/Tumor_T1T2/nnUNet_raw/Dataset004_Mri/labelsTr/'
    path = os.path.join(options.input, directory)
    if not os.path.exists(path):
        os.mkdir(path)
    for file in glob.glob('*.nii.gz'):
        input_img = os.path.join(options.input, file)
        output_img = os.path.join(options.output, file)
        print('Pre-processing: ', file)
        correct_header(input_img, output_img)
        correct_num_col(output_img, output_img)
        conform_v3(output_img, output_img)

   # print('Network Prediction')
    #nUNet_predict -i path -o options.output -t Task033_Mri -m 3d_fullres
    sys.exit(0)
