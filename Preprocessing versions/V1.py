import optparse
import sys
import numpy as np
import nibabel as nib
import os
from nibabel.freesurfer.mghformat import MGHHeader
from scipy.ndimage import affine_transform
from numpy.linalg import inv

def options_parse():
    """
        Command line option parser
        """
    parser = optparse.OptionParser(version='$Id: conform.py, v1 $')
    parser.add_option('--input', '-i', dest='input')
    parser.add_option('--output', '-o', dest='output')
    (fin_options, args) = parser.parse_args()
    return fin_options

def correct_header(input, output):
    img1 = nib.load(input)
    corr_affine = img1.get_qform()
    img1.set_sform(corr_affine)
    img1.update_header()
    img1.to_filename(output)

def rsp_conform(input, output):
    
    img = nib.load(input)
    h1 = MGHHeader.from_header(img)
    
    x1, y1, z1=img.shape[:3]
    
    h1.set_data_shape([x1, y1, z1])
    
    sx, sy, sz=img.header.get_zooms()
    h1.set_zooms([sx, sy, sz])
    
    h1['Mdc'] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    h1['Pxyz_c'] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]
    
    ras2ras=np.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    vox2vox=inv(h1.get_affine())@ ras2ras @ img.affine
    
    new_data = affine_transform(img.get_fdata(), inv(vox2vox), output_shape=h1.get_data_shape(), order=1)
    new_img = nib.MGHImage(new_data, h1.get_affine(), h1)
    
    new_img.set_data_dtype(np.uint8)
    nib.save(new_img, output)

if __name__ == "__main__":
    # Command Line options are error checking done here
    options = options_parse()
    #print("Reading input: {} ...".format(options.input))
    correct_header(options.input, options.output)
    rsp_conform(options.output, options.output)
    sys.exit(0)
