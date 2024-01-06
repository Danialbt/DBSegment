import optparse
import sys
import numpy as np
import nibabel as nib
import os
from nibabel.freesurfer.mghformat import MGHHeader
from scipy.ndimage import affine_transform
from numpy.linalg import inv
import SimpleITK as sitk

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


def resample_image(input, output, out_spacing=[1.0, 1.0, 1.0]):
    is_label = False
    reader = sitk.ImageFileReader()
    reader.SetFileName(input)
    itk_image = reader.Execute();

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def rsp_conform(input, output):
    
    img = nib.load(input)
    h1 = MGHHeader.from_header(img)
    
    original_size = img.shape[:3]
    original_spacing = img.header.get_zooms()
    out_spacing =[1, 1, 1]
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]
    h1.set_data_shape(out_size) 
    h1.set_zooms([1,1,1])

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
   # new_img = resample_image(options.output, options.output, out_spacing=[1.0, 1.0, 1.0])
   # writer = sitk.ImageFileWriter()
   # writer.SetFileName(options.output)
   # writer.Execute(new_img)
    rsp_conform(options.output, options.output)
    sys.exit(0)
