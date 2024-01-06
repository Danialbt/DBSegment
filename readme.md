# Welcome to the DBSegment!

One of the most significant tasks in medical imaging is image segmentation, which extracts target
segments (such as organs, tissues, lesions, etc.) from images so that analysis is made easier. Since
the development of U-Net, a fully automated, end-to-end neural network specifically tailored for
segmentation tasks, deep learning has demonstrated remarkable potential across almost all online
segmentation difficulties. In recent months, nnU-Net, or "no-new-net," a framework directly evolved
from U-Net design, has also seen widespread popularity.

This tool generates 30 deep brain structures segmentation and a brain mask from T1-Weighted MRI. The whole procedure should take ~1 min for one case. For a definition of the resulting labels refer to the paper or the provided ITK labels file labels.txt (Mehri Baniasadi et al 2022).

# What is nnU-Net?
Image datasets are enormously diverse: image dimensionality (2D, 3D), modalities/input channels (RGB image, CT, MRI, microscopy, ...), 
image sizes, voxel sizes, class ratios, target structure properties, and more change substantially between datasets. 
Traditionally, given a new problem, a tailored solution needs to be manually designed and optimized  - a process that 
is prone to errors, not scalable, and where success is overwhelmingly determined by the experimenter's skill. Even 
for experts, this process is anything but simple. There are not only many design choices and data properties that need to 
be considered, but they are also tightly interconnected, rendering reliable manual pipeline optimization all but impossible! 


**nnU-Net is a semantic segmentation method that automatically adapts to a given dataset. It will analyze the provided 
training cases and automatically configure a matching U-Net-based segmentation pipeline. No expertise is required on your 
end! You can simply train the models and use them for your application**.

Upon release, nnU-Net was evaluated on 23 datasets belonging to competitions from the biomedical domain. Despite competing 
with handcrafted solutions for each respective dataset, nnU-Net's fully automated pipeline scored several first places on 
open leaderboards! Since then nnU-Net has stood the test of time: it continues to be used as a baseline and method 
development framework ([9 out of 10 challenge winners at MICCAI 2020](https://arxiv.org/abs/2101.00232) and 5 out of 7 
in MICCAI 2021 built their methods on top of nnU-Net, 
 [It won AMOS2022 with nnU-Net](https://amos22.grand-challenge.org/final-ranking/))!

Please cite the [following paper](https://www.google.com/url?q=https://www.nature.com/articles/s41592-020-01008-z&sa=D&source=docs&ust=1677235958581755&usg=AOvVaw3dWL0SrITLhCJUBiNIHCQO) when using nnU-Net:

    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
    method for deep learning-based biomedical image segmentation. Nature Methods, 18(2), 203-211.


## Aim Of This Study

The goal of this study is to further develop and extend the DBSegment framework previously
developed at Luxembourg Centre for Systems Biomedicine (LCSB) (Mehri Baniasadi et al 2022), a comprehensive tool
for brain segmentation using several MRI modalities such as T1, T1GD, and T2. Its design and
implementation are the main objectives. The main goals are as follows:

**1. Robustness:** To develop a flexible and reliable DBSegment framework that can precisely
segment various brain areas from MRI scans while accommodating changes in image
quality and modality-specific features.

**2. Enable multi-modal processing:** The existing framework worked with an unimodal input
relying solely on T1 MRI imaging. It should be extended to allow multiple inputs of the same
subject at the same time, for example, T1 and T2 MRI, to potentially further boost
segmentation performance.

**3. Performance analysis and improvement:** To boost the DBSegment framework's
performance by methodical experimentation that includes adjusting hyperparameters such
as batch size and assessing various optimization algorithms like Stochastic Gradient
Descent (SGD), Adam, and AvaGrad.

**4. Enable further application areas, in particular:**

  **a. Neurodegenerative Disease Diagnosis:** In clinical applications, brain
segmentation is essential to help identify neurodegenerative diseases such as
Parkinson. Utilizing a high-performance platform, we can effectively perform brain
segmentation and conduct Morphometric Analysis to facilitate early-stage diagnosis
of neurodegenerative diseases by comparing healthy individuals with patients.
Planning the diagnosis and treatment of several neurological disorders might
benefit from it.
  **b. Glioma Segmentation Algorithm:** To develop a specialized algorithm for the
accurate segmentation of brain Tumors, notably gliomas, from MRI data within the
DBSegment framework. This algorithm will make use of deep learning approaches
and cutting-edge image processing techniques.
  **c. Enabling statistical analysis:** To enable thorough analysis of segmented brain
Tumor data, with an emphasis on figuring out how much of the brain tissue has
been infiltrated by the Tumor. This analysis tries to measure the Tumor's geographic
distribution and percentage coverage throughout the brain, assisting in pre-surgery
planning and offering significant insights for medical professionals. By achieving
these objectives, this study aims to contribute significantly to the field of medical
imaging and computational neurology, offering a valuable tool for clinicians to
enhance their understanding of brain Tumors and improve pre-surgical decision-
making processes. The DBSegment framework's versatility and performance
enhancements are expected to provide a valuable resource for a wide range of
medical applications beyond this research.

