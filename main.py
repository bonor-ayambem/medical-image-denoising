### install necessary packages in CLI before running this file
# pip install numpy
# pip install nibabel
# pip install nipy -q
# pip install nilearn -q

import numpy as np
import nibabel as nib
import nilearn as nilearn
import os

from nilearn import plotting
from nilearn import image
from scipy import ndimage

# read .mnc format data 
img1 = nib.load('original_mnc_files/t1_clean.mnc')
img2 = nib.load('original_mnc_files/t2_clean.mnc')
img3 = nib.load('original_mnc_files/pd_clean.mnc')
img4 = nib.load('original_mnc_files/t1_unclean_3.mnc')
img5 = nib.load('original_mnc_files/t2_unclean_3.mnc')
img6 = nib.load('original_mnc_files/pd_unclean_3.mnc')
img7 = nib.load('original_mnc_files/t1_unclean_9.mnc')
img8 = nib.load('original_mnc_files/t2_unclean_9.mnc')
img9 = nib.load('original_mnc_files/pd_unclean_9.mnc')

# directly save images
os.makedirs('clean_images', exist_ok=True)
os.makedirs('unclean_images', exist_ok=True)
nib.save(img1, "clean_images/t1_clean.nii")
nib.save(img2, "clean_images/t2_clean.nii")
nib.save(img3, "clean_images/pd_clean.nii")
nib.save(img4, "unclean_images/t1_unclean_3.nii")
nib.save(img5, "unclean_images/t2_unclean_3.nii")
nib.save(img6, "unclean_images/pd_unclean_3.nii")
nib.save(img7, "unclean_images/t1_unclean_9.nii")
nib.save(img8, "unclean_images/t2_unclean_9.nii")
nib.save(img9, "unclean_images/pd_unclean_9.nii")

# load data with file path to nii image
niimg1 = nib.load("clean_images/t1_clean.nii")
niimg2 = nib.load("clean_images/t2_clean.nii")
niimg3 = nib.load("clean_images/pd_clean.nii")
niimg4 = nib.load("unclean_images/t1_unclean_3.nii")
niimg5 = nib.load("unclean_images/t2_unclean_3.nii")
niimg6 = nib.load("unclean_images/pd_unclean_3.nii")
niimg7 = nib.load("unclean_images/t1_unclean_9.nii")
niimg8 = nib.load("unclean_images/t2_unclean_9.nii")
niimg9 = nib.load("unclean_images/pd_unclean_9.nii")

os.makedirs('visual_clean_images', exist_ok=True)
os.makedirs('visual_unclean_images', exist_ok=True)
os.makedirs('visual_processed', exist_ok=True)

# visualize images
# clean images
plotting.plot_anat(niimg1, threshold = 3, title="T1 Clean", output_file="visual_clean_images/t1_clean.png")
plotting.plot_anat(niimg2, threshold = 3, title="T2 Clean", output_file="visual_clean_images/t2_clean.png")
plotting.plot_anat(niimg3, threshold = 3, title="PD Clean", output_file="visual_clean_images/pd_clean.png")
# 3% noise
plotting.plot_anat(niimg4, threshold = 3, title="Original T1 3% Noise", output_file="visual_unclean_images/t1_unclean_3.png")
plotting.plot_anat(niimg5, threshold = 3, title="Original T2 3% Noise", output_file="visual_unclean_images/t2_unclean_3.png")
plotting.plot_anat(niimg6, threshold = 3, title="Original PD 3% Noise", output_file="visual_unclean_images/pd_unclean_3.png")
# 9% noise
plotting.plot_anat(niimg7, threshold = 3, title="Original T1 9% Noise", output_file="visual_unclean_images/t1_unclean_9.png")
plotting.plot_anat(niimg8, threshold = 3, title="Original T2 9% Noise", output_file="visual_unclean_images/t2_unclean_9.png")
plotting.plot_anat(niimg9, threshold = 3, title="Original PD 9% Noise", output_file="visual_unclean_images/pd_unclean_9.png")


# perform gaussian smoothing on unclean images and visualize results
# noise = 3%
smoothed_t1_noise3 = image.smooth_img(niimg4, 3) 
plotting.plot_anat(smoothed_t1_noise3, title="T1 3%% Noise Smoothing %imm" % 3, output_file="visual_processed/t1_processed_3.png")
smoothed_t2_noise3 = image.smooth_img(niimg5, 3)
plotting.plot_anat(smoothed_t2_noise3, title="T2 3%% Noise Smoothing %imm" % 3, output_file="visual_processed/t2_processed_3.png")
smoothed_pd_noise3 = image.smooth_img(niimg6, 3)
plotting.plot_anat(smoothed_pd_noise3, title="PD 3%% Noise Smoothing %imm" % 3, output_file="visual_processed/pd_processed_3.png")
# noise = 9%
smoothed_t1_noise9 = image.smooth_img(niimg7, 3) 
plotting.plot_anat(smoothed_t1_noise3, title="T1 9%% Noise Smoothing %imm" % 3, output_file="visual_processed/t1_processed_9.png")
smoothed_t2_noise9 = image.smooth_img(niimg8, 3)
plotting.plot_anat(smoothed_t2_noise3, title="T2 9%% Noise Smoothing %imm" % 3, output_file="visual_processed/t2_processed_9.png")
smoothed_pd_noise9 = image.smooth_img(niimg9, 3)
plotting.plot_anat(smoothed_pd_noise3, title="PD 9%% Noise Smoothing %imm" % 3, output_file="visual_processed/pd_processed_9.png")


# calculate CV
# CV for clean images
clean_t1_data = niimg1.get_fdata()
clean_t1_std = ndimage.standard_deviation(clean_t1_data)
clean_t1_mean = ndimage.mean(clean_t1_data)
print('For original clean t1 image: mean = ', clean_t1_mean, ' std = ', clean_t1_std, ' and CV = ', clean_t1_std/clean_t1_mean)
clean_t2_data = niimg2.get_fdata()
clean_t2_std = ndimage.standard_deviation(clean_t2_data)
clean_t2_mean = ndimage.mean(clean_t2_data)
print('For original clean t2 image: mean = ', clean_t2_mean, ' std = ', clean_t2_std, ' and CV = ', clean_t2_std/clean_t2_mean)
clean_pd_data = niimg3.get_fdata()
clean_pd_std = ndimage.standard_deviation(clean_pd_data)
clean_pd_mean = ndimage.mean(clean_pd_data)
print('For original clean pd image: mean = ', clean_pd_mean, ' std = ', clean_pd_std, ' and CV = ', clean_pd_std/clean_pd_mean)

# CV for original 3% noise 
noise3_t1_data = niimg4.get_fdata()
noise3_t1_std = ndimage.standard_deviation(noise3_t1_data)
noise3_t1_mean = ndimage.mean(noise3_t1_data)
print('For original 3% noise t1 image: mean = ', noise3_t1_mean, ' std = ', noise3_t1_std, ' and CV = ', noise3_t1_std/noise3_t1_mean)
noise3_t2_data = niimg5.get_fdata()
noise3_t2_std = ndimage.standard_deviation(noise3_t2_data)
noise3_t2_mean = ndimage.mean(noise3_t2_data)
print('For original 3% noise t2 image: mean = ', noise3_t2_mean, ' std = ', noise3_t2_std, ' and CV = ', noise3_t2_std/noise3_t2_mean)
noise3_pd_data = niimg6.get_fdata()
noise3_pd_std = ndimage.standard_deviation(noise3_pd_data)
noise3_pd_mean = ndimage.mean(noise3_pd_data)
print('For original 3% noise pd image: mean = ', noise3_pd_mean, ' std = ', noise3_pd_std, ' and CV = ', noise3_pd_std/noise3_pd_mean)
# CV for processed 3% noise
processed_t1_data_3 = smoothed_t1_noise3.get_fdata()
processed_t1_std_3 = ndimage.standard_deviation(processed_t1_data_3)
processed_t1_mean_3 = ndimage.mean(processed_t1_data_3)
print('For processed 3% noise t1 image: mean = ', processed_t1_mean_3, ' std = ', processed_t1_std_3, ' and CV = ', processed_t1_std_3/processed_t1_mean_3)
processed_t2_data_3 = smoothed_t2_noise3.get_fdata()
processed_t2_std_3 = ndimage.standard_deviation(processed_t2_data_3)
processed_t2_mean_3 = ndimage.mean(processed_t2_data_3)
print('For processed 3% noise t2 image: mean = ', processed_t2_mean_3, ' std = ', processed_t2_std_3, ' and CV = ', processed_t2_std_3/processed_t2_mean_3)
processed_pd_data_3 = smoothed_pd_noise3.get_fdata()
processed_pd_std_3 = ndimage.standard_deviation(processed_pd_data_3)
processed_pd_mean_3 = ndimage.mean(processed_pd_data_3)
print('For processed 3% noise pd image: mean = ', processed_pd_mean_3, ' std = ', processed_pd_std_3, ' and CV = ', processed_pd_std_3/processed_pd_mean_3)

# CV for original 9% noise 
noise9_t1_data = niimg7.get_fdata()
noise9_t1_std = ndimage.standard_deviation(noise9_t1_data)
noise9_t1_mean = ndimage.mean(noise9_t1_data)
print('For original 9% noise t1 image: mean = ', noise9_t1_mean, ' std = ', noise9_t1_std, ' and CV = ', noise9_t1_std/noise9_t1_mean)
noise9_t2_data = niimg8.get_fdata()
noise9_t2_std = ndimage.standard_deviation(noise9_t2_data)
noise9_t2_mean = ndimage.mean(noise9_t2_data)
print('For original 9% noise t2 image: mean = ', noise9_t2_mean, ' std = ', noise9_t2_std, ' and CV = ', noise9_t2_std/noise9_t2_mean)
noise9_pd_data = niimg9.get_fdata()
noise9_pd_std = ndimage.standard_deviation(noise9_pd_data)
noise9_pd_mean = ndimage.mean(noise9_pd_data)
print('For original 9% noise pd image: mean = ', noise9_pd_mean, ' std = ', noise9_pd_std, ' and CV = ', noise9_pd_std/noise9_pd_mean)
# CV for processed 9% noise
processed_t1_data_9 = smoothed_t1_noise3.get_fdata()
processed_t1_std_9 = ndimage.standard_deviation(processed_t1_data_9)
processed_t1_mean_9 = ndimage.mean(processed_t1_data_9)
print('For processed 9% noise t1 image: mean = ', processed_t1_mean_9, ' std = ', processed_t1_std_9, ' and CV = ', processed_t1_std_9/processed_t1_mean_9)
processed_t2_data_9 = smoothed_t2_noise9.get_fdata()
processed_t2_std_9 = ndimage.standard_deviation(processed_t2_data_9)
processed_t2_mean_9 = ndimage.mean(processed_t2_data_9)
print('For processed 9% noise t2 image: mean = ', processed_t2_mean_9, ' std = ', processed_t2_std_9, ' and CV = ', processed_t2_std_9/processed_t2_mean_9)
processed_pd_data_9 = smoothed_pd_noise9.get_fdata()
processed_pd_std_9 = ndimage.standard_deviation(processed_pd_data_9)
processed_pd_mean_9 = ndimage.mean(processed_pd_data_9)
print('For processed 9% noise pd image: mean = ', processed_pd_mean_9, ' std = ', processed_pd_std_9, ' and CV = ', processed_pd_std_9/processed_pd_mean_9)