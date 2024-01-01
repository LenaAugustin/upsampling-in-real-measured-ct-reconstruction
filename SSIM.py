import skimage
from skimage import io, metrics
import numpy as np
import sys

"""
    Compute mean SSIM and mean PSNR for a set of images. Usage: "python SSIM.py patient_id"; e.g. python SSIM.py 'L277'
"""

# store ssim and psnr for each image
ssim = []
psnr = []
patient_id = sys.argv[1]

# Load the images
#TODO adjust image paths
reco = skimage.io.imread(f'out/{patient_id}_my_upsampled_sinogram.tif')
target = skimage.io.imread(f'{patient_id}_my_target.tif')

# remove unnecessary dimensions
reco = np.squeeze(reco)
target = np.squeeze(target)

print(f'patient id: {patient_id}')

# normalize value range to [0.0, 1.0]
val_min = reco.min()
val_range = reco.max() - val_min
reco_psnr = (reco - val_min) / val_range
target_psnr = (target - val_min) / val_range

for i in range(len(reco)):
    # Calculate SSIM
    ssim_value = skimage.metrics.structural_similarity(reco[i], target[i], data_range=1)
    ssim.append(ssim_value)
    # Calculate PSNR
    psnr_value = skimage.metrics.peak_signal_noise_ratio(reco_psnr[i], target_psnr[i])
    psnr.append(psnr_value)

ssim_mean = np.mean(np.array(ssim))
psnr_mean = np.mean(np.array(psnr))
ssim_std = np.std(np.array(ssim))
psnr_std = np.std(np.array(psnr))

# print results
print(f"results: ssim_mean {ssim_mean:.5f}, psnr_mean {psnr_mean:.5f} dB, ssim_std {ssim_std:.5f}, psnr_std {psnr_std:.5f} dB")