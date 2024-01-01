from dataset import SinogramDataset
import numpy as np
import torch
from pathlib import Path
from helper import save_to_tiff_stack, load_tiff_stack_with_metadata
from torch_radon import RadonFanbeam
import torch.nn.functional as F
from scipy.ndimage import zoom
import sys

"""
    Calculates the interpolated images for network evaluation. This includes removing incomplete slices, subsampling the images and interpolating and reconstructing them.
    Usage: "python interpolate.py patient_id subsampling_factor"; e.g. python interpolate.py 'L277' 2
"""


metadata = None
patient_id = sys.argv[1]
subsampling_factor = sys.argv[1]

def load_data(patient_id):
    global metadata
    # load data
    # TODO: change path
    data_path = 'projections/{}_my_sinogram_file'.format(patient_id)
    data, meta = load_tiff_stack_with_metadata(Path('{}.tif'.format(data_path)))
    data = np.array(data)
    data = data[::subsampling_factor]
    data = zoom(data, (subsampling_factor,1,1), order=3)
    metadata = meta
    return data

    
def reconstruct_full_set(prj_0, metadata, path):   
        idx_slice = 18
        image_size = 512
        voxel_size = 0.7  # [mm]
        filter_name = "hann"
        reco = []
        for i in range(prj_0.shape[2]):
            prj = np.copy(np.flip(prj_0[:, :, i], axis=1))
            angles = np.array(metadata['angles'])[:metadata['rotview']] + (np.pi / 2)
            vox_scaling = 1 / voxel_size

            radon = RadonFanbeam(image_size,
                                angles,
                                source_distance=vox_scaling * metadata['dso'],
                                det_distance=vox_scaling * metadata['ddo'],
                                det_count=prj.shape[1],
                                det_spacing=vox_scaling * metadata['du'],
                                clip_to_circle=False)
            sinogram = torch.tensor(prj * vox_scaling).cuda()

            with torch.no_grad():
                filtered_sinogram = radon.filter_sinogram(sinogram, filter_name=filter_name)
                fbp = radon.backprojection(filtered_sinogram)
                fbp = fbp.cpu().detach().numpy()
            reco.append(fbp)
        reco = np.array(reco)
        # Scale reconstruction to HU values following the DICOM-CT-PD
        # User Manual Version 3: WaterAttenuationCoefficient description
        fbp_hu = 1000 * ((reco - metadata['hu_factor']) / metadata['hu_factor'])
        # TODO: change name of file
        save_to_tiff_stack(fbp_hu, Path(path + '_interpolated.tif'))


def main():
    data = load_data(patient_id)
    reconstruct_full_set(data, metadata, patient_id)
         


if __name__ == "__main__":
    main()
    


