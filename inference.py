import onnx
import onnxruntime as ort
import numpy as np
from helper import save_to_tiff_stack, load_tiff_stack_with_metadata
from pathlib import Path
from scipy.ndimage import zoom
import torch 
from torch_radon import RadonFanbeam
import sys

"""
    Uses the .onnx file of a trained model to calculate the model's output for the given test set. Usage "python inference.py patient_id"; e.g. python inference.py 'L145'
"""

outputs = []
patient_id = sys.argv[1]
subsampling_factor = 2 # TODO: set correct scaling factor

def reconstruct_full_set(prj_0, metadata, path):   
        print("patient id", patient_id)
        idx_slice = 18
        image_size = 512
        voxel_size = 0.7  # [mm]
        filter_name = "hann"
        prj_0 = prj_0.squeeze()
        reco = []
        prj_0 = np.swapaxes(prj_0, 1, 2)
        prj_0 = np.swapaxes(prj_0, 0, 2)
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
        save_to_tiff_stack(fbp_hu, Path(path + patient_id + '_U-Net_sub_2_inference.tif'))

# Load the ONNX model
# TODO: change name of onnx file
onnx_model = onnx.load('U-Net_residual_sub_2.onnx')
onnx.checker.check_model(onnx_model)
# Create an ONNX Runtime session
ort_sess = ort.InferenceSession('U-Net_residual_sub_2.onnx')

# load subsampled sinogram
# TODO: adapt path to subsampled sinograms
data_path = '../subsampled/{}_subsampled_{}'.format(patient_id, subsampling_factor)
data, meta = load_tiff_stack_with_metadata(Path('{}.tif'.format(data_path)))

# interpolate sinogram
data = zoom(data, (subsampling_factor,1,1), order=3)

# convert data to correct shape for reco
data = data.swapaxes(0,2)
data = data.swapaxes(1,2)

for slice in data:    
    print(slice.shape)
    slice = np.expand_dims(slice, axis=0)
    slice = np.expand_dims(slice, axis=0)
    # perform inference
    output = ort_sess.run(None, {'input': slice})
    outputs.append(output)

# reconstruct sinograms
# TODO: adapt output path
reco = reconstruct_full_set(np.array(outputs), meta, 'out/')


