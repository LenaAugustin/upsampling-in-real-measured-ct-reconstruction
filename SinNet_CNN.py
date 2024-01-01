import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import SinogramDataset
import numpy as np
import torch
from pathlib import Path
from helper import save_to_tiff_stack, load_tiff_stack_with_metadata
from torch_radon import RadonFanbeam
import torch.nn.functional as F

"""
    Convolutional neural network for sinogram upsampling. This code is called by train.py
"""


# TODO: set subsampling factor
subsampling_factor = 2
loss_fn = torch.nn.MSELoss()
validation_output = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# meta data of validation set for reconstruction
meta_val = None
idx_slice = 18
image_size = 512
voxel_size = 0.7  # [mm]
filter_name = "hann"

class SinNet(pl.LightningModule):
    def __init__(self, D=20, channels_in=1, channels_out=1, C=64):
        super(SinNet, self).__init__()
        self.D = D
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(1, 64, 3, padding=1))
        self.conv.extend([nn.Conv2d(64, 64, 3, padding=1) for _ in range(D)])
        self.conv.append(nn.Conv2d(64, 1, 3, padding=1))

    def forward(self, x):
        D = self.D
        h = F.relu(self.conv[0](x))
        for i in range(D):
            h = F.relu(self.conv[i + 1](h))
        y = self.conv[D+1](h) + x
        return y


    def training_step(self, batch, batch_idx):
        x, y, metadata = batch
        losses = []
        # propagate through the network
        out = self.forward(x.unsqueeze(0))
        out = torch.squeeze(out)
        target = torch.squeeze(y)
        loss = loss_fn(out, target)
        losses.append(loss)
        return torch.stack(losses).mean()
    

    def validation_step(self, batch, batch_idx):
        global validation_output
        x, y, metadata = batch
        # propagate through the network
        out = self.forward(x.unsqueeze(0))
        out = torch.squeeze(out)
        validation_output.append(out)
        return None

    def reconstruct(self, prj_0, metadata):  
        prj_0 = np.array(prj_0.cpu().detach().numpy())
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
                print("filtered sinogram", filtered_sinogram.dtype, angles.dtype)
                fbp = radon.backprojection(filtered_sinogram)
                fbp = fbp.cpu().detach().numpy()
            reco.append(fbp)
        reco = np.array(reco)
        # Scale reconstruction to HU values following the DICOM-CT-PD
        # User Manual Version 3: WaterAttenuationCoefficient description
        fbp_hu = 1000 * ((reco - metadata['hu_factor']) / metadata['hu_factor'])
        # TODO: change name of file
        save_to_tiff_stack(fbp_hu, Path(f'out/CNN_reco_sub_{subsampling_factor}.tif'))


    # stack slices and reconstruct them
    def on_validation_end(self):
        global validation_output
        out = torch.stack(validation_output)
        # back to original shape ([angles, detector positions, num_slices])
        out = out.swapaxes(0, 2)
        out = out.swapaxes(0, 1)
        out = self.reconstruct(out, meta_glob)
        # reset validation output
        validation_output = []
        return None
    
    # TODO: change learning rate and optimiser
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0001)

    def train_dataloader(self):
        # TODO provide list of patient IDs for training
        patient_id_list = ['patient_1', 'patient_2' , '...']
        patient_data_list = []
        patient_target_list = []
        # load data and target and prepare them for dataloader
        for patient in patient_id_list:
            data_path_subsampled = '{}_subsampled_{}'.format(patient, subsampling_factor)
            patient_data, meta = load_tiff_stack_with_metadata(Path('{}.tif'.format(data_path_subsampled)))
            patient_target, meta = load_tiff_stack_with_metadata(Path('projections/{}_my_target_sinogram.tif'.format(patient)))
            # reshape data from shape [angles, detector positions, num_slices] to [num_slices, angles, detector_positions]
            patient_data = patient_data.swapaxes(0,2)
            patient_data = patient_data.swapaxes(1,2)
            patient_target = patient_target.swapaxes(0,2)
            patient_target = patient_target.swapaxes(1,2)
            patient_target_list.append(patient_target)
            patient_data_list.append(patient_data)
        train_dataset = SinogramDataset(patient_data_list, patient_target_list, meta, subsampling_factor)
        return DataLoader(train_dataset, batch_size=1, shuffle=False)

    def val_dataloader(self):
        global meta_val
        # TODO patient ID for validation
        patient_id = ['patient_val']
        patient_data_list = []
        patient_target_list = []
        # load data and target and prepare them for dataloader
        data_path_subsampled = '{}_subsampled_{}'.format(patient, subsampling_factor)
        patient_data, meta = load_tiff_stack_with_metadata(Path('{}.tif'.format(data_path_subsampled)))
        # store meta data of the validation set so it can be reconstructed
        meta_val = meta
        patient_target, meta = load_tiff_stack_with_metadata(Path('projections/{}_my_target_sinogram.tif'.format(patient)))
        # reshape data from shape [angles, detector positions, num_slices] to [num_slices, angles, detector_positions]
        patient_data = patient_data.swapaxes(0,2)
        patient_data = patient_data.swapaxes(1,2)
        patient_target = patient_target.swapaxes(0,2)
        patient_target = patient_target.swapaxes(1,2)
        patient_data_list.append(patient_data)
        patient_target_list.append(patient_target)
        val_dataset = SinogramDataset(patient_data_list, patient_target_list, meta, subsampling_factor)
        return DataLoader(val_dataset, batch_size=1, shuffle=False)





