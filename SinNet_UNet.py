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
    U-Net for sinogram upsampling. This code is called by train.py
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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # pool only in angle dimension
            nn.MaxPool2d((2,1)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SinNet(pl.LightningModule):
    def __init__(self, n_channels_in=1, n_channels_out=1, base_num_features=16, bilinear=True):
        super(SinNet, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        
        self.inc = DoubleConv(n_channels_in, base_num_features)
        self.down1 = Down(base_num_features, 2*base_num_features)
        self.down2 = Down(2*base_num_features, 4*base_num_features)
        self.down3 = Down(4*base_num_features, 8*base_num_features)
        self.down4 = Down(8*base_num_features, 16*base_num_features // factor)
        self.up1 = Up(16*base_num_features, 8*base_num_features // factor, bilinear)
        self.up2 = Up(8*base_num_features, 4*base_num_features // factor, bilinear)
        self.up3 = Up(4*base_num_features, 2*base_num_features // factor, bilinear)
        self.up4 = Up(2*base_num_features, base_num_features, bilinear)
        self.outc = OutConv(base_num_features, n_channels_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_ = self.up1(x5, x4)
        x_ = self.up2(x_, x3)
        x_ = self.up3(x_, x2)
        x_ = self.up4(x_, x1)
        logits = self.outc(x_)
        return logits + x


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
                fbp = radon.backprojection(filtered_sinogram)
                fbp = fbp.cpu().detach().numpy()
            reco.append(fbp)
        reco = np.array(reco)
        # Scale reconstruction to HU values following the DICOM-CT-PD
        # User Manual Version 3: WaterAttenuationCoefficient description
        fbp_hu = 1000 * ((reco - metadata['hu_factor']) / metadata['hu_factor'])
        # TODO: change name of file
        save_to_tiff_stack(fbp_hu, Path(f'out/U-Net_reco_sub_{subsampling_factor}.tif'))


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
        return torch.optim.SGD(self.parameters(), lr=0.0005)

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


