from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import zoom

"""
    Dataloader for the network. Upsamples the input sinogram and returns network input and corresponding target
"""


class SinogramDataset(Dataset):
    def __init__(self, patient_data, patient_target, metadata, subsampling_factor):
        self.patient_data = patient_data
        self.patient_target = patient_target
        self.metadata = metadata
        self.subsampling_factor = subsampling_factor

    def __len__(self):
        return len(self.patient_data) * patient_data[0].shape[0]  # patient_data[0].shape[0] slices per patient (assuming all sets have the same number of slices)

    def __getitem__(self, idx):
        patient_idx = idx // patient_data[0].shape[0]
        slice_idx = idx % patient_data[0].shape[0]
        slice_data = self.patient_data[patient_idx][slice_idx]
        slice_data= zoom(slice_data, (self.subsampling_factor,1), order=3)
        slice_target = self.patient_target[patient_idx][slice_idx]
        return np.squeeze(slice_data), np.squeeze(slice_target), self.metadata
