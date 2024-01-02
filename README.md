# upsampling-in-real-measured-ct-reconstruction
This repository contains the source code for the paper "Neural Network-Based Sinogram Upsampling in Real Measured CT Reconstruction" published in the proceedings of the BVM conference 2024. The project contains code for a residual CNN and a residual U-Net to upsample subsampled sinograms. For reconstructing the sinograms, we use Matteo Ronchetti's [*torch-radon*](https://github.com/matteo-ronchetti/torch-radon). Our code uses some code from [*helix2fan*](https://github.com/faebstn96/helix2fan).

Augustin, L., Wagner, F., Thies, M., \& Maier, A. (2024). Neural Network-Based Sinogram Upsampling in Real Measured CT Reconstruction. In Maier, A., Deserno, T. M., Maier-Hein, K. H., Handels, H., Palm, C. \& Tolxdorff, T. (Eds.), *Bildverarbeitung für die Medizin 2024: Proceedings, German Conference on Medical Image Computing, March 10.-12.2024 Erlangen*. Springer Vieweg, Wiesbaden.



# Overview over Python files

Code that needs to be changed (e.g. the path to the sinograms) is marked "TODO"

- dataset.py: dataloader for the network
- helper.py: helper functions like loading sinograms and saving them
- inference.py: perform inference by passing the test set through an .onnx file of the network
- interpolate.py: subsample and interpolate sinograms. Used for evaluation of the network
- SinNet_CNN.py: CNN. Called by train.py
- SinNet_U-Net.py: U-Net. Called by train.py
- SSIM.py: calculate mean SSIM and PSNR of a set of reconstructed images
- train.py: run this code to train the network





