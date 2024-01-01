# upsampling-in-real-measured-ct-reconstruction
Source code for the paper "Neural Network-Based Sinogram Upsampling in Real Measured CT Reconstruction" published in the proceedings of the BVM conference 2024

Augustin, L., Wagner, F., Thies, M., \& Maier, A. (2024). Neural Network-Based Sinogram Upsampling in Real Measured CT Reconstruction. In Maier, A., Deserno, T. M., Maier-Hein, K. H., Handels, H., Palm, C. \& Tolxdorff, T. (Eds.), *Bildverarbeitung für die Medizin 2024: Proceedings, German Conference on Medical Image Computing, March 10.-12.2024 Erlangen*. Springer Vieweg, Wiesbaden.

# Overview over python files

Code that needs to be changed (e.g. the path to the sinograms) is marked "TODO"

- dataset.py: Dataloader for the network
- helper.py: helper functions like loading sinograms and saving them
- inference.py: pass the test set through an .onnx file of the network
- interpolate.py: subsample and interpolate sinograms. Used for evaluation of the network
- SinNet_CNN.py: Convolutional network. Called by train.py
- SinNet_U-Net.py: U-Net. Called by train.py
- SSIM.py: calculate mean SSIM and PSNR of a set of reconstructed images
- train.py: run this code to train the network
