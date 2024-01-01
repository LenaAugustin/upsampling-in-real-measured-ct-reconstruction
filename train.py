import torch
torch.cuda.empty_cache()
# TODO: select if you want to use U-Net or CNN
#from SinNet_CNN import SinNet
from SinNet_UNet import SinNet
from pytorch_lightning import Trainer
import warnings
warnings.filterwarnings("ignore")

"""
    Call this class if you want to train a model. Usage: "python train.py"
"""


def main():
    # create instance of model. Creates a U-Net if SinNet_U-Net is imported and a CNN if SinNet_CNN is imported
    model = SinNet()
    # create a PyTorch Lightning trainer. Setting sanity_val_steps to zero prevents lightning from performing time-intensive sanity checking
    # TODO: set num_epochs
    trainer = Trainer(max_epochs=50, num_sanity_val_steps=0)
    res = trainer.fit(model) # train model
    torch.onnx.export(model,                                
                  torch.randn(1, 1, 2304, 736),         # shape of model input
                  "U-Net_sub_2.onnx",                   # TODO: change name of saved model
                  input_names = ['input'],              # the model's input names
                  output_names = ['output'])            # the model's output names

if __name__ == "__main__":
    main()


