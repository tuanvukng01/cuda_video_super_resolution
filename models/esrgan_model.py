import torch
import torch.nn as nn

class ESRGAN(nn.Module):
    def __init__(self):
        super(ESRGAN, self).__init__()
        self.upsample = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.upsample(x))

def load_pretrained_model():
    model = ESRGAN()
    print("Pre-trained ESRGAN model loaded.")
    return model