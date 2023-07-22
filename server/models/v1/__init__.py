

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torchvision import transforms

device = torch.device("cpu")
class_names = ['NORMAL', 'PNEUMONIA BACTERIA', 'PNEUMONIA VIRAL']
mean = std = .5
OUTPUT_DIM = len(class_names)
model_path = os.path.join(os.getcwd(), "models/static/pneunomia_lenet.pt")


class LeNet(nn.Module):
    def __init__(self, output_dim):
        super(LeNet, self).__init__()
        self.maxpool2d = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.convs = nn.Sequential(
            nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size =5
            ),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size = 5
            ),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, output_dim)
        )
    def forward(self, x):
        # x = [batch size, 1, 32, 32]
        x = self.convs(x)
        # x = [batch_size, 16, 5, 5]
        x = x.view(x.shape[0], -1) # x = [batch size, 16*5*5]
        x = self.classifier(x)
        return x

print(" *   LOADING v1 MODEL")
pneumonia_lenet = LeNet(OUTPUT_DIM).to(device)
pneumonia_lenet.load_state_dict(torch.load(model_path, map_location=device))
print("\n *  LOADING v1 MODEL COMPLETE")

def preprocess_img(img):
    """
    takes in a pillow image and pre process it
    """
    preproces_1 =  nn.Sequential(
    transforms.Resize([32,32]),
    transforms.Grayscale(1)
    )
    preprocess_2 =  nn.Sequential(
        transforms.Normalize(mean=[mean], std=[std], inplace=False)
    )
    img = preprocess_2(transforms.ToTensor()(preproces_1(img)))
    return img


def make_prediction(model, image, device):
    image = torch.unsqueeze(image, 0).to(device)
    preds = model(image)
    preds = F.softmax(preds, dim=1).detach().cpu().numpy().squeeze()

    predicted_label = np.argmax(preds)

    all_preds = [
        {
        'label': int(i),
        'class_label': class_names[i],
        'probability': float(np.round(preds[i], 2)),
        } for i, _ in enumerate(preds)
    ]

    res ={
        'label': int(predicted_label),
        'class_label': class_names[predicted_label],
        'probability': float(np.round(preds[predicted_label], 2)),
        'predictions': all_preds
    }
    return res
