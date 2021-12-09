import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
from torch import nn 
from torch.nn import functional as F
import numpy as np
from torchvision import transforms

device = torch.device("cpu")
INPUT_DIM = 96 * 96
OUTPUT_DIM = 3
dropout = .5
class_names = ['NORMAL', 'PNEUMONIA BACTERIA', 'PNEUMONIA VIRAL']
mean = std = .5
model_path = os.path.join(os.getcwd(), "model/static/chest-x-ray.pt")

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=.5):
        super(MLP, self).__init__()
        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = F.relu(self.input_fc(x))
        x = self.dropout(x)
        x = F.relu(self.hidden_fc(x))
        x = self.dropout(x)
        outputs = self.output_fc(x)
        return outputs, x
        
print(" *   LOADING MODEL")
model = MLP(INPUT_DIM, OUTPUT_DIM, dropout).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
print("\n *   DONE LOADING THE MODEL")

def make_prediction(model, image):
    image = image.to(device)
    preds, _ = model(image)
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
        'predictions': all_preds, 'meta': {
            "programmer": "@crispengari",
            "main": "computer vision (cv)",
            "description": "given a medical chest-x-ray image of a human being we are going to classify weather a person have pneumonia virus, pneumonia bacteria or none of those(normal).",
            "language": "python",
            "library": "pytorch"
         }
    }
    return res

def preprocess_img(img):
    """
    takes in a pillow image and pre process it
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    preproces_1 =  nn.Sequential(
    transforms.Resize([96,96]),
        transforms.Grayscale(1)
    )
    preprocess_2 =  nn.Sequential(
        transforms.Normalize(mean=[mean], std=[std], inplace=False)
    )
    img = preprocess_2(transforms.ToTensor()(preproces_1(img)))
    return img
