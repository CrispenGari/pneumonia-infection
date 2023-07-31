import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torchvision import transforms

class_names = ["NORMAL", "PNEUMONIA BACTERIA", "PNEUMONIA VIRAL"]
device = torch.device("cpu")
INPUT_DIM = 96 * 96
OUTPUT_DIM = len(class_names)
dropout = 0.5
mean = std = 0.5
model_path = os.path.join(os.getcwd(), "models/static/pneumonia_mlp.pt")


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 250),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(250, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, output_dim),
        )

    def forward(self, x):
        # x = [batch size, height, width]
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        # x = [batch size, height * width]
        x = self.classifier(x)  # x = [batch_size, output_dim]
        return x


print(" *   LOADING v0 MODEL")
pneumonia_mpl = MLP(INPUT_DIM, OUTPUT_DIM, dropout).to(device)
pneumonia_mpl.load_state_dict(torch.load(model_path, map_location=device))
print("\n *  LOADING v0 MODEL COMPLETE")


def make_prediction(model, image):
    image = image.to(device)
    preds = model(image)
    preds = F.softmax(preds, dim=1).detach().cpu().numpy().squeeze()

    predicted_label = np.argmax(preds)

    all_preds = [
        {
            "label": int(i),
            "class_label": class_names[i],
            "probability": float(np.round(preds[i], 2)),
        }
        for i, _ in enumerate(preds)
    ]

    res = {
        "top_prediction": {
            "label": int(predicted_label),
            "class_label": class_names[predicted_label],
            "probability": float(np.round(preds[predicted_label], 2)),
        },
        "all_predictions": all_preds,
    }
    return res


def preprocess_img(img):
    """
    takes in a pillow image and pre process it
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    preproces_1 = nn.Sequential(transforms.Resize([96, 96]), transforms.Grayscale(1))
    preprocess_2 = nn.Sequential(
        transforms.Normalize(mean=[mean], std=[std], inplace=False)
    )
    img = preprocess_2(transforms.ToTensor()(preproces_1(img)))
    return img
