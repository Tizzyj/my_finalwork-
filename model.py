import torch
import torchvision.transforms as T
from torchvision import models
import numpy as np
import cv2

class FERModel:
    def __init__(self, device="cpu", num_classes=7):
        self.device = device
        # MobileNetV3 large backbone
        self.net = models.mobilenet_v3_large(pretrained=True)
        # replace classifier
        in_features = self.net.classifier[0].in_features if hasattr(self.net.classifier[0], 'in_features') else 1280
        self.net.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features, num_classes)
        )
        self.net.to(self.device)
        self.net.eval()
        self.transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((128,128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def load_checkpoint(self, path):
        ck = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ck["state_dict"])
        print("Loaded checkpoint", path)

    @torch.no_grad()
    def predict_logits(self, crop):
        # crop: BGR numpy image
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        t = self.transforms(img).unsqueeze(0).to(self.device)
        logits = self.net(t).cpu().numpy()[0]
        return logits