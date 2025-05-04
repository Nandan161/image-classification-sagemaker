import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

def Net():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 133)
    )
    return model

def model_fn(model_dir):
    logger.info(f"Loading model from {model_dir}")
    model = Net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f, map_location=torch.device('cpu')))
    model.eval()
    return model

def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    if content_type == JPEG_CONTENT_TYPE:
        return Image.open(io.BytesIO(request_body)).convert("RGB")
    elif content_type == JSON_CONTENT_TYPE:
        request = json.loads(request_body)
        url = request['url']
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content)).convert("RGB")
    else:
        raise Exception(f"Unsupported content type: {content_type}")

def predict_fn(input_object, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(input_object).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs[0], dim=0)
    return probs.tolist()
