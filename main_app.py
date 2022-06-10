# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


def accuracy(pred, classes):
    _, preds = torch.max(pred, dim=1)
    return torch.tensor(torch.sum(preds == classes).item() / len(preds))


class ClassificationBase(nn.Module):

    def train_step(self, batch):
        imgs, labels = batch
        out = self(imgs)
        loss = F.cross_entropy(out, labels)
        return loss

    def validating_step(self, batch):
        imgs, labels = batch
        out = self(imgs)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"valiLoss": loss.detach(), "accuracy": acc}

    def epoch_data(self, outputs):
        batch_losses = [x["valiLoss"] for x in outputs]
        batch_accuracy = [x["accuracy"] for x in outputs]
        curr_loss = torch.stack(batch_losses).mean()
        curr_accuracy = torch.stack(batch_accuracy).mean()

        return {"valiLoss": curr_loss, "accuracy": curr_accuracy}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], lastLR: {:.5f}, trainLoss: {:.4f}, valiLoss: {:.4f}, accuracy: {:.4f}".format(
            epoch, result['lRates'][-1], result['trainLoss'], result['valiLoss'], result['accuracy']))


def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class ResNet9(ClassificationBase):
    def __init__(self, in_channels, end_classes):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, end_classes))

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.res1(output) + output
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.res2(output) + output
        output = self.classifier(output)
        return output


def getDevice():
    if torch.cuda.is_available:
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def toDevice(data, device):
    if isinstance(data, (list, tuple)):
        return [toDevice(x, device) for x in data]
    return data.to(device, non_blocking=True)


# model = load_model('./plant_disease.h5')

CLASS_NAMES = ['Apple-Apple_scab',
               'Apple-Black_rot',
               'Apple-Cedar_apple_rust',
               'Apple-healthy',
               'Blueberry-healthy',
               'Cherry_(including_sour)-Powdery_mildew',
               'Cherry_(including_sour)-healthy',
               'Corn_(maize)-Cercospora_leaf_spot Gray_leaf_spot',
               'Corn_(maize)-Common_rust_',
               'Corn_(maize)-Northern_Leaf_Blight',
               'Corn_(maize)-healthy',
               'Grape-Black_rot',
               'Grape-Esca_(Black_Measles)',
               'Grape-Leaf_blight_(Isariopsis_Leaf_Spot)',
               'Grape-healthy',
               'Orange-Haunglongbing_(Citrus_greening)',
               'Peach-Bacterial_spot',
               'Peach-healthy',
               'Pepper,_bell-Bacterial_spot',
               'Pepper,_bell-healthy',
               'Potato-Early_blight',
               'Potato-Late_blight',
               'Potato-healthy',
               'Raspberry-healthy',
               'Soybean-healthy',
               'Squash-Powdery_mildew',
               'Strawberry-Leaf_scorch',
               'Strawberry-healthy',
               'Tomato-Bacterial_spot',
               'Tomato-Early_blight',
               'Tomato-Late_blight',
               'Tomato-Leaf_Mold',
               'Tomato-Septoria_leaf_spot',
               'Tomato-Spider_mites Two-spotted_spider_mite',
               'Tomato-Target_Spot',
               'Tomato-Tomato_Yellow_Leaf_Curl_Virus',
               'Tomato-Tomato_mosaic_virus',
               'Tomato-healthy']


model = toDevice(ResNet9(3, len(CLASS_NAMES)), getDevice())
model.load_state_dict(torch.load('plant-disease-model.pth'))
model.eval()


def predict_image(img, model):
    xb = toDevice(img.unsqueeze(0), getDevice())
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return CLASS_NAMES[preds[0].item()]


st.title("Plant Disease Detection")
st.markdown("Upload an image of the plant leaf")


plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict')

if submit:

    if plant_image is not None:

        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)

        opencv_image = cv2.resize(opencv_image, (256, 256))
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    tensor = transform(opencv_image)

    opencv_image.shape = (1, 256, 256, 3)

    # Y_pred = model.predict(opencv_image)
    # result = CLASS_NAMES[np.argmax(Y_pred)]
    result = predict_image(tensor, model)
    st.title(str("This is "+result.split('-')
                 [0] + " leaf with " + result.split('-')[1]))
