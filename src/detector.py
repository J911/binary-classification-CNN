# Jaemin Lee (aka, J911)
# 2019

import cv2
import torch
import numpy as np 
from model import Net

model = Net()
model.eval()

def load_model(path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)

def start_detect():
    cam = cv2.VideoCapture(0)
    cam.set(3, 1280)
    cam.set(4, 720)

    label = ''

    while True:
        f, img = cam.read()
        img = cv2.cvtColor(img, f)

        display_img = cv2.putText(cv2.flip(img, 1),  label, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
        cv2.imshow("Custom Capture", display_img)
        img = np.moveaxis(img, -1, 0)

        input = torch.tensor([img])
        output = model.forward(input)
        result = output.data.max(1, keepdim=True)[1][0][0].item()
        
        if result == 1: 
            label = 'right'
        else:
            label = 'left'

        if cv2.waitKey(1) == 27: # esc
            break

    cam.release()

if __name__ == '__main__':
    load_model('./model_save')
    start_detect() 
