from __future__ import print_function

import cv2
import numpy as np
import matplotlib.pyplot as plt

from audio_sheet_retrieval.sheet_utils.omr import OpticalMusicRecognizer, prepare_image
from audio_sheet_retrieval.sheet_utils.omr import SegmentationNetwork
from audio_sheet_retrieval.sheet_utils import system_detector, bar_detector

# this is the default height of our staff systems
# (it is fixed as we process it with a convolutional neural network)
SYSTEM_HEIGHT = 160

# path to sheet image file
# sheet_img_path = "sheet_image.png"
sheet_img_path = "testing/Golden Hour.png"

# path to audio file
audio_path = "audio.mp3"

def resize_img(I):
    print(I.shape)
    width = 835
    scale = float(width)/I.shape[1]
    height = int(scale*I.shape[0])
    I = cv2.resize(I, (width, height))
    print(I.shape)
    return I

# laod sheet image
sheet_image = cv2.imread(sheet_img_path, 0)
sheet_image_resized = resize_img(sheet_image)
# cv2.imshow('original', sheet_image)
# cv2.imshow('resized', sheet_image_resized)
cv2.waitKey(0)

# Automatic system detection
# OMR module - initialize with pretrained models
net = system_detector.build_model()
