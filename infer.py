import torch
from model.conditional_unet2d import ConditionalUnet2D
from model.rectified_flow import RectifiedFlow
import cv2
import os
import numpy as np

def infer(
        checkpoint_path,
        base_channels
)