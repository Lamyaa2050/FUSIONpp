import os
import shutil
import sys
import csv
from collections import defaultdict
import random
import torch
from transformers import LxmertTokenizer, LxmertModel, CLIPProcessor, CLIPModel, CLIPTextModel, CLIPTokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import supervision as sv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from PIL import Image, ImageFilter
import numpy as np
from io import BytesIO
import openai
import base64
import requests
from simple_lama_inpainting import SimpleLama
import zipfile
import ipywidgets as widgets
from IPython.display import display
from accelerate import Accelerator
from tabulate import tabulate
import gc
from concurrent.futures import ThreadPoolExecutor
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import json
from termcolor import colored
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler