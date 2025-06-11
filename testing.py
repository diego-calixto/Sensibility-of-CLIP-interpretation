from PIL import Image
from utils import *
import clip
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
from pertubation import perturb
import sys
import os
import pandas as pd

# img_input = Image.open("dog_and_car.png")
# txt_input = "a dog in a car waiting for traffic lights"

# #int_method = ["selfattn", "gradcam", "maskclip", "eclip", "game", "rollout", "surgery", "m2ib", "rise"]
# int_method = ["selfattn", "gradcam", "maskclip", "eclip", "game", "rollout", "rise"]

# for method in int_method:
#     a = simple_hm(method, img_input, txt_input)
#     if a is not None:
#         print(method + " ok")

#simple_hm("surgery", img_input, txt_input)

log = {
        'img_id': [1],
        'interpretability_method': [1],
        'label': [1],
        'similarity_original': [1],
        'similarity_perturb': [1],
        'hm_original': [1],
        'hm_perturbed': [1],
        'alpha': [1],
        'clip_model': [1]
    }

pd.DataFrame.from_dict(log).to_pickle("E:\projetos\Grad-Eclip\\results")