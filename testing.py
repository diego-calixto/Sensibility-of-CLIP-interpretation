from PIL import Image
from utils import *
import clip
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
from pertubation import perturb
import sys
import os
import pandas as pd
import analysis

device = "cuda" if torch.cuda.is_available() else "cpu"

img_input = Image.open("dog_and_car.png")
txt_input = "a dog in a car waiting for traffic lights"

# #int_method = ["selfattn", "gradcam", "maskclip", "eclip", "game", "rollout", "surgery", "m2ib", "rise"]
# int_method = ["selfattn", "gradcam", "maskclip", "eclip", "game", "rollout", "rise"]

# for method in int_method:
#     a = simple_hm(method, img_input, txt_input)
#     if a is not None:
#         print(method + " ok")

a, b, c, d = analysis_pertub(img_input, txt_input, "eclip")

print(a, c)

# img_processed = preprocess(img_input).unsqueeze(0)

# img_keepsize = imgprocess_keepsize(img_input).to(device).unsqueeze(0)

# w, h = img_input.size
# resize = Resize((h,w))


# log = {
#         'img_id': [1],
#         'interpretability_method': [1],
#         'label': [1],
#         'similarity_original': [1],
#         'similarity_perturb': [1],
#         'hm_original': [1],
#         'hm_perturbed': [1],
#         'alpha': [1],
#         'clip_model': [1]
#     }

# pd.DataFrame.from_dict(log).to_pickle("E:\projetos\Grad-Eclip\\results")