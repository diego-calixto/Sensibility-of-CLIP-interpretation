import torch
import clip
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

def perturb (img_input, txt_input, epsilon=8/255):
    img_input.requires_grad = True 
    text = clip.tokenize(txt_input).to(device)

    image_features = model.encode_image(img_input)
    image_features = F.normalize(image_features, dim=-1)

    text_features = model.encode_text(text)
    text_features = F.normalize(text_features, dim=-1)

    # similarity matrix
    cosine = (image_features @ text_features.T)

    cosine.backward()

    # pertubation calculation
    perturbation = epsilon * img_input.grad.sign()
    perturbed_image = img_input + perturbation
    perturbed_image = torch.clamp(perturbed_image, 0, 1).detach()

    return perturbed_image





