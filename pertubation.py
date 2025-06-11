import torch
import clip
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

def perturb (img_input, txt_input):

    # clip inference process
    image = preprocess(img_input).unsqueeze(0).to(device)
    image.requires_grad = True 
    text = clip.tokenize(txt_input).to(device)

    image_features = model.encode_image(image)
    image_features = F.normalize(image_features, dim=-1)

    text_features = model.encode_text(text)
    text_features = F.normalize(text_features, dim=-1)

    # similarity matrix
    cosine = (image_features @ text_features.T)

    cosine.backward()

    # pertubation calculation
    alpha = 8/255
    perturbation = alpha * image.grad.sign()
    perturbed_image = image + perturbation
    perturbed_image = torch.clamp(perturbed_image, 0, 1).detach()

    to_pil = ToPILImage()
    perturbed_image_pil = to_pil(perturbed_image.squeeze())

    return perturbed_image_pil





