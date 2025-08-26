import torch
import clip
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

def denormalize_clip(tensor):
    """Desnormaliza tensor do CLIP de volta para [0,1]"""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=tensor.device).view(3, 1, 1)
    return tensor * std + mean

def perturb (img_input, txt_input):

    # clip inference process
    #image = preprocess(img_input).to(device).unsqueeze(0)
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
    alpha = 32/255
    perturbation = alpha * img_input.grad.sign()
    perturbed_image = img_input + perturbation

    # perturbed_image_denorm = denormalize_clip(perturbed_image.squeeze(0))
    # perturbed_image = torch.clamp(perturbed_image_denorm, 0, 1).detach()
    
    #perturbed_image = torch.clamp(perturbed_image, 0, 1).detach()

    return perturbed_image.detach()





