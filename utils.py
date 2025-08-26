import time
import numpy as np
import torch
import re
import torch.nn.functional as F
import Game_MM_CLIP.clip as mm_clip
import cv2
import clip
from torchvision.transforms import Resize, ToPILImage
import matplotlib.pyplot as plt
from pertubation import perturb

from generate_emap import clipmodel, preprocess, imgprocess_keepsize, mm_clipmodel, mm_interpret, \
        clip_encode_dense, grad_eclip, grad_cam, mask_clip, compute_rollout_attention, \
        surgery_model, clip_surgery_map, m2ib_model, m2ib_clip_map, rise


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption
    
def generate_hm(hm_type, img, txt_embedding, txts, resize, img_keepsized):
    start = time.time()
    emap = 0
    outputs, v_final, last_input, v, q_out, k_out,\
        attn, att_output, map_size = clip_encode_dense(img_keepsized)
    img_embedding = F.normalize(outputs[:,0], dim=-1)
    cosines = (img_embedding @ txt_embedding.T)[0]

    if hm_type == "selfattn":
        emap = attn[0,:1,1:].detach().reshape(*map_size)
    elif "gradcam" in hm_type:
        emap = [grad_cam(c, last_input, map_size) for c in cosines]
        emap = torch.stack(emap, dim=0).sum(0)
    elif "maskclip" in hm_type:
        emap = mask_clip(txt_embedding.T, v_final, k_out, map_size)
        emap = emap.sum(0)
    elif "eclip" in hm_type:
        emap = [grad_eclip(c, q_out, k_out, v, att_output, map_size, withksim=False) \
            if "wo-ksim" in hm_type else grad_eclip(c, q_out, k_out, v, att_output, map_size, withksim=True) \
            for c in cosines]
        emap = torch.stack(emap, dim=0).sum(0)  
    elif "game" in hm_type:
        start = time.time()
        #img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
        text_tokenized = mm_clip.tokenize(txts).to(device)
        emap = mm_interpret(model=mm_clipmodel, image=img, texts=text_tokenized, device=device)    
        emap = emap.sum(0) 
    elif "rollout" in hm_type:
        start = time.time()
        #img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
        text_tokenized = mm_clip.tokenize(txts).to(device)
        attentions = mm_interpret(model=mm_clipmodel, image=img, texts=text_tokenized, device=device, rollout=True)      
        emap = compute_rollout_attention(attentions)[0]
    elif "surgery" in hm_type:
        start = time.time()
        #img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
        all_texts = ['airplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform', 'potted plant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table', 'track', 'train', 'tree', 'truck', 'tv monitor', 'wall', 'water', 'window', 'wood']
        all_texts = txts + all_texts
        emap = clip_surgery_map(model=surgery_model, image=img, texts=all_texts, device=device)[0,:,:,0]
    elif "m2ib" in hm_type:
        start = time.time()
        #img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
        if isinstance(txts, str):
            txts = [txts]
        emap = m2ib_clip_map(model=m2ib_model, image=img, texts=txts, device=device)
        emap = torch.tensor(emap)
    elif "rise" in hm_type:
        start = time.time()
        #img_clipreprocess = preprocess(img).unsqueeze(0)
        emap = rise(model=clipmodel, image=img, txt_embedding=txt_embedding, device=device)
        print(emap.shape)
    else:
        raise ValueError(f"[generate_hm] hm_type '{hm_type}' not recognized")
    end = time.time()
    
    print("processing time: ", end-start)
    
    emap -= emap.min()
    emap /= emap.max()
    emap = resize(emap.unsqueeze(0))[0]
    return emap

def visualize(hmap, raw_image, resize):
    image = np.asarray(raw_image.copy())
    hmap = resize(hmap.unsqueeze(0))[0].cpu().numpy()
    color = cv2.applyColorMap((hmap*255).astype(np.uint8), cv2.COLORMAP_JET) # cv2 to plt
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    c_ret = np.clip(image * (1 - 0.5) + color * 0.5, 0, 255).astype(np.uint8)
    return c_ret



def simple_hm(hm_type, img, txt, resize, img_keepsize, print=False):
    """
    ## hm_types

    ['eclip-wo-ksim_gt', 'eclip-wo-ksim_pred', 'eclip_gt', 'eclip_pred', 'game_gt', 'game_pred', 
    'gradcam_gt', 'gradcam_pred', 'maskclip_gt', 'maskclip_pred', 'selfattn', 'surgery_gt', 'surgery_pred', 'm2ib_gt', 'm2ib_pred']
    """

    text_processed = clip.tokenize(txt).to(device)
    text_embedding = clipmodel.encode_text(text_processed)
    text_embedding = F.normalize(text_embedding, dim=-1)
    
    hm = generate_hm(hm_type, img, text_embedding, [txt], resize, img_keepsize)

    if(print):
        c_ret = visualize(hm, img.copy(), resize)
        plt.imshow(c_ret)
        plt.axis('off')
        plt.show()
    
    return hm

def simple_similarity(img, txt):

    text_processed = clip.tokenize(txt).to(device)
    
    text_embedding = clipmodel.encode_text(text_processed)
    text_embedding = F.normalize(text_embedding, dim=-1)

    ori_img_embedding = clipmodel.encode_image(img)
    ori_img_embedding = F.normalize(ori_img_embedding, dim=-1)
    
    return(ori_img_embedding @ text_embedding.T).item()

def to_prompt(label):

    labels_map = {
        0: 'truck',
        1: 'car',
        2: 'plane',
        3: 'ship',
        4: 'cat',
        5: 'dog',
        6: 'equine',
        7: 'deer',
        8: 'frog',
        9: 'bird',
    }

    return f"a photo of a {labels_map[label]}"

def denormalize_clip(tensor):
    """Desnormaliza tensor do CLIP de volta para [0,1]"""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=tensor.device).view(3, 1, 1)
    return tensor * std + mean

def analysis_pertub(img, label, int_method):
    img_processed = preprocess(img).to(device).unsqueeze(0)
    img_keepsize = imgprocess_keepsize(img).to(device).unsqueeze(0).to(torch.float32)

    w, h = img.size
    resize = Resize((h,w))

    similarity_original = simple_similarity(img_processed, label)
    hm_original = simple_hm(int_method, img_processed, label, resize, img_keepsize)

    img_processed_perturbed = perturb(img_processed.clone(), label)
    img_perturbed_denorm = denormalize_clip(img_processed_perturbed.squeeze(0))

    img_pil_perturbed = ToPILImage()(torch.clamp(img_perturbed_denorm.cpu(), 0, 1))

    img_keepsize_perturbed = imgprocess_keepsize(img_pil_perturbed).to(device).unsqueeze(0).to(torch.float32)

    similarity_perturbed = simple_similarity(img_processed_perturbed, label)
    hm_perturbed = simple_hm(int_method, img_processed_perturbed, label, resize, img_keepsize_perturbed)

    return similarity_original, similarity_perturbed, hm_original, hm_perturbed
