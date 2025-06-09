import time
import numpy as np
import torch
import re
import torch.nn.functional as F
import Game_MM_CLIP.clip as mm_clip
import cv2
import clip
from torchvision.transforms import Resize
import matplotlib.pyplot as plt

from generate_emap import clipmodel, preprocess, imgprocess_keepsize, mm_clipmodel, mm_interpret, \
        clip_encode_dense, grad_eclip, grad_cam, mask_clip, compute_rollout_attention, \
        surgery_model, clip_surgery_map, m2ib_model, m2ib_clip_map, rise


device = "cuda" if torch.cuda.is_available() else "cpu"

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
    
def generate_hm(hm_type, img, txt_embedding, txts, resize):
    start = time.time()
    img_keepsized = imgprocess_keepsize(img).to(device).unsqueeze(0)
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
        img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
        text_tokenized = mm_clip.tokenize(txts).to(device)
        emap = mm_interpret(model=mm_clipmodel, image=img_clipreprocess, texts=text_tokenized, device=device)    
        emap = emap.sum(0) 
    elif "rollout" in hm_type:
        start = time.time()
        img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
        text_tokenized = mm_clip.tokenize(txts).to(device)
        attentions = mm_interpret(model=mm_clipmodel, image=img_clipreprocess, texts=text_tokenized, device=device, rollout=True)      
        emap = compute_rollout_attention(attentions)[0]
    elif "surgery" in hm_type:
        start = time.time()
        img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
        all_texts = ['airplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform', 'potted plant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table', 'track', 'train', 'tree', 'truck', 'tv monitor', 'wall', 'water', 'window', 'wood']
        all_texts = txts + all_texts
        emap = clip_surgery_map(model=surgery_model, image=img_clipreprocess, texts=all_texts, device=device)[0,:,:,0]
    elif "m2ib" in hm_type:
        start = time.time()
        img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
        emap = m2ib_clip_map(model=m2ib_model, image=img_clipreprocess, texts=txts, device=device)
        emap = torch.tensor(emap)
    elif "rise" in hm_type:
        start = time.time()
        img_clipreprocess = preprocess(img).unsqueeze(0)
        emap = rise(model=clipmodel, image=img_clipreprocess, txt_embedding=txt_embedding, device=device)
        print(emap.shape)
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




def simple_hm(hm_type, img, txt, print=False):
    """
    ## hm_types

    ['eclip-wo-ksim_gt', 'eclip-wo-ksim_pred', 'eclip_gt', 'eclip_pred', 'game_gt', 'game_pred', 
    'gradcam_gt', 'gradcam_pred', 'maskclip_gt', 'maskclip_pred', 'selfattn', 'surgery_gt', 'surgery_pred', 'm2ib_gt', 'm2ib_pred']
    """

    text_processed = clip.tokenize(txt).cpu()
    text_embedding = clipmodel.encode_text(text_processed)
    text_embedding = F.normalize(text_embedding, dim=-1)

    w, h = img.size
    resize = Resize((h,w))

    
    hm = generate_hm(hm_type, img, text_embedding, [txt], resize)

    if(print):
        c_ret = visualize(hm, img.copy(), resize)
        plt.imshow(c_ret)
        plt.axis('off')
        plt.show()
    
    return hm


# hm_types = ['eclip-wo-ksim_gt', 'eclip-wo-ksim_pred', 'eclip_gt', 'eclip_pred', 'game_gt', 'game_pred',
#         'gradcam_gt', 'gradcam_pred', 'maskclip_gt', 'maskclip_pred', 'selfattn', 'surgery_gt', 'surgery_pred', 'm2ib_gt', 'm2ib_pred']

#def generate_hm(hm_type, img, txt_embedding, txts, resize):

