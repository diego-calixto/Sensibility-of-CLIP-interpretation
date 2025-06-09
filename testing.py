from PIL import Image
from utils import *
import clip
from torchvision.transforms import Resize
import matplotlib.pyplot as plt

img_input = Image.open("dog_and_car.png")
txt_input = "a dog in a car waiting for traffic lights"

# text_processed = clip.tokenize(txt_input[0]).cpu()
# text_embedding = clipmodel.encode_text(text_processed)
# text_embedding = F.normalize(text_embedding, dim=-1)


# # hm_types = ['eclip-wo-ksim_gt', 'eclip-wo-ksim_pred', 'eclip_gt', 'eclip_pred', 'game_gt', 'game_pred',
# #         'gradcam_gt', 'gradcam_pred', 'maskclip_gt', 'maskclip_pred', 'selfattn', 'surgery_gt', 'surgery_pred', 'm2ib_gt', 'm2ib_pred']

# #def generate_hm(hm_type, img, txt_embedding, txts, resize):

# w, h = img_input.size
# resize = Resize((h,w))

# hm = generate_hm("gradcam_gt", img_input, text_embedding, txt_input, resize)

# c_ret = visualize(hm, img_input.copy(), resize)
# plt.imshow(c_ret)
# plt.axis('off')
# plt.show()

simple_hm("m2ib_gt" ,img_input, txt_input, True)








# perturbed_image = perturb(img_input, txt_input)
# image_processed = preprocess(img_input).unsqueeze(0).to(device)
# # Comparar os tensores (ver se são diferentes)
# difference = image_processed != perturbed_image # Vai retornar um tensor de valores booleanos

# # Converter para valores numéricos (1 para True e 0 para False)
# num_differences = difference.sum().item()  # Soma os 1's, ou seja, os valores diferentes
# print(num_differences)

# perturbed_image_features = model.encode_image(perturbed_image)
# perturbed_image_features = F.normalize(perturbed_image_features, dim=-1)

# print(perturbed_image_features)

# #similarity matrix with perturbed image
# perturbed_cosine = (perturbed_image_features @ text_features.T)

# print(f"Cosine similarity before perturbation: {cosine.item()}")
# print(f"Cosine similarity after perturbation: {perturbed_cosine.item()}")
