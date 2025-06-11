from utils import simple_hm, simple_similarity
from pertubation import perturb

def analysis_pertub(img, label, int_method):
    # 'similarity_original': [],
    # 'similarity_perturb': [],
    # 'hm_original': [],
    # 'hm_perturbed': []

    similarity_original = simple_similarity(img, label)
    hm_original = simple_hm(int_method, img, label)

    img_perturbed = perturb(img, label)

    similarity_perturbed = simple_similarity(img_perturbed, label)
    hm_perturbed = simple_hm(int_method, img_perturbed, label)

    return similarity_original, similarity_perturbed, hm_original, hm_perturbed