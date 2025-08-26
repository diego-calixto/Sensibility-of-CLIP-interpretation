import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity
import sys

# Função para calcular Spearman Rank Correlation
def calculate_spearman(hm_original, hm_perturbed):
    # Achatar os mapas e calcular a correlação de Spearman
    flat_original = np.array(hm_original).flatten()
    flat_perturbed = np.array(hm_perturbed).flatten()
    spearman_corr, _ = spearmanr(flat_original, flat_perturbed)
    return spearman_corr

# Função para calcular Top-K Intersection
def topk_intersection(map1, map2, k):
    # Achatar os tensores
    flat1 = map1.flatten()
    flat2 = map2.flatten()
    
    # Obter os índices dos top-K maiores valores
    topk_indices_1 = torch.topk(torch.tensor(flat1), k).indices
    topk_indices_2 = torch.topk(torch.tensor(flat2), k).indices

    # Calcular a interseção dos índices
    set1 = set(topk_indices_1.tolist())
    set2 = set(topk_indices_2.tolist())
    
    intersection = set1.intersection(set2)
    
    return len(intersection), len(intersection) / k

def calculate_ssim (hm_original, hm_perturbed) :
    
    hm_original = hm_original.reshape(224, 224)
    hm_perturbed = hm_perturbed.reshape(224, 224)

    return structural_similarity(hm_original, hm_perturbed, data_range=hm_perturbed.max() - hm_perturbed.min())

def similarity_diff (sim_original, sim_perturbed):
    return (sim_perturbed - sim_original) / sim_original


if len(sys.argv) > 1:
    file = sys.argv[1]
else :
    print("Missing a parameter. Please input the file name.\n\nmetrics.py [file_name]")
    sys.exit()
# file = "results_simple.pkl"

# log = pd.read_pickle("\home\dhvc\projetos\backup\" + file)
log = pd.read_pickle(file)

k = 100  # Definir o valor de k para o cálculo do Top-K Intersection
spearman_results = []
topk_results = []
k_value = []
ssim = []
sim_diff = []


for idx, row in log.iterrows():
    print("photo index: ", idx)
    
    hm_original = np.array(row['hm_original'])  
    hm_perturbed = np.array(row['hm_perturbed'])
    sim_original = row['similarity_original']
    sim_perturbed = row['similarity_perturb']
    
    
    _ , inter_ratio = topk_intersection(hm_original, hm_perturbed, k)
    
    # Adicionar os resultados nas listas
    spearman_results.append(calculate_spearman(hm_original, hm_perturbed))
    topk_results.append(inter_ratio)
    k_value.append(k)
    ssim.append(calculate_ssim(hm_original, hm_perturbed))
    sim_diff.append(similarity_diff(sim_original, sim_perturbed))

    

print("End of the metrics calculation\n")
# Adicionando as novas colunas ao DataFrame
log['spearman_rank_correlation'] = spearman_results
log['top_k_intersection'] = topk_results
log['k_value'] = k_value
log['ssim'] = ssim
log['simimilarity_diff'] = sim_diff
    
# Verificando os resultados
print(log.head())

print("saving log on pickle")
try:
    log.to_pickle(file)
except Exception as e:
    print(f"Error saving file: {e}")