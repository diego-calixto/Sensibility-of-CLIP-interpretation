import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr

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


log = pd.read_pickle("/home/dhvc/projetos/backup/results_full_16_255.pkl")

k = 100  # Definir o valor de k para o cálculo do Top-K Intersection
spearman_results = []
topk_results = []
k_value = []

# Iterando sobre as linhas do DataFrame
for idx, row in log.iterrows():
    print("photo index: ", idx)
    # Converter os valores de 'hm_original' e 'hm_perturbed' para tensors ou listas
    hm_original = np.array(row['hm_original'])  # Ou algum outro formato
    hm_perturbed = np.array(row['hm_perturbed'])
    
    # Calcular o Spearman
    spearman_corr = calculate_spearman(hm_original, hm_perturbed)
    
    # Calcular o Top-K Intersection
    inter_count, inter_ratio = topk_intersection(hm_original, hm_perturbed, k)
    
    # Adicionar os resultados nas listas
    spearman_results.append(spearman_corr)
    topk_results.append(inter_ratio)
    k_value.append(k)

print("End of the metrics calculation\n")
# Adicionando as novas colunas ao DataFrame
log['spearman_rank_correlation'] = spearman_results
log['top_k_intersection'] = topk_results
log['k_value'] = k_value
    
# Verificando os resultados
print(log.head())

print("saving log on pickle")
try:
    log.to_pickle("./results_with_metrics_16_255.pkl")
except Exception as e:
    print(f"Error saving file: {e}")