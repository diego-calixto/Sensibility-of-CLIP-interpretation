import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def getValues (metric):

    log_list = [pd.read_pickle("../backup/results_full_4_255.pkl"), 
                pd.read_pickle("../backup/results_full_8_255.pkl"), 
                pd.read_pickle("../backup/results_full_16_255.pkl"), 
                pd.read_pickle("../backup/results_full_32_255.pkl")]

    print("pickles loaded")
    
    data = []

    for log in log_list:
        
        data.append(
            log.groupby('img_id').agg({
            metric: 'mean',  # Média da metrica
            }).reset_index()
        )
    
    return [val["spearman_rank_correlation"].iloc[0] for val in data]

# spearman_rank_correlation top_k_intersection  k_value      ssim  simimilarity_diff
# data_values = getValues('spearman_rank_correlation')

# print("values collected")]
["selfattn", "gradcam", "maskclip", "eclip", "game", "rollout", "surgery", "m2ib", "rise"]
1.0
1.0
1.0
1.0
0.447763629524116
0.4769023773673391
0.631786395490853
0.5970895118081463
0.009760994611701195

game =[0, 0.447763629524116, ]

x = [4, 8, 16, 32]  # Valores do eixo x (norma L∞ de perturbação)
# top_k = np.exp(-x)  # Exemplo de dados para a linha "Top-k attack"
# center_attack = np.exp(-x/2)  # Exemplo de dados para a linha "Center attack"
# random_sign = np.exp(-x/3)  # Exemplo de dados para a linha "Random Sign Perturbation"



# Plotando o gráfico
plt.figure(figsize=(6, 5))

plt.plot(x, data_values, label='Top-k attack', marker='o', color='b')

# plt.plot(x, center_attack, label='Center attack', marker='s', color='orange')
# plt.plot(x, random_sign, label='Random Sign Perturbation', marker='^', color='g')

# Adicionando título e labels aos eixos
plt.title('Simple Gradient')
plt.xlabel(r'$L_\infty$ norm of perturbation')
plt.ylabel('Top-1000 Intersection')

plt.xticks(x)
# Adicionando uma legenda
plt.legend()

# Exibindo o gráfico
plt.grid(True)
plt.savefig("grafico_spearman.png", dpi=300, bbox_inches='tight')