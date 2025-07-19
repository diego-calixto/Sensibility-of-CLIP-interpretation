import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def getValues (metric):
    log1 = pd.read_pickle("./metrics_results_simple.pkl")
    log2 = pd.read_pickle("./metrics_results_simple_2.pkl")
    log3 = pd.read_pickle("./metrics_results_simple_3.pkl")
    log4 = pd.read_pickle("./metrics_results_simple_4.pkl")

    log_list = [pd.read_pickle("./metrics_results_simple.pkl"), 
                pd.read_pickle("./metrics_results_simple_2.pkl"), 
                pd.read_pickle("./metrics_results_simple_3.pkl"), 
                pd.read_pickle("./metrics_results_simple_4.pkl")]
    
    data = []

    for log in log_list:
        
        data.append(
            log.groupby('img_id').agg({
            metric: 'mean',  # Média da metrica
            }).reset_index()
        )
    
    return [val["spearman_rank_correlation"].iloc[0] for val in data]

# spearman_rank_correlation top_k_intersection  k_value      ssim  simimilarity_diff
data_values = getValues('spearman_rank_correlation')



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
plt.show()