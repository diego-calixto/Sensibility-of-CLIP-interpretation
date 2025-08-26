import pandas as pd
import os

def concat_batches():
    # Caminho onde os arquivos de batches estão localizados
    batch_dir = "./"
    batch_files = [f for f in os.listdir(batch_dir) if f.startswith('results_batch')]

    batch_files_sorted = sorted(batch_files, key=lambda x: int(x.split('_')[2].split('.')[0]))

    # Inicializar uma lista para armazenar os DataFrames
    dfs = []
    i = 0
    # Ler e concatenar todos os arquivos de batches
    for batch_file in batch_files_sorted:
        print(batch_file)
        batch_df = pd.read_pickle(os.path.join(batch_dir, batch_file))
        dfs.append(batch_df)
        


    # Concatenar todos os DataFrames em um só
    final_df = pd.concat(dfs, ignore_index=True)

    # Salvar o DataFrame completo em um único arquivo
    final_df.to_pickle("./results_full.pkl")

    # Confirmar a conclusão
    print("All the batches were combined in a unique file")

concat_batches()