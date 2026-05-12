import scipy.sparse as sp
import numpy as np
from tqdm import tqdm

def convert_to_toolkit_format(input_path, output_path):
    print(f"Processando {input_path}...")
    
    with open(input_path, 'r') as f:
        # Lê o cabeçalho
        header = f.readline().split()
        num_samples = int(header[0])
        num_labels = int(header[1])

        indices = []
        indptr = [0]
        
        # tqdm ajuda a monitorar o progresso em arquivos grandes
        for line in tqdm(f, total=num_samples):
            parts = line.strip().split()
            # Extrai apenas o ID antes do ':' (ex: de '45:1.0' pega o 45)
            line_labels = [int(p.split(':')[0]) for p in parts]
            
            indices.extend(line_labels)
            indptr.append(len(indices))

    # Criação da matriz CSR
    data = np.ones(len(indices), dtype=np.int8)
    y_matrix = sp.csr_matrix((data, indices, indptr), shape=(num_samples, num_labels))
    
    # Salva no formato que o seu README exige
    sp.save_npz(output_path, y_matrix)
    print(f"Salvo com sucesso: {output_path} ({y_matrix.shape})")

# Execute para o treino e para o teste
convert_to_toolkit_format("data/amazon-300k/trn_X_Y.txt", "data/amazon-300k/y_train.npz")
convert_to_toolkit_format("data/amazon-300k/tst_X_Y.txt", "data/amazon-300k/y_test.npz")