import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Criar pastas para salvar os resultados
base_path = "Prática - Processamento e análise de formas/resultados/"

# Criar as subpastas para cada questão
pastas_questoes = ["questao_negativo", "questao_r_mais_k", "questao_log", "questao_exp"]

for pasta in pastas_questoes:
    os.makedirs(os.path.join(base_path, pasta), exist_ok=True)


def salvar_imagem_histograma(imagem, titulo_imagem, titulo_histograma, pasta, nome_base):
    # Salvar a imagem transformada
    plt.imshow(imagem, cmap='gray')
    plt.title(titulo_imagem)
    plt.axis('off')  # Remove os eixos
    plt.savefig(f"{pasta}/{nome_base}_imagem.png")
    plt.close()

    # Salvar o histograma da imagem
    plt.figure()
    plt.hist(imagem.ravel(), bins=256, range=(0, 256), color='black')
    plt.title(titulo_histograma)
    plt.xlabel('Intensidade')
    plt.ylabel('Frequência')
    plt.savefig(f"{pasta}/{nome_base}_histograma.png")
    plt.close()



# lendo a img
imagem_colorida = cv2.imread("Prática - Processamento e análise de formas/imagem.jpg", cv2.IMREAD_COLOR)

# transformando em escala de cinza
imagem_cinza = cv2.cvtColor(imagem_colorida, cv2.COLOR_BGR2GRAY)



# 
# Questão 1 - T(r) produz o negativo de uma imagem
pasta_questao = os.path.join(base_path, "questao_negativo")

# Negativo da imagem
negativo = 255 - imagem_cinza

# Salvar a imagem negativa e seu histograma
salvar_imagem_histograma(negativo, 'Imagem Negativa', 'Histograma da Imagem Negativa', pasta_questao, 'negativo')



# 
# Questão 2 - T(r) = r + k, com valores crescentes e negativos de k
pasta_questao = os.path.join(base_path, "questao_r_mais_k")

def transform_r_plus_k(imagem, k):
    return cv2.add(imagem, k)

# Testando com valores diferentes de k
for k in [-100, 0, 50, 100]:
    imagem_transformada = transform_r_plus_k(imagem_cinza, k)
    
    # Salvar a imagem e seu histograma para cada valor de k
    salvar_imagem_histograma(imagem_transformada, f'Imagem com T(r) = r + {k}', f'Histograma com T(r) = r + {k}', pasta_questao, f"r_mais_{k}")



# 
# Questão 3 - T(r) = log(r)
pasta_questao = os.path.join(base_path, "questao_log")

def transform_log(imagem):
    c = 255 / np.log(1 + np.max(imagem))  # Para normalizar a imagem
    log_transformada = c * np.log(1 + imagem)
    return np.array(log_transformada, dtype=np.uint8)

# Aplicar a transformação logarítmica
imagem_log = transform_log(imagem_cinza)

# Salvar a imagem logarítmica e seu histograma
salvar_imagem_histograma(imagem_log, 'Imagem com T(r) = log(r)', 'Histograma com T(r) = log(r)', pasta_questao, 'log')



# 
# Questão 4 - T(r) = exp(r)
pasta_questao = os.path.join(base_path, "questao_exp")

def transform_exp(imagem):
    exp_transformada = np.exp(imagem / 255.0)  # Normalizar os valores antes da exponencial
    exp_transformada = 255 * (exp_transformada / np.max(exp_transformada))  # Normalizar de volta para [0, 255]
    return np.array(exp_transformada, dtype=np.uint8)

# Aplicar a transformação exponencial
imagem_exp = transform_exp(imagem_cinza)

# Salvar a imagem exponencial e seu histograma
salvar_imagem_histograma(imagem_exp, 'Imagem com T(r) = exp(r)', 'Histograma com T(r) = exp(r)', pasta_questao, 'exp')