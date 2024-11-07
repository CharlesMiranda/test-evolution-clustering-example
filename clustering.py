from tslearn.clustering import KShape
from minisom import MiniSom
import matplotlib.pyplot as plt
import numpy as np
from utils import *

# Definir o número ótimo de clusters como uma constante
OPTIMAL_K = 5

# Número de repetições para cada algoritmo
N_REPETITIONS = 10

def run_and_evaluate_clustering(data, algorithm, algorithm_name):
    """
    Executa um algoritmo de clustering várias vezes e coleta as medidas de similaridade SBD.

    Args:
        data: Dados das séries temporais.
        algorithm: Função que executa o algoritmo de clustering e retorna rótulos e centróides.
        algorithm_name: Nome do algoritmo para exibição nos gráficos.

    Returns:
        Uma lista de listas, onde cada sublista contém os valores SBD de uma execução do algoritmo.
    """

    sbd_all = []
    for i in range(N_REPETITIONS):
        labels, centroids = algorithm(data, OPTIMAL_K)  # Chama a função do algoritmo
        similarity = process_similarity(data, labels, centroids, f"{algorithm_name} (i={i+1})")
        sbd_all.append(similarity)
        plot_individual_clusters(data, labels, centroids, f"{algorithm_name} (i={i+1})")
    return sbd_all


def run_kshape(data, n_clusters):
    """Executa o KShape e retorna rótulos e centróides."""
    ks = KShape(n_clusters=n_clusters, verbose=True, random_state=42) # Mantendo a seed para reprodutibilidade
    ks.fit(data)
    return ks.labels_, normalize_ks_centroids(ks.cluster_centers_)



def run_som(data, n_clusters):
    """Executa o SOM e retorna rótulos e centróides."""
    som = MiniSom(n_clusters, 1, data.shape[1], sigma=0.3, learning_rate=0.5)
    som.random_weights_init(data)
    som.train_random(data, 100)
    labels = np.array([som.winner(x)[0] for x in data])
    centroids = np.array([data[labels == i].mean(axis=0) for i in range(n_clusters)])
    return labels, centroids


# Carregar os dados apenas uma vez
values = load_data_from_json(r'data.json')
data = values[1]



# Executar e avaliar K-Shape
kshape_sbd = run_and_evaluate_clustering(data, run_kshape, "K-Shape")

# Box plot para K-Shape
plt.figure() # Criar uma nova figura para cada boxplot
bp = plt.boxplot(kshape_sbd, showfliers=True)
plt.xticks(range(1, N_REPETITIONS + 1), range(1, N_REPETITIONS + 1))
plt.ylabel('K-Shape Shape-Based Distance')
plt.title("Distribuição SBD para K-Shape") # Adicionando título
plt.show()
print(get_box_plot_data([str(i) for i in range(1, N_REPETITIONS + 1)], bp))




# Executar e avaliar SOM
som_sbd = run_and_evaluate_clustering(data, run_som, "SOM")

# Box plot para SOM
plt.figure()
bp = plt.boxplot(som_sbd, showfliers=True)
plt.xticks(range(1, N_REPETITIONS + 1), range(1, N_REPETITIONS + 1))
plt.ylabel('SOM Shape-Based Distance')
plt.title("Distribuição SBD para SOM") # Adicionando título

plt.show()

print(get_box_plot_data([str(i) for i in range(1, N_REPETITIONS + 1)], bp))