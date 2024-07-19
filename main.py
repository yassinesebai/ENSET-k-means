# Importation des librairies de Python nécessaires.
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Chargement des données ou bien simulation aléatoire des données de dimension supérieure à 4
data = np.random.rand(100, 5)

#Visualisation et taille des données.
print("Taille des données : ", data.shape)

# Implémentation de l’algorithme K-moyennes avec les stratégies d’initialisation des centres suivantes
# Aléatoire
kmeans_random = KMeans(n_clusters=3, init='random', n_init=10)
kmeans_random.fit(data)

#K-means++
kmeans_pp = KMeans(n_clusters=3, init='k-means++', n_init=10)
kmeans_pp.fit(data)

#Implémentation des méthodes de validation de Clustering.
silhouette_avg = silhouette_score(data, kmeans_pp.labels_)
calinski_harabasz = calinski_harabasz_score(data, kmeans_pp.labels_)
print("Silhouette Score : ", silhouette_avg)
print("Calinski-Harabasz Index : ", calinski_harabasz)

#Détermination du meilleur modèle de Clustering (meilleurs paramètres).
if silhouette_score(data, kmeans_random.labels_) > silhouette_avg:
    best_model = kmeans_random
else:
    best_model = kmeans_pp

# Représentation des données avec les poids des centres obtenus.
centers = best_model.cluster_centers_

#Donner la nouvelle matrice des observations.
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

#Afficher les valeurs propres et les vecteurs propres associés aux axes principaux.
print("Valeurs propres : ", pca.explained_variance_)
print("Vecteurs propres : ", pca.components_)

#Donner l’inertie de chaque axe.
print("Inertie de chaque axe : ", pca.explained_variance_ratio_)

#Vérifier que la somme des inerties de chaque axe égale la dimension de la base de données.
print("Somme des inerties : ", np.sum(pca.explained_variance_ratio_))

#Représenter les données ainsi que les centres obtenus par l’algorithme du k-means sur les deux axes principaux.
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=best_model.labels_)
plt.scatter(pca.transform(centers)[:, 0], pca.transform(centers)[:, 1], marker='x', c='red')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Clustering avec PCA')
plt.show()