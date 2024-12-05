from sklearn.cluster import KMeans
import numpy as np

def train_kmeans(x, n_clusters=3, random_state=42):
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans_predict = kmeans_model.fit_predict(x)
    return kmeans_model, kmeans_predict
