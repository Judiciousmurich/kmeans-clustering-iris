import matplotlib.pyplot as plt

def visualize_clusters(x, kmeans_predict, kmeans_model):
    """Visualizes the clusters and centroids using matplotlib."""
    # Plotting the clusters
    plt.scatter(x[kmeans_predict == 0, 0], x[kmeans_predict == 0, 1], s=100, c='red', label='Setosa')
    plt.scatter(x[kmeans_predict == 1, 0], x[kmeans_predict == 1, 1], s=100, c='blue', label='Versicolour')
    plt.scatter(x[kmeans_predict == 2, 0], x[kmeans_predict == 2, 1], s=100, c='green', label='Virginica')
    
    # Plotting the centroids
    plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], s=100, c='yellow', label='Centroids')
    plt.legend()
    plt.title("K-Means Clustering of Iris Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
