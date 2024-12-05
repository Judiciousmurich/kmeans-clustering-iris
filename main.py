from src.load_dataset import load_iris_dataset
from src.preprocess import preprocess_dataset
from src.kmeans_model import train_kmeans
from src.visualize import visualize_clusters

# Step 1: Load the dataset
iris = load_iris_dataset()

# Step 2: Preprocess the dataset
iris = preprocess_dataset(iris)

# Step 3: Prepare the data for K-Means
x = iris.iloc[:, [1, 2, 3, 4]].values

# Step 4: Train the K-Means model
kmeans_model, kmeans_predict = train_kmeans(x)

# Step 5: Visualize the clusters
visualize_clusters(x, kmeans_predict, kmeans_model)
