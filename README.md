# K-Means Clustering on the Iris Dataset

## Description
This project applies K-Means clustering on the Iris dataset to identify patterns and clusters based on flower measurements.

## Files
- `data/iris.csv`: The input dataset.
- `notebooks/kmeans_iris.ipynb`: Jupyter Notebook for the clustering workflow.
- `src/kmeans_clustering.py`: Python script for clustering.

  ##  Steps to Run
Load the Dataset
The dataset is loaded from data/iris.csv. You can replace this path with the location of your own Iris dataset if necessary.

## Preprocess the Data
The dataset is cleaned by removing the "Iris-" prefix from the species names.

##  Train the K-Means Model
The K-Means algorithm is applied with 3 clusters (as there are 3 species in the Iris dataset). The model is trained on the features SepalLengthCm, SepalWidthCm, PetalLengthCm, and PetalWidthCm.

## Visualize the Results
The resulting clusters and their centroids are visualized using matplotlib. Each cluster is shown with a different color, and the centroids are marked with yellow.

## Output
The output will be a plot displaying the Iris dataset clustered into 3 categories (Setosa, Versicolor, Virginica), along with the cluster centroids.
