import pandas as pd

def load_iris_dataset(file_path="./data/iris.csv"):
    iris = pd.read_csv("./data/iris.csv")
    return iris
