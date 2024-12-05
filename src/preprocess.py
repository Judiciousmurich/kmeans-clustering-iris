def preprocess_dataset(iris):
    """Preprocesses the dataset by cleaning the species column."""
    iris['Species'] = iris['Species'].str.replace('Iris-', '', regex=False)
    return iris
