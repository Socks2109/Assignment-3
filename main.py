import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris 
# from sklearn.metrics import  
from cluster import (KMeans)
from cluster.visualization import plot_3d_clusters


def main(): 
    # Set random seed for reproducibility
    np.random.seed(42)
    # Using sklearn iris dataset to train model
    og_iris = np.array(load_iris().data)
    
    # Initialize your KMeans algorithm
    model = KMeans(k = 3, metric = "euclidean", max_iter = 1000, tol = 1e-4)
    
    # Fit model
    model.fit(og_iris)

    # Load new dataset
    df = np.array(pd.read_csv('data/iris_extended.csv', 
                usecols = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']))

    # Predict based on this new dataset
    
    
    # You can choose which scoring method you'd like to use here:
    
    
    # Plot your data using plot_3d_clusters in visualization.py

    
    # Try different numbers of clusters

    
    # Plot the elbow plot

    
    # Question: 
    # Please answer in the following docstring how many species of flowers (K) you think there are.
    # Provide a reasoning based on what you have done above: 
    
    """
    How many species of flowers are there: 
    
    Reasoning: 
    
    
    
    
    """

    
if __name__ == "__main__":
    main()