import numpy as np
from scipy.spatial.distance import cdist

class KMeans():
    def __init__(self, k: int, metric:str, max_iter: int, tol: float):
        """
        Args:
            k (int): 
                The number of centroids you have specified. 
                (number of centroids you think there are)
                
            metric (str): 
                The way you are evaluating the distance between the centroid and data points.
                (euclidean)
                
            max_iter (int): 
                The number of times you want your KMeans algorithm to loop and tune itself.
                
            tol (float): 
                The value you specify to stop iterating because the update to the centroid 
                position is too small (below your "tolerance").
        """ 
        # In the following 4 lines, please initialize your arguments
        self.k = k
        self.metric = metric
        self.max_iter = max_iter
        self.tol = tol
        
        # In the following 2 lines, you will need to initialize 1) centroid, 2) error (set as numpy infinity)
        self.centroids = None
        self.error = np.inf
    
    def fit(self, matrix: np.ndarray):
        """
        This method takes in a matrix of features and attempts to fit your created KMeans algorithm
        onto the provided data 
        
        Args:
            matrix (np.ndarray): 
                This will be a 2D matrix where rows are your observation and columns are features.
                
                Observation meaning, the specific flower observed.
                Features meaning, the features the flower has: Petal width, length and sepal length width 
        """
        
        # In the line below, you need to randomly select where the centroid's positions will be.
        # Also set your initialized centroid to be your random centroid position
        self.centroids = matrix[np.random.choice(len(matrix), self.k, replace=False)]
        
        # In the line below, calculate the first distance between your randomly selected centroid positions
        # and the data points
        distance = cdist(matrix, self.centroids, metric = self.metric)
        
        # In the lines below, Create a for loop to keep assigning data points to clusters, updating centroids, 
        # calculating distance and error until the iteration limit you set is reached
        for _ in range(self.max_iter):
            # Within the loop, find the each data point's closest centroid
            closest_centroids = np.argmin(distance, axis=1)
        
            # Within the loop, go through each centroid and update the position.
            # Essentially, you calculate the mean of all data points assigned to a specific cluster. This becomes the new position for the centroid
            new_centroids = []
            
            for i in range(self.k):
                new_centroids.append(np.mean(matrix[closest_centroids == i], axis=0))

            new_centroids = np.array(new_centroids)
            # Within the loop, calculate distance of data point to centroid then calculate MSE or SSE (inertia)
            distance = cdist(matrix, new_centroids, metric = self.metric)
            SSE = np.sum((matrix - new_centroids[closest_centroids]) ** 2)

        
            # Within the loop, compare your previous error and the current error
            # Break if the error is less than the tolerance you've set 
            if np.abs(self.error - SSE) < self.tol:
            
                # Set your error as calculated inertia here before you break!
                self.error = SSE
                break
            
            # Set error as calculated inertia
            self.error = SSE

            # Update the centroid position
            self.centroids = new_centroids
    
    def predict(self, matrix: np.ndarray) -> np.ndarray:
        """
        Predicts which cluster each observation belongs to.
        Args:
            matrix (np.ndarray): 
                

        Returns:
            np.ndarray: 
                An array/list of predictions will be returned.
        """
        # In the line below, return data point's assignment
        closest_distance = cdist(matrix, self.centroids, metric = self.metric)
        predict = np.argmin(closest_distance, axis=1)
        return predict

    
    def get_error(self) -> float:
        """
        The inertia of your KMeans model will be returned

        Returns:
            float: 
                inertia of your fit

        """
        return self.error
    
    
    def get_centroids(self) -> np.ndarray:
    
        """
        Your centroid positions will be returned. 
        """
        # In the line below, return centroid location
        return self.centroids
        
    