
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import sys

import numpy as np
import math

from utlis import load_transform_matrix, transform_matrix_constraint

class GHFilter:
    def __init__(self, g, h, x_shape_numpy, dt=1):
        """
        Initializes the GHFilter class.

        Args:
            g: The gain parameter for the filter.
            h: The measurement matrix for the filter.
            x_shape_numpy: The shape of the state vector.
            dt: The time step for the filter. Default is 1.
        """
        self.velocity = np.zeros(x_shape_numpy.shape)
        self.h = h
        self.g = g
        self.dt = dt
        self.x_predicted = np.zeros(x_shape_numpy.shape)
        self.x_updated = np.zeros(x_shape_numpy.shape)

    def predict(self, x):
        """
        Predicts the next state based on the current state and velocity.

        Args:
            x: The current state vector.

        Returns:
            The predicted state vector.
        """
        x_est = x + self.velocity * self.dt
        return x_est

    def update(self, x_measurement, x_est):
        """
        Updates the state based on the measurement and predicted state.

        Args:
            x_measurement: The measured state vector.
            x_est: The predicted state vector.

        Returns:
            The updated state vector.
        """
        residual = x_measurement - x_est
        self.velocity = self.velocity + self.h * (residual) / self.dt
        x_updated = x_est + self.g * residual
        return x_updated

    def filter(self, x_measurement):
        """
        Filters the measurement to estimate the state.

        Args:
            x_measurement: The measured state vector.

        Returns:
            The updated state vector and the predicted state vector.
        """
        if np.all(self.x_updated== 0) :
            if not(math.isnan(x_measurement[1, 1])):
                self.x_updated = x_measurement
        elif not(np.all(self.x_updated== 0)):
            self.x_predicted = self.predict(self.x_updated)
            if math.isnan(x_measurement[1, 1]):
                self.x_updated = self.x_predicted
            elif not(math.isnan(x_measurement[1, 1])):
                self.x_updated = self.update(x_measurement, self.x_predicted)
        return self.x_updated, self.x_predicted
    
    
def main():
    collect_data_path = './data/transform_matrix_8.npy'
    transform_matrix_array=np.load(collect_data_path)
    x_shape=np.zeros((4,4))
    gh_filter=GHFilter(0.5,0.5,x_shape)
    for i in range(1000):
        tmx=transform_matrix_array[i,:,:]
        x_updated,x_predicted=gh_filter.filter(tmx)
        x_normed=transform_matrix_constraint(x_updated)
        gh_filter.x_updated=x_normed
        
    

if __name__ == "__main__":
    main()
    

    
