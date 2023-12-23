
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import sys

import numpy as np
import math

from utlis import transform_matrix_constraint,tranform_matrix2quaternion,quaternion2transform_matrix,quaternion2rotation_matrix

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
        
        x_est = x + np.squeeze(self.velocity * self.dt)
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
        self.velocity = np.squeeze(self.velocity) + self.h * (residual) / self.dt
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
            if not(np.isnan(x_measurement).all()):
                self.x_updated = x_measurement
        elif not(np.all(self.x_updated== 0)):
            self.x_predicted = self.predict(self.x_updated)
            if np.isnan(x_measurement).all():
                self.x_updated = self.x_predicted
            elif not(np.isnan(x_measurement).all()):
                self.x_updated = self.update(x_measurement, self.x_predicted)
        return np.squeeze(self.x_updated), np.squeeze(self.x_predicted)
    
    
def main():
    collect_data_path = './data/transform_matrix_8.npy'
    transform_matrix_array = np.load(collect_data_path)
    t_shape = np.zeros((4, 4))
    t_gh_filter = GHFilter(0.5, 0.5, t_shape)
    
    qxyz_shape = np.zeros(7)
    qxyz_gh_filter = GHFilter(0.5, 0.5, qxyz_shape)
    
    for i in range(1000):
        t = transform_matrix_array[i,:,:]
        
        t_updated, t_predicted = t_gh_filter.filter(t)
        t_normed = transform_matrix_constraint(t_updated)
        t_gh_filter.x_updated = t_normed
        
        q = tranform_matrix2quaternion(t)
        xyz = t[0:3,3]
        qxyz = np.hstack((q, xyz))
        qxyz_updated, qxyz_predicted = qxyz_gh_filter.filter(qxyz)
        t_updated_from_qxyz=quaternion2transform_matrix(qxyz_updated)
        print(t_updated_from_qxyz)
    return
        
    

if __name__ == "__main__":
    main()
    

    
