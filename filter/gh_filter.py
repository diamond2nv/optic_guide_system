
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import sys

class GHFilter():
    def __init__(self,g,h,x_shape_numpy,dt=1) -> None:
        # self.velocity=np.zeros(x0.shape[0])
        self.velocity=np.zeros(x_shape_numpy.shape)
        self.h=h
        self.g=g
        self.dt=dt
        self.x_predicted=np.zeros(x_shape_numpy.shape)
        self.x_updated=np.zeros(x_shape_numpy.shape)
    def predict(self, x):
        x_est = x + self.velocity * self.dt
        return x_est
    def update(self,x_measurement,x_est):
        residual = x_measurement - x_est
        self.velocity = self.velocity + self.h * (residual) / self.dt
        x_updated = x_est + self.g * residual
        return x_updated
    def filter(self,x_measurement):
        if self.x_updated.all==0:
            if not(math.isnan(x_measurement[1,1])):
                self.x_updated=x_measurement
        elif not(self.x_updated.all==0):
            self.x_predicted=self.predict(self.x_updated)
            if math.isnan(x_measurement[1,1]):
                self.x_updated=self.x_predicted
            elif not(math.isnan(x_measurement[1,1])):
                self.x_updated=self.update(x_measurement,self.x_predicted)
        return self.x_updated,self.x_predicted
    
    
def main():
    collect_data_path = './data/transform_matrix_8.npy'
    transform_matrix_array=np.load(collect_data_path)
    print(transform_matrix_array.shape)
    x_shape=np.zeros((4,4))
    gh_filter=GHFilter(0.5,0.5,x_shape)
    for i in range(1000):
        tmx=transform_matrix_array[i,:,:]
        x_updated,x_predicted=gh_filter.filter(tmx)
        
    

if __name__ == "__main__":
    main()
    

    
