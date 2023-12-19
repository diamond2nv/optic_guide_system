
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import sys

def generate_transform_matrix():
    # 生成一个 4*4的数组
    matrix = np.zeros((4, 4))
    return matrix




def tranform_matrix2quaternion(transform_matrix):
    # 将旋转平移矩阵转换为四元数
    rotation_matrix = transform_matrix[0:3, 0:3]
    r3 = R.from_matrix(rotation_matrix)
    quaternion = r3.as_quat()
    return quaternion

def transform_matrixarray2quaternionarray(transform_matrix_array):
    quaternion_array = np.zeros((transform_matrix_array.shape[0], 4))
    for i in range(transform_matrix_array.shape[0]):
        quaternion_array[i] = tranform_matrix2quaternion(transform_matrix_array[i])
    return quaternion_array

def quaternion2transform_matrix(quaternion):
    # 将四元数转换为旋转平移矩阵
    r3 = R.from_quat(quaternion)
    rotation_matrix = r3.as_matrix()
    return rotation_matrix


class GHFilter():
    def __init__(self,g,h,x0,dt=1) -> None:
        self.velocity=np.zeros(x0.shape[0])
        self.g=g
        self.h=h
        self.dt=dt
    def predict(self, x):
        x_est = x + self.velocity * self.dt
        return x_est
    def update(self,x_measurement,x_est):
        residual = x_measurement - x_est
        self.velocity = self.velocity + self.h * (residual) / self.dt
        x_updated = x_est + self.g * residual
        return x_updated

def main():
    collect_data_path = './data/transform_matrix_8.npy'
    transform_matrix_array=np.load(collect_data_path)
    quaternionarray=transform_matrixarray2quaternionarray(transform_matrix_array)
    gh_filter=GHFilter(0.1,0.1,quaternionarray[6,:])
    for i in range(1000):
        pass
    return

main()
    

    
