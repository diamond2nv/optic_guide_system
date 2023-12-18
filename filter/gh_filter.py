
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import sys

def generate_transform_matrix():
    # 生成一个 4*4的数组
    matrix = np.zeros((4, 4))
    return matrix

def save_transform_matrix(matrix, path):
    np.save(path, matrix)
    

def load_transform_matrix(path):
    matrix = np.load(path)
    return matrix

def g_h_filter_predict(quaternion, velocity, dt):
    quaternion_hat = quaternion + velocity * dt
    return quaternion_hat

def g_h_filter_update(quanternion_hat,measured_quaternion, measured_velocity, dt, g, h):
    residual = measured_quaternion - quanternion_hat
    
    return

def tranform_matrix2quaternion(transform_matrix):
    # 将旋转平移矩阵转换为四元数
    rotation_matrix = transform_matrix[0:3, 0:3]
    r3 = R.from_matrix(rotation_matrix)
    quaternion = r3.as_quat()
    return quaternion

def quaternion2transform_matrix(quaternion):
    # 将四元数转换为旋转平移矩阵
    r3 = R.from_quat(quaternion)
    rotation_matrix = r3.as_matrix()
    return rotation_matrix


class gh_filter():
    def __init__(self,g,h,x0,dt=1) -> None:
        self.velocity=np.zeros((x0.size[0],1))
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

    
