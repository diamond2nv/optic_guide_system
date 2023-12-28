import numpy as np
from scipy.spatial.transform import Rotation as R

class Converter():
    def __init__(self):
        pass
    
    def tranform_matrix2quaternion(self,transform_matrix):
        # 将旋转平移矩阵转换为四元数
        
        if transform_matrix is None:
            return np.array([np.nan, np.nan, np.nan, np.nan])
        rotation_matrix = transform_matrix[0:3, 0:3]
        r3 = R.from_matrix(rotation_matrix)
        quaternion = r3.as_quat()
        return quaternion
    
    def transform_matrixarray2quaternionarray(self,transform_matrix_array):
        quaternion_array = np.zeros((transform_matrix_array.shape[0], 4))
        for i in range(transform_matrix_array.shape[0]):
            quaternion_array[i] = self.tranform_matrix2quaternion(transform_matrix_array[i])
        return quaternion_array
    
    def quaternion2rotation_matrix(self,quaternion):
        # 将四元数转换为旋转平移矩阵
        if np.all(quaternion == 0):
            return np.zeros((3,3))
        
        r3 = R.from_quat(quaternion)
        rotation_matrix = r3.as_matrix()
        return rotation_matrix
    
    def quaternion2transform_matrix(self,qxyz):
        t_from_qxyz = np.hstack((self.quaternion2rotation_matrix(qxyz[0:4]),np.reshape(qxyz[4:7],(3,1))))
        t_from_qxyz=np.vstack((t_from_qxyz,np.array([0,0,0,1])))
        return t_from_qxyz
    
    def transform_matrix_constraint(self,transform_matrix):
        if(np.all(transform_matrix== 0)) :
            return transform_matrix
        
        normed_transform_matrix=transform_matrix
        for i in range(2):
            x=transform_matrix[0:3,i]
            x=x/np.linalg.norm(x)
            normed_transform_matrix[0:3,i]=x
        return normed_transform_matrix
    
    def quaternion_constraint(self,quaternion):
        if(np.all(quaternion== 0)) :
            return quaternion
        
        normed_quaternion=quaternion
        normed_quaternion=normed_quaternion/np.linalg.norm(normed_quaternion)
        return normed_quaternion
    
    def qxyz_constraint(self,qxyz):
        if(np.all(qxyz== 0)) :
            return qxyz
        
        normed_qxyz=qxyz