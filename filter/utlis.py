import numpy as np
from scipy.spatial.transform import Rotation as R


def save_transform_matrix(matrix, path):
    np.save(path, matrix)
    

def load_transform_matrix(path):
    matrix = np.load(path)
    return matrix

def load(index):
    path='./data/transform_matrix_'+str(index)+'.npy'
    transform_matrix_array=np.load(path)
    path='./data/start_time_'+str(index)+'.npy'
    start_time_array=np.load(path)
    path='./data/end_time_'+str(index)+'.npy'
    end_time_array=np.load(path)
    return transform_matrix_array, start_time_array, end_time_array
    

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
    if np.all(quaternion == 0):
        return np.zeros((3,3))
    
    r3 = R.from_quat(quaternion)
    rotation_matrix = r3.as_matrix()
    return rotation_matrix

def transform_matrix_constraint(transform_matrix):
    if(np.all(transform_matrix== 0)) :
        return transform_matrix
    
    normed_transform_matrix=transform_matrix
    for i in range(2):
        x=transform_matrix[0:3,i]
        x=x/np.linalg.norm(x)
        normed_transform_matrix[0:3,i]=x
    return normed_transform_matrix