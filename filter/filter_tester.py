import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from gh_filter import GHFilter
import sys
import matplotlib.pyplot as plt


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
    r3 = R.from_quat(quaternion)
    rotation_matrix = r3.as_matrix()
    return rotation_matrix

def gh_test(g=0.5,h=0.5):
    transform_matrix_array, start_time_array, end_time_array=load(2)
    x_updated_array=np.zeros((1000,4))
    x_predicted_array=np.zeros((1000,4))
    tma_1=(transform_matrix_array[:,:,1])
    #print(tma_1.shape)

    for i in range(1000):
        if math.isnan(tma_1[i, 1]):
            #print("nan")
            continue
        elif not(math.isnan(tma_1[i, 1])):
            gh_filter=GHFilter(g,h,tma_1[i,:])
            x_updated=tma_1[i,:]
            break
        
    for j in range(i+1,1000):
        last_measurement=tma_1[j-1,:]
        measurement=tma_1[j,:]
        
        #print("last_measurement:"+str(last_measurement))
        #print("now_measurement:"+str(measurement))
        # if math.isnan(last_measurement[1]):
        #     x_predicted=gh_filter.predict(x_updated)
        # elif not(math.isnan(last_measurement[1])):
        #     x_predicted=gh_filter.predict(x_updated)
        x_predicted=gh_filter.predict(x_updated)
        x_predicted_array[j,:]=x_predicted
            
        if math.isnan(measurement[1]):
            x_updated=x_predicted
        elif not(math.isnan(measurement[1])):
            x_updated=gh_filter.update(measurement, x_predicted)
        x_updated_array[j,:]=x_updated 
        
    x_origin=tma_1   
    return x_origin,x_predicted_array, x_updated_array

def save_plot():
    for i in range(0,10,1):
        for j in range(0,10,1):
            x_origin,x_predicted_array, x_updated_array=gh_test(i/10,i/10)
            plt.figure()
            plt.plot(x_origin[:200, 0], label='x_origin[:, 0]')
            plt.plot(x_updated_array[:200, 0], label='x_updated_array[:, 0]')
            # 添加图例
            plt.legend()

            # 设置坐标轴标签
            plt.xlabel('Index')
            plt.ylabel('Value')
            
            save_path='./figure/gh_g'+str(i)+"_h"+str(j)+'.png'

            plt.savefig(save_path)
            print(i/10,j/10)
            
save_plot()


        









