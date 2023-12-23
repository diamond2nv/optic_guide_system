import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from gh_filter import GHFilter
import matplotlib.pyplot as plt
from utlis import load,transform_matrixarray2quaternionarray,quaternion2transform_matrix

from gh_filter import GHFilter
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

        measurement=tma_1[j,:]

        x_predicted=gh_filter.predict(x_updated)
        x_predicted_array[j,:]=x_predicted
            
        if math.isnan(measurement[1]):
            x_updated=x_predicted
        elif not(math.isnan(measurement[1])):
            x_updated=gh_filter.update(measurement, x_predicted)
        x_updated_array[j,:]=x_updated 
        
    x_origin=tma_1   
    return x_origin,x_predicted_array, x_updated_array

def quanternion_test():
    tmxa,sta,eta=load(12)
    qa=transform_matrixarray2quaternionarray(tmxa)
    gh_shape=np.zeros(4)
    gh_filter=GHFilter(0.5,0.5,gh_shape)
    quanternion_updated_array=np.zeros((1000,4))
    r_updated_array=np.zeros((1000,3,3))
    for i in range(1000):
        q=qa[i,:]
        x_updated,x_predicted=gh_filter.filter(q) 
        quanternion_updated_array[i,:]=np.squeeze(x_updated)
        tmx_updated=quaternion2transform_matrix(x_updated)
        r_updated_array[i,:,:]=tmx_updated
    return quanternion_updated_array,r_updated_array

quanternion_test()
    



        









