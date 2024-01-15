import os
import numpy as np
from Converter import Converter

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
collect_data_path = _BASE_DIR + "/data/transform_matrix_12.npy"
tmx = np.load(collect_data_path)

def read():
    # 读取25个transform matrix
    tmxs = np.zeros((25*1000,4,4))
    for i in range(25):
        _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        collect_data_path = _BASE_DIR + "/data/transform_matrix_" + str(i+1) + ".npy"
        tmx = np.load(collect_data_path)
        # 将tmx存入tmxs
        tmxs[i*1000:(i+1)*1000,:,:] = tmx
        
    return tmxs

if __name__ == "__main__":
    tmxs = read()
    conv=Converter()
    Rs = tmxs[:, 0:3, 0:3]
    quas=conv.transform_matrixarray2quaternionarray(tmxs)
    
    t = tmxs[:, 0:3, 3]
    t.reshape(25000,3)
    qaddt=np.hstack((quas,t))
    useqt=qaddt[10000:15000,:]
    
    useqt = useqt[~np.isnan(useqt).any(axis=1)]
    variances = np.var(useqt, axis=0)
    print(variances)
    
    pass