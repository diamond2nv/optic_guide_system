import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R

class Marker():
    def __init__(self,params):
        if params['shape'] ==4:
            self.shape= aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        elif params['shape'] ==5:
            self.shape= aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
        else:
            pass
        self.matrix=np.full((4, 4), np.nan)
        self.qxyz=np.full((7), np.nan)
        self.side_length=params['side_length']
        
    def qxyz_update_from_matrix(self):
        rotation_matrix = self.matrix[0:3, 0:3]
        r3 = R.from_matrix(rotation_matrix)
        self.quanternion = r3.as_quat()
        xyz = self.matrix[0:3,3]
        self.qxyz = np.hstack((self.quanternion, xyz))
        return self.qxyz
        
    def matrix_update_from_qxyz(self):
        r3 = R.from_quat(self.qxyz[0:4])
        rotation_matrix = r3.as_matrix()
        t_from_qxyz = np.hstack((rotation_matrix,np.reshape(self.qxyz[4:7],(3,1))))
        t_from_qxyz=np.vstack((t_from_qxyz,np.array([0,0,0,1])))
        self.matrix = t_from_qxyz
        return self.matrix
        