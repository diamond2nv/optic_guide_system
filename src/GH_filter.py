import numpy as np

class GH_filter():
    def __init__(self,params):
        self.data_shape=params['gh_filter']['data_shape']
        
        self.h=params['gh_filter']['h']
        self.g=params['gh_filter']['g']
        self.dt=params['gh_filter']['dt']
        
        self.velocity=np.zeros(self.data_shape)
        self.x_predicted=np.zeros(self.data_shape)
        self.x_updated=np.zeros(self.data_shape)
        
        self.max_unmeansure_frames=params['gh_filter']['max_unmeansure_frames']
        self.current_unmeansure_frames=0
        
    def predict(self,x):
        x_est=x+np.squeeze(self.velocity*self.dt)
        return x_est
    
    def update(self,x_measurement,x_est):
        residual=x_measurement-x_est
        self.velocity=np.squeeze(self.velocity)+self.h*(residual)/self.dt
        x_updated=x_est+self.g*residual
        return x_updated
    
    def filt(self,x_measurement):
        if np.all(self.x_updated==0):
            if not(np.isnan(x_measurement).all()):
                self.x_updated=x_measurement
        elif not(np.all(self.x_updated==0)):
            self.x_predicted=self.predict(self.x_updated)
            if np.isnan(x_measurement).all():
                self.x_updated=self.x_predicted
                self.current_unmeansure_frames+=1
                if self.current_unmeansure_frames>self.max_unmeansure_frames:
                    self.x_updated=np.zeros(self.x_updated.shape)
            else:
                self.x_updated=self.update(x_measurement,self.x_predicted)
                self.current_unmeansure_frames=0
        return self.x_updated,self.x_predicted
    
    def filt_marker_qxyz(self,marker):
        self.filted_qxyz,self.predicted_qxyz=self.filt(marker.qxyz)
        marker.filted_qxyz=self.filted_qxyz
        marker.matrix_update_from_qxyz()