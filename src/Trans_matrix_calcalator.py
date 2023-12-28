from Config_loader import Config_loader
from Camera import Camera
from Marker import Marker
 
class Transform_Matrix_Calculator():
 
    def __init__(self):
        config_loader=Config_loader('transform_matrix_calculator.yaml')
        params=config_loader.get_params()
        self.params=params
        if params['show_print']:
            print(params)
        if params['mode']['online']: 
            #self.camera_setup(params)
            self.marker_setup(params)
        elif not(params['mode']['online']):
            pass

    def camera_setup(self, params):
        self.camera=Camera(params)
        if params['show_print']:
            print("Camera init success")
        
    def marker_setup(self,params):
        if params['needle_marker']['shape']:
            self.needle_marker=Marker(params['needle_marker'])
            if params['show_print']:
                print("Marker : needle_marker init success")
        else:
            pass
        if params['ref_marker']['shape']:
            self.ref_marker=Marker(params['ref_marker'])
            if params['show_print']:
                print("Marker : ref_marker init success")
        else:
            pass  
        if params['show_print']:
            print("Marker(s) init success")
        
    def calc_c2m(self,params):
        if params['needle_marker']['shape']:
            self.camera.trans_matrix_calc(self.needle_marker)
            if params['show_print'] :
                print("needle_marker's transform matrix:")
                print(self.needle_marker.matrix)
        else:
            pass
        
        if params['ref_marker']['shape']:
            self.camera.trans_matrix_calc(self.ref_marker)
            if params['show_print'] :
                print("ref_marker's transform matrix:")
                print(self.ref_marker.matrix)
        else:
            pass 
        