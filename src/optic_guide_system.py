from Config_loader import Config_loader
from Camera import Camera
from Marker import Marker
from GH_filter import GH_filter
from UKF_filter import UKF_filter
from Imu import Imu
from collections import deque
from datetime import datetime
import threading


class OGS:

    def __init__(self):
        self.filter = None
        self.ref_marker = None
        self.camera = None
        self.needle_marker = None
        self.imu = None

        self.imu_data_queue = deque(maxlen=46847)
        self.needle_marker_queue = deque(maxlen=100)
        self.ref_marker_queue = deque(maxlen=1000)

        config_loader = Config_loader('optic_guide_system.yaml')
        params = config_loader.get_params()
        self.params = params
        if params['show_print']:
            print(params)
        if params['mode']['online']:
            self.camera_setup(params)
            self.marker_setup(params)
            self.filter_setup(params)
            self.imu_setup(params)
        elif not (params['mode']['online']):
            pass

    def camera_setup(self, params):
        self.camera = Camera(params)
        if params['show_print']:
            print("Camera init success")

    def marker_setup(self, params):
        if params['needle_marker']['shape']:
            self.needle_marker = Marker(params['needle_marker'])
            if params['show_print']:
                print("Marker : needle_marker init success")
        else:
            pass
        if params['ref_marker']['shape']:
            self.ref_marker = Marker(params['ref_marker'])
            if params['show_print']:
                print("Marker : ref_marker init success")
        else:
            pass
        if params['show_print']:
            print("Marker(s) init success")

    def filter_setup(self, params):
        if params['mode']['filter']:
            if params['mode']['filter'] == 1:
                self.filter = GH_filter(params)
            elif params['mode']['filter'] == 2:
                self.filter= UKF_filter(params)
            else:
                if params['show_print']:
                    print("Filter init failed, filter mode is wrong")
            if params['show_print']:
                print("Filter init success")
        else:
            pass

    def imu_setup(self, params):
        if params['mode']['imu']:
            self.imu = Imu(params)
            if params['show_print']:
                print("IMU init success")
        else:
            pass

    def process_marker(self, marker, queue, params):
        self.camera.trans_matrix_calc(marker)
        timestamp = datetime.now()  # 获取当前时间
        data = {'time': timestamp, 'matrix': marker.matrix}
        queue.append(data)  # 将时间和矩阵作为一个集合添加到队列中
        if params['show_print']:
            print(f"{marker.name}'s transform matrix:")
            print(marker.matrix)

    def calc_c2m(self, params):
        try:
            if params['needle_marker']['shape']:
                self.process_marker(self.needle_marker, self.needle_marker_queue, params)
            if params['ref_marker']['shape']:
                self.process_marker(self.ref_marker, self.ref_marker_queue, params)
        except:
            if params['show_print']:
                print("calc_c2m failed")
        # print("calc_c2m finish")

    def run_imu_in_loop(self, params):
        while True:
            self.imu.receive_data(self.imu_data_queue)
            # print("imu finish")

    def run_calc_c2m_in_loop(self):
        while True:
            self.calc_c2m(self.params)

    def start_imu_thread(self):
        thread = threading.Thread(target=self.run_imu_in_loop, args=(self.params,))
        thread.start()

    def start_calc_c2m_thread(self):
        thread = threading.Thread(target=self.run_calc_c2m_in_loop)
        thread.start()

    def filt(self, params):
        if params['mode']['filter']:
            if params['mode']['filter'] == 1:
                if params['needle_marker']['shape']:
                    self.filter.filt_marker_qxyz(self.needle_marker)
                if params['show_print']:
                    print("needle_marker's filtered transform matrix:")
                    print(self.needle_marker.matrix)
            elif params['mode']['filter'] == 2:
                pass
            else:
                if params['show_print']:
                    print("Filter failed, filter mode is wrong")
        else:
            pass



if __name__ == "__main__":
    ogs = OGS()
    if ogs.imu is None:
        while True:
            ogs.calc_c2m(ogs.params)
            ogs.filt(ogs.params)
    else:
        ogs.start_imu_thread()
        ogs.start_calc_c2m_thread()
        start_time = datetime.now()  # 记录开始时间
        last_imu_queue_length = 0  # 记录上一次imu队列的长度
        while True:
            ogs.filter.filt(ogs.imu_data_queue,ogs.needle_marker_queue)
            # print("imu:"+str(len(ogs.imu_data_queue)))
            # print("needle_marker:"+str(len(ogs.needle_marker_queue)))
            # print("ref_marker:"+str(len(ogs.ref_marker_queue)))
            print("filter:"+str(len(ogs.filter.queue)))
            if ogs.filter.queue:
                print("quaternion:" + str(ogs.filter.queue[-1]['ukf_state'].quaternion))
                print("position:" + str(ogs.filter.queue[-1]['ukf_state'].p))
                print("velocity:" + str(ogs.filter.queue[-1]['ukf_state'].v))
                print("gyro_bias:" + str(ogs.filter.queue[-1]['ukf_state'].b_gyro))
                print("acc_bias:" + str(ogs.filter.queue[-1]['ukf_state'].b_acc))
                 # if len(ogs.imu_data_queue) // 1000 > last_imu_queue_length:
            #     end_time = datetime.now()  # 记录结束时间
            #     duration = end_time - start_time  # 计算用时
            #     print(f"Time taken to add 1000 items to imu_data_queue: {duration}")
            #     start_time = datetime.now()  # 重置开始时间
            #     last_imu_queue_length = len(ogs.imu_data_queue) // 1000
            # if len(ogs.imu_data_queue) == ogs.imu_data_queue.maxlen:
            #     end_time = datetime.now()  # 记录结束时间
            #     total_duration = end_time - start_time  # 计算总用时
            #     print(f"Time taken to fill imu_data_queue: {total_duration}")
            #     sys.exit()  # 完全终止程序

