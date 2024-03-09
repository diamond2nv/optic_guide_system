#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

# import numpy as np
import sys

import serial  # 导入模块
import serial.tools.list_ports
import threading
import struct
import time
import platform
# from copy import deepcopy
# import sys
# import os
# import math

from serial import EIGHTBITS, PARITY_NONE, STOPBITS_ONE

FRAME_HEAD = str('fc')
FRAME_END = str('fd')
TYPE_IMU = str('40')
TYPE_AHRS = str('41')
TYPE_INSGPS = str('42')
TYPE_GEODETIC_POS = str('5c')
TYPE_GROUND = str('f0')
TYPE_SYS_STATE = str('50')
TYPE_BODY_ACCELERATION = str('62')
TYPE_ACCELERATION = str('61')
IMU_LEN = str('38')  # //56
AHRS_LEN = str('30')  # //48
INSGPS_LEN = str('48')  # //72
GEODETIC_POS_LEN = str('20')  # //32
SYS_STATE_LEN = str('64')  # // 100
BODY_ACCELERATION_LEN = str('10')  # // 16
ACCELERATION_LEN = str('0c')  # 12
PI = 3.141592653589793
DEG_TO_RAD = 0.017453292519943295
isrun = True


class Imu:
    # def __init__(self,params):
    #     self.port=params['imu']['port']
    #     self.bps=params['imu']['bps']
    #     self.timeout=params['imu']['timeout']
    def __init__(self) -> None:
        self.serial_ = None
        self.port = '/dev/ttyUSB0'
        self.bps = 921600
        self.timeout = 20
        self.showprint = True
        self.gyroscope = {"x": 0, "y": 0, "z": 0}
        self.accelerometer = {"x": 0, "y": 0, "z": 0}
        self.Magnetometer = {"x": 0, "y": 0, "z": 0}
        self.ahrs = {}
        self.timestamp = 0

    def UsePlatform(self):
        sys_str = platform.system()
        if self.showprint:
            if sys_str == "Windows":
                print("Call Windows tasks")
            elif sys_str == "Linux":
                print("Call Linux tasks")
            else:
                print("Other System tasks: %s" % sys_str)
        return sys_str

    def find_serial(self):
        port_list = list(serial.tools.list_ports.comports())
        for port in port_list:
            if port.device == self.port:
                return True
        return False

    def open_port(self):
        if self.find_serial():
            if self.showprint:
                print("find this port : " + self.port)
        else:
            if self.showprint:
                print("error:  unable to find this port : " + self.port)
            exit(1)
        try:
            self.serial_ = serial.Serial(port=self.port, baudrate=self.bps, bytesize=EIGHTBITS, parity=PARITY_NONE,
                                         stopbits=STOPBITS_ONE,
                                         timeout=self.timeout)
            if self.showprint:
                print("baud rates = " + str(self.serial_.baudrate))
        except:
            if self.showprint:
                print("error:  unable to open port .")
            exit(1)

    def receive_data(self):
        self.open_port()
        # 尝试打开串口

        # 循环读取数据

        while self.serial_.isOpen():
            # rbdata = ser.readline()
            # # rbdata = ser.read_all()
            #
            # if len(rbdata) != 0:
            #     rxdata = rbdata.hex()
            #     print(rxdata)
            if not threading.main_thread().is_alive():
                if self.showprint:
                    print('done')
                break
            check_head = self.serial_.read().hex()
            # 校验帧头
            if check_head != FRAME_HEAD:
                continue
            head_type = self.serial_.read().hex()
            # 校验数据类型
            if (head_type != TYPE_IMU and head_type != TYPE_AHRS and head_type != TYPE_INSGPS and
                    head_type != TYPE_GEODETIC_POS and head_type != 0x50 and head_type != TYPE_GROUND and
                    head_type != TYPE_SYS_STATE and head_type != TYPE_BODY_ACCELERATION and head_type != TYPE_ACCELERATION):
                continue
            check_len = self.serial_.read().hex()
            # 校验数据类型的长度
            if head_type == TYPE_IMU and check_len != IMU_LEN:
                continue
            elif head_type == TYPE_AHRS and check_len != AHRS_LEN:
                continue
            elif head_type == TYPE_INSGPS and check_len != INSGPS_LEN:
                continue
            elif head_type == TYPE_GEODETIC_POS and check_len != GEODETIC_POS_LEN:
                continue
            elif head_type == TYPE_SYS_STATE and check_len != SYS_STATE_LEN:
                continue
            elif head_type == TYPE_GROUND or head_type == 0x50:
                continue
            elif head_type == TYPE_BODY_ACCELERATION and check_len != BODY_ACCELERATION_LEN:
                if self.showprint:
                    print(
                        "check head type " + str(TYPE_BODY_ACCELERATION) + " failed;" + " check_LEN:" + str(check_len))
                continue
            elif head_type == TYPE_ACCELERATION and check_len != ACCELERATION_LEN:
                if self.showprint:
                    print("check head type " + str(TYPE_ACCELERATION) + " failed;" + " ckeck_LEN:" + str(check_len))
                continue
            check_sn = self.serial_.read().hex()
            head_crc8 = self.serial_.read().hex()
            crc16_H_s = self.serial_.read().hex()
            crc16_L_s = self.serial_.read().hex()

            # 读取并解析IMU数据
            if head_type == TYPE_IMU:
                data_s = self.serial_.read(int(IMU_LEN, 16))
                IMU_DATA = struct.unpack('12f ii', data_s[0:56])
                self.gyroscope["x"] = IMU_DATA[0]
                self.gyroscope["y"] = IMU_DATA[1]
                self.gyroscope["z"] = IMU_DATA[2]
                self.accelerometer["x"] = IMU_DATA[3]
                self.accelerometer["y"] = IMU_DATA[4]
                self.accelerometer["z"] = IMU_DATA[5]
                self.Magnetometer["x"] = IMU_DATA[6]
                self.Magnetometer["y"] = IMU_DATA[7]
                self.Magnetometer["z"] = IMU_DATA[8]
                self.timestamp = IMU_DATA[12]
                if self.showprint:
                    print(IMU_DATA)
                    print("Gyroscope_X(rad/s): " + str(IMU_DATA[0]))
                    print("Gyroscope_Y(rad/s) : " + str(IMU_DATA[1]))
                    print("Gyroscope_Z(rad/s) : " + str(IMU_DATA[2]))
                    print("Accelerometer_X(m/s^2) : " + str(IMU_DATA[3]))
                    print("Accelerometer_Y(m/s^2) : " + str(IMU_DATA[4]))
                    print("Accelerometer_Z(m/s^2) : " + str(IMU_DATA[5]))
                    print("Magnetometer_X(mG) : " + str(IMU_DATA[6]))
                    print("Magnetometer_Y(mG) : " + str(IMU_DATA[7]))
                    print("Magnetometer_Z(mG) : " + str(IMU_DATA[8]))
                    print("IMU_Temperature : " + str(IMU_DATA[9]))
                    print("Pressure : " + str(IMU_DATA[10]))
                    print("Pressure_Temperature : " + str(IMU_DATA[11]))
                    print("Timestamp(us) : " + str(IMU_DATA[12]))
            # 读取并解析AHRS数据
            elif head_type == TYPE_AHRS:
                data_s = self.serial_.read(int(AHRS_LEN, 16))
                AHRS_DATA = struct.unpack('10f ii', data_s[0:48])
                self.ahrs["RollSpeed"] = AHRS_DATA[0]
                self.ahrs["PitchSpeed"] = AHRS_DATA[1]
                self.ahrs["HeadingSpeed"] = AHRS_DATA[2]
                self.ahrs["Roll"] = AHRS_DATA[3]
                self.ahrs["Pitch"] = AHRS_DATA[4]
                self.ahrs["Heading"] = AHRS_DATA[5]
                self.ahrs["Q1"] = AHRS_DATA[6]
                self.ahrs["Q2"] = AHRS_DATA[7]
                self.ahrs["Q3"] = AHRS_DATA[8]
                self.ahrs["Q4"] = AHRS_DATA[9]
                self.timestamp = AHRS_DATA[10]
                if self.showprint:
                    print(AHRS_DATA)
                    print("RollSpeed(rad/s): " + str(AHRS_DATA[0]))
                    print("PitchSpeed(rad/s) : " + str(AHRS_DATA[1]))
                    print("HeadingSpeed(rad) : " + str(AHRS_DATA[2]))
                    print("Roll(rad) : " + str(AHRS_DATA[3]))
                    print("Pitch(rad) : " + str(AHRS_DATA[4]))
                    print("Heading(rad) : " + str(AHRS_DATA[5]))
                    print("Q1 : " + str(AHRS_DATA[6]))
                    print("Q2 : " + str(AHRS_DATA[7]))
                    print("Q3 : " + str(AHRS_DATA[8]))
                    print("Q4 : " + str(AHRS_DATA[9]))
                    print("Timestamp(us) : " + str(AHRS_DATA[10]))
            # 读取并解析INSGPS数据
            elif head_type == TYPE_INSGPS:
                data_s = self.serial_.read(int(INSGPS_LEN, 16))
                INSGPS_DATA = struct.unpack('16f ii', data_s[0:72])
                if self.showprint:
                    print(INSGPS_DATA)
                    print("BodyVelocity_X:(m/s)" + str(INSGPS_DATA[0]))
                    print("BodyVelocity_Y:(m/s)" + str(INSGPS_DATA[1]))
                    print("BodyVelocity_Z:(m/s)" + str(INSGPS_DATA[2]))
                    print("BodyAcceleration_X:(m/s^2)" + str(INSGPS_DATA[3]))
                    print("BodyAcceleration_Y:(m/s^2)" + str(INSGPS_DATA[4]))
                    print("BodyAcceleration_Z:(m/s^2)" + str(INSGPS_DATA[5]))
                    print("Location_North:(m)" + str(INSGPS_DATA[6]))
                    print("Location_East:(m)" + str(INSGPS_DATA[7]))
                    print("Location_Down:(m)" + str(INSGPS_DATA[8]))
                    print("Velocity_North:(m)" + str(INSGPS_DATA[9]))
                    print("Velocity_East:(m/s)" + str(INSGPS_DATA[10]))
                    print("Velocity_Down:(m/s)" + str(INSGPS_DATA[11]))
                    print("Acceleration_North:(m/s^2)" + str(INSGPS_DATA[12]))
                    print("Acceleration_East:(m/s^2)" + str(INSGPS_DATA[13]))
                    print("Acceleration_Down:(m/s^2)" + str(INSGPS_DATA[14]))
                    print("Pressure_Altitude:(m)" + str(INSGPS_DATA[15]))
                    print("Timestamp:(us)" + str(INSGPS_DATA[16]))
            # 读取并解析GPS数据
            elif head_type == TYPE_GEODETIC_POS:
                data_s = self.serial_.read(int(GEODETIC_POS_LEN, 16))
                if self.showprint:
                    print(" Latitude:(rad)" + str(struct.unpack('d', data_s[0:8])[0]))
                    print("Longitude:(rad)" + str(struct.unpack('d', data_s[8:16])[0]))
                    print("Height:(m)" + str(struct.unpack('d', data_s[16:24])[0]))
            elif head_type == TYPE_SYS_STATE:
                data_s = self.serial_.read(int(SYS_STATE_LEN, 16))
                if self.showprint:
                    print("Unix_time:" + str(struct.unpack('i', data_s[4:8])[0]))
                    print("Microseconds:" + str(struct.unpack('i', data_s[8:12])[0]))
                    print(" System_status:" + str(struct.unpack('d', data_s[0:2])[0]))
                    print("System_Z(m/s^2): " + str(struct.unpack('f', data_s[56:60])[0]))
            elif head_type == TYPE_BODY_ACCELERATION:
                data_s = self.serial_.read(int(BODY_ACCELERATION_LEN, 16))
                if self.showprint:
                    print(" System_status:" + str(struct.unpack('d', data_s[0:2])[0]))
                    print("BodyAcceleration_Z(m/s^2): " + str(struct.unpack('f', data_s[8:12])[0]))
            elif head_type == TYPE_ACCELERATION:
                data_s = self.serial_.read(int(ACCELERATION_LEN, 16))
                if self.showprint:
                    print(" System_status:" + str(struct.unpack('d', data_s[0:2])[0]))
                    print("Acceleration_Z(m/s^2): " + str(struct.unpack('f', data_s[8:12])[0]))

    def run(self):
        self.UsePlatform()
        tr = threading.Thread(target=self.receive_data)
        tr.start()
        while True:
            try:
                if tr.is_alive():
                    time.sleep(1)
                else:
                    break
            except(KeyboardInterrupt, SystemExit):
                break


def main():
    imu = Imu()
    imu.run()


if __name__ == "__main__":
    main()
