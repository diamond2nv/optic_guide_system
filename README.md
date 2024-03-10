# Optic_Guide_System

## 使用指南

### 硬件
* [Intel Realsense D435i相机](https://www.intelrealsense.com/depth-camera-d435i/)
* [Marker](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
* [WHEELTEC IMU](https://3.cn/1-WigwxM)

### 安装

```bash
mkdir optic_guide_system
cd optic_guide_system
git clone ...
conda create -n optic_guide_system python=3.10
conda activate optic_guide_system
conda env create -f environment.yaml
```

## 功能简述

* 使用intel realsense系列相机，识别Marker，并实现对Marker位姿的识别和解算
* Online & Offline模式支持，可使用脚本保存数据集
* 多线程处理
* G-H滤波器降噪
* 无迹卡尔曼滤波器数据融合，高速高精度Marker追踪

## 自用记录：
ukfm： Time taken to filter: 0:01:37.017488
       Time taken to filter: 0:01:58.445728
IMU： Time taken to fill imu_data_queue: 0:00:04.306781
      Time taken to fill imu_data_queue: 0:00:04.319105

