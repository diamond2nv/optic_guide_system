import pyrealsense2 as rs
import numpy as np
import cv2

# 创建一个管道
pipeline = rs.pipeline()

# 创建一个配置并配置管道以从设备流式传输
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# 开始流式传输
pipeline.start(config)

try:
    while True:
        # 等待一组帧
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # 将深度帧和颜色帧转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 将深度图像转换为彩色图像以便可视化
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 堆叠图像
        images = np.hstack((color_image, depth_colormap))

        # 显示图像
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:
    # 停止流式传输
    pipeline.stop()