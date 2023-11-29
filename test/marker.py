import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
import open3d as o3d

#设置了RealSense管道，配置了所需的流，启动了管道，并准备好对帧进行对齐。它为从RealSense相机捕获同步的深度和彩色数据提供了基础
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

def get_aligned_images():
    """
    Retrieves aligned color and depth images along with camera intrinsic parameters.

    Returns:
        color_image (numpy.ndarray): Aligned color image.
        depth_image (numpy.ndarray): Aligned depth image.
        depth_image_8bit (numpy.ndarray): Aligned depth image converted to 8-bit.
        intr_matrix (numpy.ndarray): Camera intrinsic matrix.
        coeffs (numpy.ndarray): Camera distortion coefficients.
    """
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    intr_matrix = np.array([
        [intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]
    ])
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
    pos = np.where(depth_image_8bit == 0)
    depth_image_8bit[pos] = 255
    color_image = np.asanyarray(color_frame.get_data())
    return color_image, depth_image, depth_image_8bit, intr_matrix, np.array(intr.coeffs)

if __name__ == "__main__":
    n = 0
    while True:
        rgb, depth, depth_8bit, intr_matrix, intr_coeffs = get_aligned_images()
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()
        corners, ids, rejected_img_points = aruco.detectMarkers(
            rgb, aruco_dict, parameters=parameters)
        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(
            corners, 0.045, intr_matrix, intr_coeffs)
        try:
            aruco.drawDetectedMarkers(rgb, corners)
            cv2.drawFrameAxes(rgb, intr_matrix, intr_coeffs, rvec, tvec, 0.05)
            
            if ids is not None and len(ids) > 0:
                # 获取第一个检测到的ArUco标记的变换矩阵
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                    corners, 0.045, intr_matrix, intr_coeffs)
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                transformation_matrix = np.eye(4)
                transformation_matrix[:3, :3] = rotation_matrix
                transformation_matrix[:3, 3] = tvec[0]

                # 定义线段的起点和方向
                start_point = np.array([0, 0, -0.017, 1])  # 1 cm along -z
                end_point = np.array([0, -0.3, -0.017, 1])    # 20 cm along -y

                # 将点变换到相机帧
                start_point_cam = np.dot(transformation_matrix, start_point)
                end_point_cam = np.dot(transformation_matrix, end_point)

                # 将三维点投影到二维图像坐标
                start_point_img = np.dot(intr_matrix, start_point_cam[:3])
                start_point_img = start_point_img / start_point_img[2]
                end_point_img = np.dot(intr_matrix, end_point_cam[:3])
                end_point_img = end_point_img / end_point_img[2]

                # 绘制线段
                cv2.line(rgb, (int(start_point_img[0]), int(start_point_img[1])),
                         (int(end_point_img[0]), int(end_point_img[1])), (255, 255, 255), 2)
                
                #计算这两个点在相机坐标系下的直线方程
                direction_vector = end_point_cam[:3] - start_point_cam[:3]
                point = start_point_cam[:3]
                print("Line equation:")
                print("(x - " + str(point[0]) + ") / " + str(direction_vector[0]) + " = (y - " + str(point[1]) + ") / " + str(direction_vector[1]) + " = (z - " + str(point[2]) + ") / " + str(direction_vector[2]))
                

                
                # Print the transformation matrix  
                print("Transformation Matrix (ArUco center to camera):")
                print(transformation_matrix)

            # 显示图
            cv2.imshow('Depth image', depth_8bit)
            cv2.imshow('RGB image', rgb)
            
            # # Create the point cloud
            # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            #     o3d.geometry.Image(rgb), o3d.geometry.Image(depth_8bit),
            #     depth_scale=1.0, depth_trunc=1.0, convert_rgb_to_intensity=False)
            # pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            #     640, 480, intr_matrix[0, 0], intr_matrix[1, 1], intr_matrix[0, 2], intr_matrix[1, 2]
            # )
            # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            #     rgbd_image, pinhole_camera_intrinsic)
            
            # # Visualize the point cloud
            # o3d.visualization.draw_geometries([pcd])
            
        except:
            cv2.imshow('RGB image', rgb)
            
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('q') or key == 27:
            pipeline.stop()
            break
            
        elif key == ord('s'):
            n = n + 1
            cv2.imwrite('./img/rgb' + str(n) + '.jpg', rgb)

    cv2.destroyAllWindows()
    
    