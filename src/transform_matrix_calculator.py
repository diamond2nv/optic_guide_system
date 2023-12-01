import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
import time

CAMERA_RESOLUTION_X = 1280
CAMERA_RESOLUTION_Y = 720
CAMERA_FRAME_RATE=30

SHOW_IMAGE=False
NEEDLE_MARKER_LENGTH=0.008
REF_MARKER_LENGTH=0.045
NEEDLE_LENGTH=0.009


class TransformMatrixCalculator():
    def __init__(self) -> None:
        """
        Initializes the TransformMatrixCalculator object.

        This constructor sets up the pipeline and configuration for the Intel RealSense camera,
        and initializes various parameters used for marker detection.

        Args:
            None

        Returns:
            None
        """
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, CAMERA_RESOLUTION_X, CAMERA_RESOLUTION_Y, rs.format.z16, CAMERA_FRAME_RATE)
        self.config.enable_stream(rs.stream.color, CAMERA_RESOLUTION_X, CAMERA_RESOLUTION_Y, rs.format.bgr8, CAMERA_FRAME_RATE)
        self.profile = self.pipeline.start(self.config)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.parameters = aruco.DetectorParameters()
        self.transform_matrix_4 = None
        self.transform_matrix_5 = None
        self.start_point_img = None
        self.end_point_img = None
        
    def get_aligned_images(self):
        """
        Retrieves aligned color and depth images along with their corresponding intrinsics and depth scaling.

        Returns:
            color_image (numpy.ndarray): Aligned color image.
            depth_image (numpy.ndarray): Aligned depth image.
            depth_image_8bit (numpy.ndarray): Aligned depth image converted to 8-bit.
            intr_matrix (numpy.ndarray): Intrinsic matrix of the color camera.
            coeffs (numpy.ndarray): Distortion coefficients of the color camera.
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
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
        self.intr_matrix = intr_matrix
        return color_image, depth_image, depth_image_8bit, intr_matrix, np.array(intr.coeffs)

    def calculate_transform_matrix(self,aruco_dict,markerLength):
        """
        Calculates the transformation matrix based on the detected ArUco markers.

        Args:
            aruco_dict (cv2.aruco.Dictionary): ArUco dictionary used for marker detection.
            markerLength (float): Length of the marker in meters.

        Returns:
            transform_matrix (numpy.ndarray): Transformation matrix.
        """
        start_point=None
        end_point=None
        rgb, depth, depth_8bit, intr_matrix, intr_coeffs = self.get_aligned_images()
        corners, ids, rejected_img_points = aruco.detectMarkers(
            rgb, aruco_dict, parameters=self.parameters)
        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(
            corners, markerLength, intr_matrix, intr_coeffs)
        try:
            aruco.drawDetectedMarkers(rgb, corners)
            cv2.drawFrameAxes(rgb, intr_matrix, intr_coeffs, rvec, tvec, 0.05)

            if ids is not None and len(ids) > 0:
                # Get the transformation matrix of the first detected ArUco marker
                
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rotation_matrix
                transform_matrix[:3, 3] = tvec[0]
                             
                #使用transform_matrix计算marker和相机的距离
                distance = np.sqrt(np.sum(np.square(tvec[0])))
                print("distance:",distance)
                
                
                if aruco_dict.markerSize == 4:
                    self.transform_matrix_4 = transform_matrix
                    print("transform_matrix_4:",transform_matrix)
                    
                if aruco_dict.markerSize == 5:
                    self.transform_matrix_5 = transform_matrix
                    print("transform_matrix_5:",transform_matrix)
                else:
                    return
                return transform_matrix
            else:
                print("no change")

        except:
            pass
        return None
        
    def get_transformed_needle_points(self,aruco_dict):
        """
        Calculates the transformed needle points based on the transformation matrix.

        Args:
            aruco_dict (cv2.aruco.Dictionary): ArUco dictionary used for marker detection.

        Returns:
            None
        """
        if self.transform_matrix_4 is None and self.transform_matrix_5 is None:
            return None, None
        if aruco_dict.markerSize == 4:
            transformation_matrix = self.transform_matrix_4
            
        if aruco_dict.markerSize == 5:
            transformation_matrix = self.transform_matrix_5
        
        # Define the start point and direction of the line segment
        start_point = np.array([0, 0, -0.001, 1])  # 1 cm along -z
        end_point = np.array([0, NEEDLE_LENGTH, -0.001, 1])  # 20 cm along -y
        # Transform the points to the camera frame
        start_point_cam = np.dot(transformation_matrix, start_point)
        end_point_cam = np.dot(transformation_matrix, end_point)

        # Project the 3D points onto the 2D image coordinates
        start_point_img = np.dot(self.intr_matrix, start_point_cam[:3])
        start_point_img = start_point_img / start_point_img[2]
        end_point_img = np.dot(self.intr_matrix, end_point_cam[:3])
        end_point_img = end_point_img / end_point_img[2]
        
        start_point_img = start_point_img[:2]
        end_point_img = end_point_img[:2]
        
        #强制将start_point_img和end_point_img转换为非负数
        start_point_img = start_point_img.astype(np.int32)
        end_point_img = end_point_img.astype(np.int32)
        
        #对start_point_img和end_point_img取绝对值
        start_point_img = np.abs(start_point_img)
        end_point_img = np.abs(end_point_img)
        
        #当self.start_point_img和self.end_point_img不同时，更新self.start_point_img和self.end_point_img
        if (self.start_point_img != start_point_img).any() or (self.end_point_img != end_point_img).any():
            self.start_point_img = start_point_img
            self.end_point_img = end_point_img
            print("start_point_img:",start_point_img,"end_point_img:",end_point_img)
        
        return start_point_img, end_point_img
        
        
if __name__ == "__main__":
    aruco_dict_4 = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    aruco_dict_5 = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
    transform_matrix_calculator = TransformMatrixCalculator()
    while(1):
        start = time.time()
        # transform_matrix_calculator.calculate_transform_matrix(aruco_dict_4,REF_MARKER_LENGTH)
        trasform_matrix=transform_matrix_calculator.calculate_transform_matrix(aruco_dict_5,NEEDLE_MARKER_LENGTH)
        start_point, end_point = transform_matrix_calculator.get_transformed_needle_points(aruco_dict_5)
        end = time.time()
        fps=int(1/(end-start))
        if SHOW_IMAGE:
            rgb, depth, depth_8bit, intr_matrix, intr_coeffs = transform_matrix_calculator.get_aligned_images()
            corners, ids, rejected_img_points = aruco.detectMarkers(
                rgb, aruco_dict_4, parameters=transform_matrix_calculator.parameters)
            aruco.drawDetectedMarkers(rgb, corners)
            cv2.putText(rgb, "FPS: {:.0f}".format(fps), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if trasform_matrix is not None:
                cv2.putText(rgb, "distance: {:.5f}".format(np.sqrt(np.sum(np.square(trasform_matrix[:3,3])))), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if start_point is not None and end_point is not None:
                    cv2.line(rgb, start_point, end_point, (255, 255, 255), 2)
                    cv2.putText(rgb, "start_point: {:.0f},{:.0f}".format(start_point[0],start_point[1]), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(rgb, "end_point: {:.0f},{:.0f}".format(end_point[0],end_point[1]), (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('rgb',rgb)
            cv2.waitKey(1)
        
        

         
