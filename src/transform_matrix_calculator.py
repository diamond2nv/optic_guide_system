import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
import time

CAMERA_RESOLUTION_X = 1280
CAMERA_RESOLUTION_Y = 720
CAMERA_FRAME_RATE=30

SHOW_IMAGE=True
NEEDLE_MARKER_LENGTH=0.004
REF_MARKER_LENGTH=0.008
NEEDLE_LENGTH=0.012185
MARKER_TO_NEEDLE_DEPTH=0.001082-0.000202/2


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
                #print("distance:",distance)
                
                
                if aruco_dict.markerSize == 4:
                    self.transform_matrix_4 = transform_matrix
                    # print("transform_matrix_4:",transform_matrix)
                    
                if aruco_dict.markerSize == 5:
                    self.transform_matrix_5 = transform_matrix
                    #print("transform_matrix_5:",transform_matrix)
                # else:
                #     return
                return transform_matrix
            else:
                print("no change")

        except:
            pass
        return None
        
    def get_transformed_needle_points(self, aruco_dict, start_point=np.array([0, 0, -0.001, 1]), end_point=np.array([0, NEEDLE_LENGTH, -0.001, 1])):
        """
        Calculates the transformed needle points based on the transformation matrix.

        Args:
            aruco_dict (cv2.aruco.Dictionary): ArUco dictionary used for marker detection.
            start_point (np.array, optional): Starting point of the needle in 3D space. Defaults to [0, 0, -0.001, 1].
            end_point (np.array, optional): Ending point of the needle in 3D space. Defaults to [0, NEEDLE_LENGTH, -0.001, 1].

        Returns:
            tuple: A tuple containing the following elements:
                - start_point_img (np.array): Starting point of the needle in image coordinates.
                - end_point_img (np.array): Ending point of the needle in image coordinates.
                - start_point_cam (np.array): Starting point of the needle in camera coordinates.
                - end_point_cam (np.array): Ending point of the needle in camera coordinates.
        """
       
        if self.transform_matrix_4 is not None and aruco_dict.markerSize == 4:
            transformation_matrix = self.transform_matrix_4
        elif self.transform_matrix_5 is not None and aruco_dict.markerSize == 5:
            transformation_matrix = self.transform_matrix_5
        else:
            return None, None, None, None

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
        
        # Force start_point_img and end_point_img to be integers
        start_point_img = start_point_img.astype(np.int32)
        end_point_img = end_point_img.astype(np.int32)
        
        # Take absolute values of start_point_img and end_point_img
        start_point_img = np.abs(start_point_img)
        end_point_img = np.abs(end_point_img)
        
        # Update self.start_point_img and self.end_point_img if they are different from start_point_img and end_point_img
        if (self.start_point_img != start_point_img).any() or (self.end_point_img != end_point_img).any():
            self.start_point_img = start_point_img
            self.end_point_img = end_point_img
        
        return start_point_img, end_point_img, start_point_cam[:3], end_point_cam[:3]
    
def transform_matrix_accuracy_test():
    aruco_dict_4 = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    aruco_dict_5 = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
    transform_matrix_calculator = TransformMatrixCalculator()
    while(1):
        start = time.time()
        transform_matrix_ref=transform_matrix_calculator.calculate_transform_matrix(aruco_dict_5,REF_MARKER_LENGTH)
        transform_matrix_needle=transform_matrix_calculator.calculate_transform_matrix(aruco_dict_4,NEEDLE_MARKER_LENGTH)
        needle_start_point_3d=np.array([0, 0, -MARKER_TO_NEEDLE_DEPTH, 1])
        needle_end_point_3d=np.array([0, NEEDLE_LENGTH, -MARKER_TO_NEEDLE_DEPTH, 1])
        needle_start_point_2d, needle_end_point_2d,_,_ = transform_matrix_calculator.get_transformed_needle_points(aruco_dict_4,needle_start_point_3d,needle_end_point_3d)
        end=time.time()
        _,needle_end_img_2d,_,needle_end_cam_3d = transform_matrix_calculator.get_transformed_needle_points(aruco_dict_4,start_point=np.array([0, 0, -0.001, 1]),end_point = np.array([0, NEEDLE_LENGTH, -0.001, 1]))
        print("needle_end_img_2d:",needle_end_img_2d,"needle_end_cam_3d:",needle_end_cam_3d)
        marker_start_point=np.array([REF_MARKER_LENGTH/2, REF_MARKER_LENGTH/2, 0, 1])
        marker_end_point=np.array([REF_MARKER_LENGTH/2, REF_MARKER_LENGTH/2, 0, 1])
        _,marker_end_img_2d,_,marker_end_cam_3d=transform_matrix_calculator.get_transformed_needle_points(aruco_dict_5,start_point=marker_start_point,end_point=marker_end_point)
        print("marker_end_img_2d:",marker_end_img_2d,"marker_end_cam_3d:",marker_end_cam_3d)
    return None
def main():
    aruco_dict_4 = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    aruco_dict_5 = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
    transform_matrix_calculator = TransformMatrixCalculator()
    while(1):
        start = time.time()
        transform_matrix_ref=transform_matrix_calculator.calculate_transform_matrix(aruco_dict_5,REF_MARKER_LENGTH)
        print("transform_matrix_ref:",transform_matrix_ref)
        transform_matrix_needle=transform_matrix_calculator.calculate_transform_matrix(aruco_dict_4,NEEDLE_MARKER_LENGTH)
        needle_start_point_3d=np.array([0, 0, -MARKER_TO_NEEDLE_DEPTH, 1])
        needle_end_point_3d=np.array([0, NEEDLE_LENGTH, -MARKER_TO_NEEDLE_DEPTH, 1])
        needle_start_point_2d, needle_end_point_2d,_,_ = transform_matrix_calculator.get_transformed_needle_points(aruco_dict_4,needle_start_point_3d,needle_end_point_3d)
        end=time.time()
        fps=int(1/(end-start))
        if SHOW_IMAGE:
            rgb, depth, depth_8bit, intr_matrix, intr_coeffs = transform_matrix_calculator.get_aligned_images()
            corners, ids, rejected_img_points = aruco.detectMarkers(
                rgb, aruco_dict_4, parameters=transform_matrix_calculator.parameters)
            aruco.drawDetectedMarkers(rgb, corners)
            cv2.putText(rgb, "FPS: {:.0f}".format(fps), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if transform_matrix_needle is not None:
                cv2.putText(rgb, "distance: {:.5f}".format(np.sqrt(np.sum(np.square(transform_matrix_needle[:3,3])))), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if needle_start_point_2d is not None and needle_end_point_2d is not None:
                    cv2.line(rgb, needle_start_point_2d, needle_end_point_2d, (255, 255, 255), 2)
                    cv2.putText(rgb, "start_point: {:.2f},{:.2f}".format(needle_start_point_2d[0],needle_start_point_2d[1]), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(rgb, "end_point: {:.2f},{:.2f}".format(needle_end_point_2d[0],needle_end_point_2d[1]), (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('rgb',rgb)
            cv2.waitKey(1)
        
        
if __name__ == "__main__":
    # transform_matrix_accuracy_test()
    
    main()
    
        
        

         
