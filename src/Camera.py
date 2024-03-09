import pyrealsense2 as rs
import cv2.aruco as aruco
import cv2
import numpy as np


class Camera():
    def __init__(self, params):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, params['camera']['resolution_x'], params['camera']['resolution_y'],
                                  rs.format.z16, params['camera']['frame_rate'])
        self.config.enable_stream(rs.stream.color, params['camera']['resolution_x'], params['camera']['resolution_y'],
                                  rs.format.bgr8, params['camera']['frame_rate'])
        self.profile = self.pipeline.start(self.config)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.parameters = aruco.DetectorParameters()
        self.color_image = None
        self.depth_image = None
        self.depth_image_8bit = None
        self.intr_matrix = None

    def get_aligned_images(self):
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
        self.color_image = color_image
        self.depth_image = depth_image
        self.depth_image_8bit = depth_image_8bit
        self.intr_coeffs = np.array(intr.coeffs)
        return color_image, depth_image, depth_image_8bit, intr_matrix, np.array(intr.coeffs)

    def trans_matrix_calc(self, marker):
        self.get_aligned_images()
        corners, ids, rejected_img_points = aruco.detectMarkers(
            self.color_image, marker.shape, parameters=self.parameters)
        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(
            corners, marker.length, self.intr_matrix, self.intr_coeffs)
        try:
            if ids is not None and len(ids) > 0:
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rotation_matrix
                transform_matrix[:3, 3] = tvec[0]

                old_qxyz = marker.qxyz
                old_v_qxyz = marker.v_qxyz
                old_acc_qxyz = marker.acc_qxyz

                marker.matrix = transform_matrix
                marker.qxyz_update_from_matrix()

                marker.v_qxyz = marker.qxyz - old_qxyz
                marker.acc_qxyz = marker.v_qxyz - old_v_qxyz

            else:
                marker.matrix = np.full((4, 4), np.nan)
                marker.qxyz = np.full((7), np.nan)
            return marker.matrix, marker.qxyz
        except:
            pass
