import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
import open3d as o3d

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

def get_aligned_images():
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
        aruco_dict_4x4 = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        aruco_dict_5x5 = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
        parameters = aruco.DetectorParameters()

        # Detect 4X4 ArUco markers
        corners_4x4, ids_4x4, _ = aruco.detectMarkers(rgb, aruco_dict_4x4, parameters=parameters)
        rvec_4x4, tvec_4x4, _ = aruco.estimatePoseSingleMarkers(corners_4x4, 0.045, intr_matrix, intr_coeffs)

        # Detect 5X5 ArUco markers
        corners_5x5, ids_5x5, _ = aruco.detectMarkers(rgb, aruco_dict_5x5, parameters=parameters)
        rvec_5x5, tvec_5x5, _ = aruco.estimatePoseSingleMarkers(corners_5x5, 0.045, intr_matrix, intr_coeffs)

        try:
            aruco.drawDetectedMarkers(rgb, corners_4x4)
            aruco.drawDetectedMarkers(rgb, corners_5x5)

            if ids_4x4 is not None and len(ids_4x4) > 0 and ids_5x5 is not None and len(ids_5x5) > 0:
                # Assuming the first marker of each type is the one you want to compare
                rvec_4x4, tvec_4x4, _ = aruco.estimatePoseSingleMarkers(corners_4x4, 0.045, intr_matrix, intr_coeffs)
                rvec_5x5, tvec_5x5, _ = aruco.estimatePoseSingleMarkers(corners_5x5, 0.045, intr_matrix, intr_coeffs)

                # Calculate transformation matrices
                transformation_matrix_4x4 = np.eye(4)
                rotation_matrix_4x4 , _ = cv2.Rodrigues(rvec_4x4)
                transformation_matrix_4x4[:3, :3]  = rotation_matrix_4x4
                transformation_matrix_4x4[:3, 3] = tvec_4x4[0]

                transformation_matrix_5x5 = np.eye(4)
                rotation_matrix_5x5 , _ = cv2.Rodrigues(rvec_5x5)
                transformation_matrix_5x5[:3, :3] = rotation_matrix_5x5
                transformation_matrix_5x5[:3, 3] = tvec_5x5[0]
                # Print transformation matrices
                trans4to5 = transformation_matrix_4x4@transformation_matrix_5x5
                print("Transformation Matrix 4X4 to 5X5:")
                print(trans4to5)


            # Display images
            cv2.imshow('Depth image', depth_8bit)
            cv2.imshow('RGB image', rgb)

            # Display depth image for ArUco regions
            depth_8bit = cv2.convertScaleAbs(depth, alpha=0.03)
            pos = np.where(depth_8bit == 0)
            depth_8bit[pos] = 255
            cv2.imshow('ArUco depth', depth_8bit)

        except Exception as e:
            print(f"Error: {e}")
            cv2.imshow('RGB image', rgb)

        key = cv2.waitKey(1)

        if key & 0xFF == ord('q') or key == 27:
            pipeline.stop()
            break

        elif key == ord('s'):
            n = n + 1
            cv2.imwrite('./img/rgb' + str(n) + '.jpg', rgb)

    cv2.destroyAllWindows()
