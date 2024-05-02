import cv2 
import numpy as np

def convert_pixel_distance_to_meters(pixel_distance, refrence_height_in_meters, refrence_height_in_pixels):
    return (pixel_distance * refrence_height_in_meters) / refrence_height_in_pixels

def convert_meters_to_pixel_distance(meters, refrence_height_in_meters, refrence_height_in_pixels):
    return (meters * refrence_height_in_pixels) / refrence_height_in_meters

def compute_homography(src_points, dst_points):
    """
    Compute the homography matrix from src_points to dst_points.
    :param src_points: Coordinates on the original image.
    :param dst_points: Coordinates on the destination image.
    :return: Homography matrix.
    """
    src_points = np.array(src_points, dtype="float32")
    dst_points = np.array(dst_points, dtype="float32")
    homography_matrix, _ = cv2.findHomography(src_points, dst_points)
    return homography_matrix
