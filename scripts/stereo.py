"""
author: Yan-Shan Lu
date: 2025/11/03
"""
import cv2
import numpy as np
from skimage.segmentation import slic
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
import cv2.ximgproc as ximg
import os
import time

def compute_initial_disparity(left_image, right_image, num_disp=64):
    """
    Compute an initial disparity map
    using OpenCV's modified H. Hirschmuller algorithm for performance concern.

    :param left_image: Left color image
    :param right_image: Right color image
    :param num_disp: Number of disparities, maximum distance between the same pixel in both images

    :return disparity: Initial disparity map
    :return sgbm: Stereo configuration
    """
    block_size = 5
    sgbm = cv2.StereoSGBM.create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=4 * 3 * block_size**2,
        P2=38 * 3 * block_size**2,
        uniquenessRatio=1,
        disp12MaxDiff=1,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_HH
    )
    disparity = sgbm.compute(left_image, right_image).astype(np.float32) / 16.0

    return disparity, sgbm

def generate_superpixels(image, n_segments=4000):
    """
    Segment the reference image into superpixels using the SLIC algorithm.

    :param image: Reference image
    :param n_segments: Number of segments for initial number of clusters

    :return: Segmentation image with labeled pixels
    """
    segments = slic(image, n_segments=n_segments, compactness=10, slic_zero=True)
    return segments

def fit_planes_to_superpixels(disparity, segments, occlusion_mask):
    """
    Refine the disparity map within each segment using RANSAC plane fitting.

    :param disparity: Initial disparity map
    :param segments: Segmentation map
    :param occlusion_mask: Occlusion mask image

    :return: Refined disparity map
    """
    refined_disparity = np.copy(disparity)
    inlier_thresh = 50  # ensure sufficient inliers

    for segment_id in np.unique(segments):
        mask = (segments == segment_id) & ~occlusion_mask  # consider only non-occluded regions
        if np.sum(mask) > inlier_thresh:
            y, x = np.where(mask)
            disparities = disparity[mask]

            # fit plane using RANSAC
            ransac = RANSACRegressor()
            ransac.fit(np.c_[x, y], disparities)
            coef = ransac.estimator_.coef_
            intercept = ransac.estimator_.intercept_

            # update disparity for this segment
            fitted_disparity = coef[0] * x + coef[1] * y + intercept
            refined_disparity[mask] = fitted_disparity

    return refined_disparity

def handle_occlusion(refined_disparity, occlusion_mask, left_image, right_image, sgbm, num_disp):
    """
    Fill occluded regions using WLS filtering.

    :param refined_disparity: Refined disparity map
    :param occlusion_mask: Occlusion mask image
    :param left_image: Left color image
    :param right_image: Right color image
    :param sgbm: Stereo configuration
    :param num_disp: Number of disparities, maximum distance between the same pixel in both images

    :return: Occlusion filled disparity map
    """
    left_disp = refined_disparity.copy()

    # add back the leftmost invalid pixels (ignored in refinement)
    invalid_px = np.zeros((left_disp.shape[0], num_disp), dtype=np.float32)
    left_disp = np.concatenate((invalid_px, refined_disparity), axis=1)
    # convert to grayscale images
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)

    # create right matcher
    left_matcher = sgbm
    right_matcher = ximg.createRightMatcher(left_matcher)
    right_disp = right_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0

    # create WLS filter
    wls_sigma = 1.0
    wls_lamda = 8000.0
    wls_filter = ximg.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(wls_lamda)
    wls_filter.setSigmaColor(wls_sigma)

    wls_disp = wls_filter.filter(left_disp, left_gray, disparity_map_right=right_disp)

    # check for NaNs
    if np.isnan(wls_disp).any():
        print("NaN detected, replacing with 0")
        wls_disp = np.nan_to_num(wls_disp, nan=0)

    # Fill occluded regions
    wls_disp_in_roi = wls_disp[:, num_disp:]
    filled_disp_in_roi = left_disp[:, num_disp:]
    filled_disp_in_roi[occlusion_mask] = wls_disp_in_roi[occlusion_mask]

    return filled_disp_in_roi


def stereo(save_img:bool=False, visualize:bool=True):
    """
    Main function applying the proposed algorithm.

    :param save_img: Whether to save the output disparity map
    :param visualize: Whether to visualize the output disparity map
    :return: void
    """
    img_dir = "../images"
    output_dir = "../outputs"
    os.makedirs(output_dir, exist_ok=True)
    scene = "devon1"

    num_disp = 200
    n_segments = 4000

    left_path = os.path.join(img_dir, f'{scene}-left.ppm')
    right_path = os.path.join(img_dir, f'{scene}-right.ppm')

    # start the timer
    start_time = time.time()

    # load stereo images
    print("Loading images....")
    left_img = cv2.imread(left_path, cv2.IMREAD_COLOR)
    right_img = cv2.imread(right_path, cv2.IMREAD_COLOR)
    if left_img is None or right_img is None:
        print(f"Missing image(s) in {scene}")
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

    print("\nComputing initial disparity map...")
    initial_disp, sgbm = compute_initial_disparity(left_img, right_img, num_disp=num_disp)

    # consider only valid disparity regions
    left_img_in_roi = left_img[:, num_disp:]
    valid_disp = initial_disp[:, num_disp:]
    occlusions = valid_disp < 0  # occlusion mask

    valid_disp = cv2.medianBlur(valid_disp, 5)

    print("\nApplying superpixel segmentation...")
    superpixels = generate_superpixels(left_img_in_roi, n_segments=n_segments)

    print("\nRANSAC plane fitting....")
    refined_disp = fit_planes_to_superpixels(valid_disp, superpixels, occlusions)

    print("\nOcclusion handling....")
    refined_disp = handle_occlusion(refined_disp, occlusions, left_img, right_img,
                                                  sgbm, num_disp=num_disp)
    refined_disp = cv2.medianBlur(refined_disp, 5)

    # calculate runtime
    end_time = time.time()
    runtime = end_time - start_time
    print("\nFinished.")
    print(f"Runtime: {runtime:.2f} seconds")

    # save the disparity map
    if save_img:
        save_path = os.path.join(output_dir, f"{scene}_disparity.pfm")
        cv2.imwrite(save_path, refined_disp)
        print(f"Saved to {save_path}")

    if visualize:
        plt.figure(figsize=(8, 4))
        plt.subplot(121);plt.axis("off")
        plt.imshow(left_img_in_roi);plt.title(f"Left Image")
        plt.subplot(122);plt.axis("off")
        plt.imshow(refined_disp);plt.title(f"Optimized Disparity")
        plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    stereo(save_img=True, visualize=True)

