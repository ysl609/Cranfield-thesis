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

import warnings
from sklearn.exceptions import UndefinedMetricWarning

def load_images(left_path, right_path):
    """
    Read stereo image pair.

    :param left_path: Path to the left image.
    :param right_path: Path to the right image.

    :return: Left and right images.
    """
    left_img = cv2.imread(left_path, cv2.IMREAD_UNCHANGED)
    right_img = cv2.imread(right_path, cv2.IMREAD_UNCHANGED)
    if left_img is None or right_img is None:
        print(f"Image(s) not found")
        return
    if left_img.ndim == 3:
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

    return left_img, right_img

def compute_initial_disparity(left_image, right_image, num_disp=64):
    """
    Compute an initial disparity map
    using OpenCV's modified SGM algorithm for better performance.

    :param left_image: Left image (RGB or grayscale)
    :param right_image: Right image (RGB or grayscale)
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

    :param image: Reference image (RGB or grayscale)
    :param n_segments: Number of segments for initial number of clusters

    :return: Segmentation image with labeled pixels
    """
    if image.ndim < 3:  # grayscale
        segments = slic(image, n_segments=n_segments, compactness=10, slic_zero=True,
                        channel_axis=None)
    else:
        segments = slic(image, n_segments=n_segments, compactness=10, slic_zero=True,
                        channel_axis=-1)
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

    # calculate sample threshold
    min_pixels = 20
    frac_of_avg_seg = 0.2
    total_pixels = disparity.size
    n_segments = len(np.unique(segments))
    if n_segments == 0:
        return refined_disparity
    avg_seg_size = total_pixels / n_segments
    thresh = max(min_pixels, int(avg_seg_size * frac_of_avg_seg))

    skipped_small = 0

    for segment_id in np.unique(segments):
        mask = (segments == segment_id) & ~occlusion_mask  # consider only non-occluded regions
        num_pixels = np.sum(mask)
        if num_pixels <= thresh:
            skipped_small += 1
            continue
        y, x = np.where(mask)
        disparities = disparity[mask]

        ### fit plane using RANSAC
        # ignore undefined-metric warnings in few segments
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            ransac = RANSACRegressor()
            ransac.fit(np.c_[x, y], disparities)

        # update disparity for this segment
        coef = ransac.estimator_.coef_
        intercept = ransac.estimator_.intercept_
        fitted_disparity = coef[0] * x + coef[1] * y + intercept
        refined_disparity[mask] = fitted_disparity

    processed = n_segments - skipped_small
    ratio = round(processed/n_segments * 100, ndigits=1)
    print(f"Processed {processed}/{n_segments} segments ({ratio}%)")
    # print(f"sample thresh={thresh}, skipped_small={skipped_small}")

    return refined_disparity

def handle_occlusion(refined_disparity, occlusion_mask, left_image, right_image, sgbm, offset):
    """
    Fill occluded regions using WLS filtering.

    :param refined_disparity: Refined disparity map
    :param occlusion_mask: Occlusion mask image
    :param left_image: Left image (RGB or grayscale)
    :param right_image: Right image (RGB or grayscale)
    :param sgbm: Stereo configuration
    :param offset: Left offset width

    :return: Occlusion filled disparity map
    """
    left_disp = refined_disparity.copy()
    filled_disp = refined_disparity.copy()

    # restore original disparity shape for WLS filtering
    left_disp = np.pad(left_disp, ((0, 0), (offset, 0)), mode='constant', constant_values=0)

    if left_image.ndim == 3:
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)
    else:
        left_gray = left_image
        right_gray = right_image

    # create WLS filter
    wls_sigma = 1.0
    wls_lamda = 8000.0
    left_matcher = sgbm
    right_matcher = ximg.createRightMatcher(left_matcher)
    right_disp = right_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    wls_filter = ximg.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(wls_lamda)
    wls_filter.setSigmaColor(wls_sigma)

    wls_disp = wls_filter.filter(left_disp, left_gray, disparity_map_right=right_disp)

    if np.isnan(wls_disp).any():
        print("NaN detected, replacing with 0")
        wls_disp = np.nan_to_num(wls_disp, nan=0)

    # fill occluded regions (in valid ROI)
    wls_disp_in_roi = crop_array(wls_disp, x0=offset)
    filled_disp[occlusion_mask] = wls_disp_in_roi[occlusion_mask]

    return filled_disp

def crop_array(arr, x0=0, x1=None, y0=0, y1=None):
    """
    Simple utility to crop a 2D array.
    """
    return arr[y0:y1, x0:x1]


def stereo(save_img:bool=False, visualize:bool=True):
    """
    Main function applying the proposed algorithm.

    :param save_img: Whether to save the output disparity map
    :param visualize: Whether to visualize the output disparity map
    :return: void
    """
    img_dir = "images"
    output_dir = "outputs"

    # config for demo images (1280x960 px)
    scene = "devon1"
    num_disp = 208  # must be divisible by 16
    n_segments = 4000

    left_path = os.path.join(img_dir, f'{scene}-left.ppm')
    right_path = os.path.join(img_dir, f'{scene}-right.ppm')

    start_time = time.time()

    print("Loading images....")
    left_img, right_img = load_images(left_path, right_path)

    print("\nComputing initial disparity map...")
    initial_disp, sgbm = compute_initial_disparity(left_img, right_img, num_disp=num_disp)

    # crop leftmost invalid disparity region (use num_disp offset)
    initial_disp_in_roi = crop_array(initial_disp, x0=num_disp)
    left_img_in_roi = crop_array(left_img, x0=num_disp)

    # occlusion detection and smoothing
    occlusions = initial_disp_in_roi < 0
    initial_disp_in_roi = cv2.medianBlur(initial_disp_in_roi, 5)

    print("\nApplying superpixel segmentation...")
    superpixels = generate_superpixels(left_img_in_roi, n_segments=n_segments)

    print("\nApplying RANSAC plane fitting....")
    refined_disp = fit_planes_to_superpixels(initial_disp_in_roi, superpixels, occlusions)

    print("\nHandling occlusion....")
    output_disp = handle_occlusion(refined_disp, occlusions, left_img, right_img,
                                                  sgbm, offset=num_disp)
    output_disp = cv2.medianBlur(output_disp, 5)

    end_time = time.time()
    runtime = end_time - start_time
    print("\nFinished.")
    print(f"Runtime: {runtime:.2f} seconds")

    if save_img:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{scene}_disparity.pfm")
        cv2.imwrite(save_path, output_disp)
        print(f"Saved to {save_path}")

    if visualize:
        plt.figure(figsize=(8, 4))
        plt.subplot(121);plt.axis("off")
        if left_img_in_roi.ndim < 3:  # grayscale
            plt.imshow(left_img_in_roi, cmap='gray'); plt.title(f"Reference Image")
        else:
            plt.imshow(left_img_in_roi); plt.title(f"Reference Image")
        plt.subplot(122);plt.axis("off")
        plt.imshow(output_disp, cmap='bone'); plt.title(f"Disparity Map")
        plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    stereo(save_img=False, visualize=True)

