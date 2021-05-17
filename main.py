import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

LVS_IMAGES_PATH = "./LVSImages"
FLANN_INDEX_KDTREE = 1
MIN_MATCH_COUNT = 10


def read_all_images(path):
    file_list = os.listdir(path)
    return [cv2.imread(path+"/"+name) for name in file_list]


def build_match_images(reference_map_frame, images):
    # prepare reference map images
    output_list_of_images = []
    reference_map_frame_gray = cv2.cvtColor(reference_map_frame, cv2.COLOR_BGR2GRAY)

    # create SIFT Object and compute keypoints and descriptors once for the reference map image.
    sift = cv2.SIFT_create()
    key_points_reference_map, reference_map_descriptors = sift.detectAndCompute(reference_map_frame_gray, None)

    for camera_frame in images:
        # prepare next camera image
        camera_frame_gray = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2GRAY)
        reference_map_frame_color = referenceMapFrame.copy()

        # use SIFT to compute keypoints and descriptors in both descent image
        key_points_camera, camera_descriptors = sift.detectAndCompute(camera_frame_gray, None)

        # use Fast Library for Approximate Nearest Neighbors (FLANN) to match features in both images
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(camera_descriptors, reference_map_descriptors, k=2)

        # store all the good matches as per Lowe's ratio test.
        good_matches_list = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good_matches_list.append(m)

        if len(good_matches_list) > MIN_MATCH_COUNT:
            src_pts = np.float32([key_points_camera[m.queryIdx].pt for m in good_matches_list]).reshape(-1, 1, 2)
            dst_pts = np.float32([key_points_reference_map[m.trainIdx].pt for m in good_matches_list]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()
            h, w, d = camera_frame.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M).astype(np.int32)
            cv2.polylines(reference_map_frame_color, [dst], True, (255, 255, 255), 5)
            dst = dst.reshape((4, 1, 2))

            cv2.circle(reference_map_frame_color, dst[0][0], 50, (255, 0, 0), 15)
            cv2.circle(reference_map_frame_color, dst[1][0], 50, (0, 0, 255), 15)
            cv2.circle(reference_map_frame_color, dst[2][0], 50, (0, 255, 0), 15)
            cv2.circle(reference_map_frame_color, dst[3][0], 50, (255, 255, 0), 15)

            cv2.circle(camera_frame, (0, 0), 100, (255, 0, 0), 20)
            cv2.circle(camera_frame, (0, h-1), 100, (0, 0, 255), 20)
            cv2.circle(camera_frame, (w-1, h-1), 100, (0, 255, 0), 20)
            cv2.circle(camera_frame, (w-1, 0), 100, (255, 255, 0), 20)

            print("Found {} good matches".format(len(good_matches_list)))
        else:
            print("Not enough matches are found - {}/{}".format(len(good_matches_list), MIN_MATCH_COUNT))
            matches_mask = None

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matches_mask,  # draw only inliers
                       flags=2)
        img3 = cv2.drawMatches(camera_frame, key_points_camera, reference_map_frame_color,
                              key_points_reference_map, good_matches_list, None, **draw_params)
        corrected_color_img = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        output_list_of_images.append(corrected_color_img)

    return output_list_of_images


if __name__ == "__main__":
    # Load a descent image and the reference map
    images = read_all_images(LVS_IMAGES_PATH)
    #cameraFrame = cv2.imread('LVS155156.jpg')
    referenceMapFrame = cv2.imread('jezero_6mperpix.tif')
    output_mathched_images = build_match_images(referenceMapFrame,images)
    print(len(output_mathched_images))
    print(output_mathched_images[0].shape)
    out = cv2.VideoWriter('registered_images.avi', cv2.VideoWriter_fourcc(*'mp4v'), 5, (6122, 5040), True)
    for i in range(len(output_mathched_images)):
        out.write(output_mathched_images[i])
        #f = open('frame_'+str(i)+'bin', 'wb')
        #f.write(output_mathched_images[i].tobytes())
        #f.close()
    out.release()
