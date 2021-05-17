import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#Load a descent image and the reference map
cameraFrame = cv.imread('LVS155156.jpg')
referenceMapFrame = cv.imread('jezero_6mperpix.tif')


#convert both to grayscale
cameraFrameGray = cv.cvtColor(cameraFrame, cv.COLOR_BGR2GRAY)
referenceMapFrameGray = cv.cvtColor(referenceMapFrame, cv.COLOR_BGR2GRAY)
# a copy of the reference map for display purposes
referenceMapFrameColor = referenceMapFrame.copy()
# use SIFT to compute keypoints and descriptors in both descent image and reference map
sift = cv.SIFT_create()
keyPointsCamera, cameraDescriptors = sift.detectAndCompute(cameraFrameGray, None)
keyPointsReferenceMap, referenceMapDescriptors = sift.detectAndCompute(referenceMapFrameGray, None)

#use Fast Library for Approximate Nearest Neighbors (FLANN) to match features in both images

FLANN_INDEX_KDTREE = 1

index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(cameraDescriptors, referenceMapDescriptors, k=2)

# store all the good matches as per Lowe's ratio test.
listOfGoodMatches = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        listOfGoodMatches.append(m)


MIN_MATCH_COUNT = 10

if len(listOfGoodMatches) > MIN_MATCH_COUNT:
    src_pts = np.float32([keyPointsCamera[m.queryIdx].pt for m in listOfGoodMatches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keyPointsReferenceMap[m.trainIdx].pt for m in listOfGoodMatches]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w, d = cameraFrame.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M).astype(np.int32)
    img2 = cv.polylines(referenceMapFrameColor, [dst], True, (255, 255, 255), 5)
    dst = dst.reshape((4, 1, 2))
    p0= dst[0]
    t0 = p0[0]
    x0 = t0[0]
    cv.circle(img2, dst[0][0], 50, (255, 0, 0), 15)
    cv.circle(img2, dst[1][0], 50, (0, 0, 255), 15)
    cv.circle(img2, dst[2][0], 50, (0, 255, 0), 15)
    cv.circle(img2, dst[3][0], 50, (255, 255, 0), 15)

    cv.circle(cameraFrame, (0, 0), 100, (255, 0, 0), 20)
    cv.circle(cameraFrame, (0, h-1), 100, (0, 0, 255), 20)
    cv.circle(cameraFrame, (w-1, h-1), 100, (0, 255, 0), 20)
    cv.circle(cameraFrame, (w-1, 0), 100, (255, 255, 0), 20)


    print("Found {} good matches".format(len(listOfGoodMatches)))
else:
    print("Not enough matches are found - {}/{}".format(len(listOfGoodMatches), MIN_MATCH_COUNT))
    matchesMask = None


draw_params = dict(matchColor=(0, 255, 0), # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask, # draw only inliers
                   flags=2)
img3 = cv.drawMatches(cameraFrame, keyPointsCamera, img2, keyPointsReferenceMap, listOfGoodMatches, None, **draw_params)
plt.imshow(img3, 'gray')
plt.show()
