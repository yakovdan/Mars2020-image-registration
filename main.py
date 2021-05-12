import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

cameraFrame = cv.imread('LVS155156.jpg')
#atlasFrame = cv.imread('atlasframe.png')
atlasFrame = cv.imread('jezero_small.jpg')
cameraFrameGray = cv.cvtColor(cameraFrame,cv.COLOR_BGR2GRAY)
atlasFrameGray = cv.cvtColor(atlasFrame,cv.COLOR_BGR2GRAY)
cameraFrameWithKeyPoints = cameraFrameGray.copy()
cameraFrameWithRichKeyPoints = cameraFrameGray.copy()
sift = cv.SIFT_create()
keyPointsCamera = sift.detect(cameraFrameGray, None)
keyPointsAtlas = sift.detect(atlasFrameGray, None)
keyPointsCamera, cameraDescriptors = sift.detectAndCompute(cameraFrameGray, None)
keyPointsAtlas, atlasDescriptors = sift.detectAndCompute(atlasFrameGray, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(cameraDescriptors, atlasDescriptors,k=2)

# store all the good matches as per Lowe's ratio test.
listOfGoodMatches = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        listOfGoodMatches.append(m)

cameraFrameWithKeyPoints = cv.drawKeypoints(cameraFrameGray, keyPointsCamera, cameraFrameWithKeyPoints, color=(255, 0, 0))
cameraFrameWithRichKeyPoints = cv.drawKeypoints(cameraFrameGray, keyPointsCamera, cameraFrameWithRichKeyPoints, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('sift_keypoints.jpg', cameraFrameWithKeyPoints)
cv.imwrite('sift_rich_keypoints.jpg', cameraFrameWithRichKeyPoints)

MIN_MATCH_COUNT = 10

if len(listOfGoodMatches) > MIN_MATCH_COUNT:
    src_pts = np.float32([keyPointsCamera[m.queryIdx].pt for m in listOfGoodMatches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keyPointsAtlas[m.trainIdx].pt for m in listOfGoodMatches]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w, d = cameraFrame.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0] ]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)
    img2 = cv.polylines(atlasFrameGray, [np.int32(dst)],True,255,3, cv.LINE_AA)
    print("Found {} good matches".format(len(listOfGoodMatches)))
else:
    print("Not enough matches are found - {}/{}".format(len(listOfGoodMatches), MIN_MATCH_COUNT))
    matchesMask = None


draw_params = dict(matchColor=(0, 255, 0), # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask, # draw only inliers
                   flags=2)
img3 = cv.drawMatches(cameraFrame, keyPointsCamera, atlasFrame, keyPointsAtlas, listOfGoodMatches, None, **draw_params)
plt.imshow(img3, 'gray')
plt.show()
