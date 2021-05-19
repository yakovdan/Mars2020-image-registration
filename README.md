# Mars2020-image-registration
In this project, I attempt to reconstruct part of the mars2020 entry, descent and landing (EDL) trajectory by registering mars2020 lander vision system images onto a 
a reference map the landing area on Mars - Jezero creater. 

This is a WIP. At present, most of the LVS images are successfully registered and the result is output to a video file

In this project, I use SIFT to compute features and their descriptors for the reference map as well as for 
each of the LVS images. I then match them using a KD-tree to compute aproximate nearest neighbours. 
Having registered the images, I find a homography matrix between each pair using RANSAC. 
Using the homography matrix, I project each LVS image that has a homography matrix onto the reference map.


