import numpy
import cv2
import os
from datetime import datetime

LVS_IMAGES_PATH = "./LVSImages"

# given a path, parse filenames and
# return the time shift from first frame, in seconds

def parse_timestamps(path):
    file_list = os.listdir(LVS_IMAGES_PATH)
    times_list = [x[4:12] for x in file_list]
    first_datetime_object = datetime.strptime(times_list[0], '%H_%M_%S')
    time_delta_list = [(datetime.strptime(timeStr, '%H_%M_%S') - first_datetime_object).total_seconds() for timeStr in times_list]
    return time_delta_list

# given a path, read all images

def read_all_images(path):
    file_list = os.listdir(LVS_IMAGES_PATH)
    return [cv2.imread(path+"/"+name) for name in file_list]

def main_func():
    time_shifts = parse_timestamps(LVS_IMAGES_PATH)
    images = read_all_images(LVS_IMAGES_PATH)

    out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, (800,800))
    for i in range(len(images)):
        out.write(images[i])
    out.release()



if __name__=="__main__":
    main_func()