import cv2
import numpy as np
from skimage import exposure
import os
import pandas as pd

from video_analysis_parameters import *


def set_thresholds(video):
    cv2.namedWindow("original")
    cv2.namedWindow("set_contrast")
    cv2.namedWindow("threshed")
    cv2.namedWindow("set_sizes")
    cv2.createTrackbar("frame", "original", 0, len(video) - 1,
                       dummy)  # creatres the trackbar/ 1st arg: trackbar name, 2) window name 3) statring val 4) max value 5) callbackfunction (function to which the new value is passed)
    cv2.createTrackbar("min", "set_contrast", video.min(), 255,
                       dummy)  # using min and max subtracted rescales the histogram (autocontrast like)
    cv2.createTrackbar("max", "set_contrast", video.max(), 255, dummy)
    cv2.createTrackbar("min_para", "set_sizes", 0, 100, dummy)
    cv2.createTrackbar("max_para", "set_sizes", 100, 100, dummy)
    while True:  # allow refreshing by always iteratingthrough the trackbar update
        frame_number = cv2.getTrackbarPos("frame", "original")
        sub_frame = video[frame_number]  # grab a single frame defined by the scroll bar to apply tresholding operation on this single frame
        # get trackbar values
        minval = cv2.getTrackbarPos("min", "set_contrast")
        maxval = cv2.getTrackbarPos("max", "set_contrast")
        min_para = cv2.getTrackbarPos("min_para", "set_sizes")
        max_para = cv2.getTrackbarPos("max_para", "set_sizes")
        # adjust contrast and apply treshold
        contrasted_frame = exposure.rescale_intensity(sub_frame, (minval, maxval))  # stretches the intensity hisotgram of the frame btw minval and maxval
        ret, thresh_frame = cv2.threshold(contrasted_frame, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # argument 2 and 3 are the min and max value but in case aof autotrehsold, not important. last argunent is the kind of trehsold. gives back a return value, to ignore
        # find the paramecia
        im2, contours, hierarchy = cv2.findContours(thresh_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # find the contours of bright element on the black bg. returns 1) new image - dont use 2) contours (list of arrays=contours, elenents in the array = points of the contour 3) hierarchy - dont use
        contour_areas = [cv2.contourArea(contour) for contour in contours]
        contours_in_range = [contour for contour, area in zip(contours, contour_areas) if min_para <= area <= max_para]  # gets the countour which areas are btw min and max  of para size
        contour_image = np.zeros(sub_frame.shape, dtype="uint8")  # creating black image
        cv2.drawContours(contour_image, contours_in_range, -1, 255,-1)  # draw the countours, -1 draws all contours , 255 stands for white. final argument is the thickness , -1 fills the contours
        for window, img in zip(["original", "set_contrast", "threshed", "set_sizes"],[sub_frame, contrasted_frame, thresh_frame, contour_image]):
            cv2.imshow(window, img)
        k = cv2.waitKey(1)
        if k == 13:
            break
    cv2.destroyAllWindows()
    return minval, maxval, min_para, max_para

if __name__ == "__main__" :
    start_video_folder = "J:\\Sylvain_Moser\\check\\trial1"
    start_video_name=os.listdir(start_video_folder)[0]
    print start_video_name
    start_video_path=os.path.join(start_video_folder,start_video_name)
    video = open_video(start_video_path)  # video is a 3d numpy array with all the frames as 2d np array
    bg = calculate_bg(video)  # use the function to calculate a background
    subtracted = bg_subtraction(video, bg)  # use the function to subtract the background
    rois = []
    conditions = []
    rois_df=pd.DataFrame()

    for i,c in zip(range(4),["up left","up right","bottom left","bottom right"]):
        roi = draw_rois(bg, "circle")  # use the function to allow you drawing the rois
        rois.append(roi)
        condition = c
        conditions.append(condition)
    #rois_df = pd.DataFrame()
    for roi, condition in zip(rois, conditions):
        cropped = circle_crop(subtracted, *roi)  # crops the image
        minval, maxval, minpara, maxpara = set_thresholds(cropped)  # call the function that allow to set the threshold on the cropped image and save the minval, maxval, minpara and maxpara variable
        (xmin, ymin), (xmax, ymax) = roi
        roi_info = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "min_contrast": minval,"max_contrast": maxval, "min_para_size": minpara, "max_para_size": maxpara, "condition": condition}
        rois_df = rois_df.append(roi_info, ignore_index=True)
    #rois_path = os.path.join(trials_folder, trial + ".csv")
    #rois_df.to_csv(rois_path, index=False)
    #trial_info = {"trial_name": trial, "rois_path": rois_path, "video_directory": trial_path}
    #trials_df = trials_df.append(trial_info, ignore_index=True)

    rois_paramecia_counts = []
    analyse_conditions = []
    for i, roi_serie in rois_df.iterrows():
        max_contrast = roi_serie["max_contrast"]
        min_contrast = roi_serie["min_contrast"]
        max_para_size = roi_serie["max_para_size"]
        min_para_size = roi_serie["min_para_size"]
        top_left = (roi_serie["xmin"], roi_serie["ymin"])
        bottom_right = (roi_serie["xmax"], roi_serie["ymax"])
        #conditions.append(roi_serie["condition"])
        cropped_video = circle_crop(subtracted, top_left, bottom_right)
        paramecia_counts = []
        for frame in cropped_video:
            contrasted_frame = exposure.rescale_intensity(frame, (min_contrast, max_contrast))
            ret, thresh_frame = cv2.threshold(contrasted_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            im2, contours, hierarchy = cv2.findContours(thresh_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find the contours of bright element on the black bg. returns 1) new image - dont use 2) contours (list of arrays=contours, elenents in the array = points of the contour 3) hierarchy - dont use
            contour_areas = [cv2.contourArea(contour) for contour in contours]
            contours_in_range = [contour for contour, area in zip(contours, contour_areas) if min_para_size <= area <= max_para_size]
            paramecia_number = len(contours_in_range)
            paramecia_counts.append(paramecia_number)
        paramecia_counts = pd.Series(paramecia_counts)
        rois_paramecia_counts.append(paramecia_counts)
    rois_paramecia_counts = pd.DataFrame(rois_paramecia_counts).T
    rois_paramecia_counts.columns = conditions
    mean_paramecia_counts=rois_paramecia_counts.mean(axis=0)
    print mean_paramecia_counts




