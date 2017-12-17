# -*- coding: utf-8 -*-
"""
Created on Tue Jun 06 01:03:34 2017

@author: smoser
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 31 00:18:23 2017

@author: smoser
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:41:41 2017

@author: smoser
"""

import cv2
import numpy as np
from skimage import exposure
import os
import pandas as pd


# enter = 13
# escape = 27

def open_video(filepath):
    cap = cv2.VideoCapture(filepath)  # create the video capture object
    frames = []  # create empty list to store the frames
    while True:
        ret, frame = cap.read()  # grab the first frame of the video. ret is boolean (true if there is a frame, false is there is none)
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert the frame to grayscale
            frames.append(frame)
        else:
            break
    frames = np.array(frames)
    return frames  # return all the frames of the videos


def play_video(filepath):
    cap = cv2.VideoCapture(filepath)
    cv2.namedWindow("test_video")  # takes the argument name of the window. creates the window
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("test_video", frame)  # displays the first frame
            cv2.waitKey(1)  # computer wait until ypu press a key. arguments time in ms that it will wait max
        else:
            cv2.destroyAllWindows()
            break


def scroll_video(video):
    cv2.namedWindow("scroll_video")
    cv2.createTrackbar("frame", "scroll_video", 0, len(video) - 1, dummy)
    while True:
        frame_number = cv2.getTrackbarPos("frame", "scroll_video")
        frame = video[frame_number]
        cv2.imshow("scroll_video", frame)
        k = cv2.waitKey(1)
        if k == 13:
            break
    cv2.destroyAllWindows()


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
        sub_frame = video[
            frame_number]  # grab a single frame defined by the scroll bar to apply tresholding operation on this single frame
        # get trackbar values
        minval = cv2.getTrackbarPos("min", "set_contrast")
        maxval = cv2.getTrackbarPos("max", "set_contrast")
        min_para = cv2.getTrackbarPos("min_para", "set_sizes")
        max_para = cv2.getTrackbarPos("max_para", "set_sizes")
        # adjust contrast and apply treshold
        contrasted_frame = exposure.rescale_intensity(sub_frame, (
        minval, maxval))  # stretches the intensity hisotgram of the frame btw minval and maxval
        ret, thresh_frame = cv2.threshold(contrasted_frame, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # argument 2 and 3 are the min and max value but in case aof autotrehsold, not important. last argunent is the kind of trehsold. gives back a return value, to ignore
        # find the paramecia
        im2, contours, hierarchy = cv2.findContours(thresh_frame, cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_SIMPLE)  # find the contours of bright element on the black bg. returns 1) new image - dont use 2) contours (list of arrays=contours, elenents in the array = points of the contour 3) hierarchy - dont use
        contour_areas = [cv2.contourArea(contour) for contour in contours]
        contours_in_range = [contour for contour, area in zip(contours, contour_areas) if
                             min_para <= area <= max_para]  # gets the countour which areas are btw min and max  of para size
        contour_image = np.zeros(sub_frame.shape, dtype="uint8")  # creating black image
        cv2.drawContours(contour_image, contours_in_range, -1, 255,
                         -1)  # draw the countours, -1 draws all contours , 255 stands for white. final argument is the thickness , -1 fills the contours
        for window, img in zip(["original", "set_contrast", "threshed", "set_sizes"],
                               [sub_frame, contrasted_frame, thresh_frame, contour_image]):
            cv2.imshow(window, img)
        k = cv2.waitKey(1)
        if k == 13:
            break
    cv2.destroyAllWindows()
    return minval, maxval, min_para, max_para


def dummy(trackbar_val):
    return


def mouse_callback(event, x, y, flags,
                   params):  # event is what you do (left,right click, x and y are position of mouse in the window,)
    img = params["image"].copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # when you click with the left button it selects the first corner of the roi. sets p1 to that corrdinate
        params["p1"] = (x, y)
        params["p2"] = None
        params["selection"] = False
    elif event == cv2.EVENT_LBUTTONUP:  # when you release the left button it selects the last corner of the roi, sets p2 to that coordinate  and says that the selection is made with selection=True
        params["p2"] = (x, y)
        params["selection"] = True
    elif event == cv2.EVENT_RBUTTONUP:  # right click resets the roi
        params["p1"] = None
        params["p2"] = None
        params["selection"] = False
    elif not params[
        "selection"]:  # if selection if false (when left bluckon clicked but not release, so while dragging) p2 is updated at each mouse movement to allow the dragging
        params["p2"] = (x, y)
        # cv2.circle(img,(x,y),5,255,-1)#parameters are image, position, radius, color, thickness
    if params["p1"] is not None and params[
        "p2"] is not None:  # if p1 and p2 is defined, while dragging of after, it draws the rectangle shape
        if params["shape"] == "rectangle":
            cv2.rectangle(img, params["p1"], params["p2"], 255, 1)
        elif params["shape"] == "circle":  # draw the circular roi
            p1 = np.array(params["p1"])
            p2 = np.array(params["p2"])
            v = p2 - p1  # the vector between p1 and p2
            d = np.linalg.norm(v)  # linalg.norm calculates the lenght of the vector v, which is the diameter
            r = 0.5 * d
            c = p1 + (
            v * r / d)  # to calculatr the center of the cricle passing by the 2 points ypu can just average them
            c = int(c[0]), int(c[1])  # making c a tuple with the 2 corrdinates, they were array before
            cv2.circle(img, c, int(r), 255, 1)
    cv2.imshow("Draw_ROIS", img)


def draw_rois(image, shape):
    my_params = {"p1": None, "p2": None, "image": image, "selection": False, "shape": shape}
    cv2.namedWindow("Draw_ROIS")
    cv2.setMouseCallback("Draw_ROIS", mouse_callback,
                         my_params)  # takes the name of the window to interact with and the call bakc function. my_params will be passed to the callback function and passed in params. as the callback function only take one paramter, you have to put them in a dictionary if zou want to pass several

    cv2.imshow("Draw_ROIS", image)  # if not i need to quick a first time to see sth
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
    p1, p2 = my_params["p1"], my_params["p2"]
    xmin = min(p1[0], p2[0])  # looks in the x coordinate of both corner for the smalles one, which is then xmin (because we cant say if p1 or p2 is more up in the image)
    xmax = max(p1[0], p2[0])
    ymin = min(p1[1], p2[1])
    ymax = max(p1[1], p2[1])
    top_left = (xmin, ymin)  # the top left is the min x coordinate and the min y coordinate
    bottom_right = (xmax, ymax)  # the bottom right is the point wich the max x and y coordinate
    return top_left, bottom_right  # return the coordinate of the2 cornes


def calculate_bg(video):
    background = video.mean(axis=0)
    background = background.astype("uint8")
    return background


def bg_subtraction(video, background):
    background = background.astype(
        "i4")  # transform the numpy array with the mena (which is float) into unsigned 8 bit array
    subtracted = video.astype(
        "i4") - background  # subtract the background to the video. needs signed 64 bits array because if not it gives overflow when bg ligher than the video
    subtracted = np.clip(subtracted, 0,
                         255)  # set the limit values to 0 and 255, everything less than 0 =0 and everything >255 =255. prevent overflow for negative values
    subtracted = subtracted.astype("uint8")
    return subtracted


def crop_video(video, p1, p2):
    cropped = video[:, p1[1]:p2[1] + 1, p1[0]:p2[
                                                  0] + 1]  # first index select all frames of the video, then select the square defined by the [y coordinate of top left(==xmin): y coordinate of bottom right(==xmax). because in opencv the point 0,0 is the upper left corner
    return cropped


def circle_crop(video, p1, p2):
    mask_shape = video.shape[1], video.shape[
        2]  # mask shape takes the width and height of the video, ommiting dimension 0 which is the frame number
    mask = np.zeros(mask_shape,
                    dtype="uint8")  # we get a mask which is a single image with the height and width of the video
    p1 = np.array(p1)
    p2 = np.array(p2)
    v = p2 - p1  # the vector between p1 and p2
    d = np.linalg.norm(v)  # linalg.norm calculates the lenght of the vector v which is the diameter
    r = 0.5 * d
    c = (p1 + p2) / 2  # to calculatr the center of the cricle passing by the 2 points ypu can just average them
    c = int(c[0]), int(c[1])  # making c a tuple with the 2 corrdinates, they were array before
    cv2.circle(mask, c, int(r), 1, -1)
    mask = mask.astype('bool')  # making bollean: all the 0 become false and all the 1 become true

    masked_video = video.copy()  # dont change the bg acciddentaly
    mask_points = np.where(~mask)
    masked_video[:, mask_points[0], mask_points[
        1]] = 0  # np.where take all the places where True so where =1. the ~take the contrary, so all the places where false and set them to 0 in the bg image
    return masked_video


def show_image(image):
    cv2.namedWindow('image')
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# when calling crop video: cropped=crop_video(subtracted, *rois) * pass the tupple as separate arguments

if __name__ == "__main__":
    # SETTING UP
    experiment_path = 'J:\Sylvain_Moser\BPC_MF_OT_ablated'
    output_path = "C:\\data"
    experiment_name = os.path.basename(experiment_path)
    experiment_output = os.path.join(output_path, experiment_name)
    if not os.path.exists(experiment_output):
        os.makedirs(experiment_output)
    # make experiment info data frame
    trials_csv = os.path.join(experiment_output, 'trials.csv')
    trials_csv_cols = ['trial_name', 'rois_path', 'video_directory']
    if not os.path.exists(trials_csv):
        trials_df = pd.DataFrame(columns=trials_csv_cols)
    else:
        trials_df = pd.read_csv(trials_csv)
    trials_folder = os.path.join(experiment_output, "trials")
    if not os.path.exists(trials_folder):
        os.makedirs(trials_folder)
    trials = os.listdir(experiment_path)  # gives all the folders in the experiment path
    trials.sort()
    for trial in trials:
        # print trial
        trial_path = os.path.join(experiment_path, trial)  # creates the path to each of the trials folder
        if os.path.isdir(trial_path) and trial not in trials_df[
            "trial_name"].values:  # if the trial path goes to a directery
            avi_files = os.listdir(trial_path)  # it appends all the video in this directory to the list of avi files
            avi_files = [avi for avi in avi_files if
                         avi[-3:] == "avi"]  # gets only the file with avi extension in the avi_file list
            avi_files.sort()
            # print avi_files
            first_timepoint = avi_files[0]  # defines which is the video of first time point
            filepath = os.path.join(trial_path, first_timepoint)  # gives the filepath to the video of first time point
            video = open_video(filepath)  # video is a 3d numpy array with all the frames as 2d np array
            bg = calculate_bg(video)  # use the function to calculate a background
            subtracted = bg_subtraction(video, bg)  # use the function to subtract the background
            rois = []
            conditions = []
            for i in range(4):
                roi = draw_rois(bg, "circle")  # use the function to allow you drawing the rois
                rois.append(roi)
                print(trial)
                condition = raw_input("Condition: ")
                conditions.append(condition)
            rois_df = pd.DataFrame()
            for roi, condition in zip(rois, conditions):
                cropped = circle_crop(subtracted, *roi)  # crops the image
                minval, maxval, minpara, maxpara = set_thresholds(
                    cropped)  # call the function that allow to set the threshold on the cropped image and save the minval, maxval, minpara and maxpara variable
                (xmin, ymin), (xmax, ymax) = roi
                roi_info = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "min_contrast": minval,
                            "max_contrast": maxval, "min_para_size": minpara, "max_para_size": maxpara,
                            "condition": condition}
                rois_df = rois_df.append(roi_info, ignore_index=True)
            rois_path = os.path.join(trials_folder, trial + ".csv")
            rois_df.to_csv(rois_path, index=False)
            trial_info = {"trial_name": trial, "rois_path": rois_path, "video_directory": trial_path}
            trials_df = trials_df.append(trial_info, ignore_index=True)
            trials_df.to_csv(trials_csv, index=False)



            # doing the analysis






            # ==============================================================================
            #     filepath="C:\\data\\test_video\\17-07-12.258.avi"
            #     video=open_video(filepath)#video is a 3d numpy array with all the frames as 2d np array
            #
            #     bg=calculate_bg(video)
            #     subtracted=bg_subtraction(video,bg)
            #     rois=draw_rois(bg,"circle")
            # ==============================================================================














            # ==============================================================================
            #     winname="scroll"
            #     cv2.namedWindow(winname,0)
            #     #cv2.imshow(winname,np.hstack((subtracted[0], subtracted[0], subtracted[0])))
            #     cv2.createTrackbar("frame",winname, 0, len(subtracted)-1, dummy) #creatres the trackbar/ 1st arg: trackbar name, 2) window name 3) statring val 4) max value 5) callbackfunction (function to which the new value is passed)
            #     cv2.createTrackbar("min",winname, subtracted.min() ,255, dummy)#using min and max subtracted rescales the histogram (autocontrast like)
            #     cv2.createTrackbar("max",winname, subtracted.max() ,255, dummy)
            #     cv2.createTrackbar("min_para",winname,0,100,dummy)
            #     cv2.createTrackbar("max_para",winname,100,100,dummy)
            #     while True:#allow refreshing by always iteratingthrough the trackbar update
            #         frame_number=cv2.getTrackbarPos("frame",winname)
            #         sub_frame=subtracted[frame_number]#grab a single frame defined by the scroll bar to apply tresholding operation on this single frame
            #         #get trackbar values
            #         minval=cv2.getTrackbarPos("min",winname)
            #         maxval=cv2.getTrackbarPos("max",winname)
            #         min_para=cv2.getTrackbarPos("min_para",winname)
            #         max_para=cv2.getTrackbarPos("max_para",winname)
            #         #adjust contrast and apply treshold
            #         contrasted_frame=exposure.rescale_intensity(sub_frame,(minval,maxval))#stretches the intensity hisotgram of the frame btw minval and maxval
            #         ret, thresh_frame=cv2.threshold(contrasted_frame,0,255,cv2.THRESH_OTSU)#argument 2 and 3 are the min and max value but in case aof autotrehsold, not important. last argunent is the kind of trehsold. gives back a return value, to ignore
            #         #find the paramecia
            #         im2, contours, hierarchy=cv2.findContours(thresh_frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#find the contours of bright element on the black bg. returns 1) new image - dont use 2) contours (list of arrays=contours, elenents in the array = points of the contour 3) hierarchy - dont use
            #         contour_areas=[cv2.contourArea(contour) for contour in contours]
            #         contours_in_range=[contour for contour, area in zip(contours,contour_areas) if min_para <=area <=max_para] #gets the countour which areas are btw min and max  of para size
            #         contour_image=np.zeros(sub_frame.shape,dtype="uint8")#creating black image
            #         cv2.drawContours(contour_image,contours_in_range,-1,255,-1)#draw the countours, -1 draws all contours , 255 stands for white. final argument is the thickness , -1 fills the contours
            #         show_image=np.hstack((contrasted_frame,thresh_frame,contour_image))#concatenate horizontally the frames(arrays)
            #         cv2.imshow("scroll",show_image)
            #         k=cv2.waitKey(1)
            #         if k==13:
            #             break
            #     cv2.destroyAllWindows()
            # ==============================================================================

            # finding paramecia



            # cv2.imshow("subtracted",subtracted[0])
            # ==============================================================================
            #     cv2.imshow("background", background)
            #   cv2.waitKey(0)#computer wait until ypu press a key. arguments time in ms that it will wait max
            # ==============================================================================
    cv2.destroyAllWindows()



