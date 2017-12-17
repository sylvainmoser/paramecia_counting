# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 14:40:59 2017

@author: smoser
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 09 16:19:42 2017

@author: smoser
"""
from video_analysis_parameters import *
import os, sys
import pandas as pd




if __name__ == "__main__" :
    experiment_path = "C:\\data\\07122017"
    trials_csv = os.path.join(experiment_path, 'trials.csv')
    output_folder=os.path.join(experiment_path,)
    if not os.path.exists(experiment_path):
        print "Experiment does not exist!"
        sys.exit()
    else:
        trials_df = pd.read_csv(trials_csv)
    para_counts_folder=os.path.join(experiment_path,"paramecia_counts")
    if not os.path.exists(para_counts_folder):
        os.makedirs(para_counts_folder)# make top folder for saving paramecia counts
    for idx,trial in trials_df.iterrows():
        # create trial folder for saving rois_paramecia_counts
        # add path to this folder to trials_csv DataFrame
        # re-save trials_csv DataFrame
        trial_name=trial["trial_name"]
        print trial_name
        paramecia_count_trial_folder=os.path.join(para_counts_folder,trial_name)
        if not os.path.exists(paramecia_count_trial_folder):
            os.makedirs(paramecia_count_trial_folder)
            rois_df=pd.read_csv(trial["rois_path"])
            video_files = os.listdir(trial["video_directory"])
            video_paths = [os.path.join(trial["video_directory"], f) for f in video_files]
            for video_path in video_paths:
                if video_path.endswith('.avi'):
                 # create filepath for saving paramecia counts
                # only run code below here if that file does not exist
                    video=open_video(video_path)
                    background=calculate_bg(video)
                    bg_sub_video=bg_subtraction(video,background)
                    rois_paramecia_counts=[]
                    video_name, ext=os.path.splitext(os.path.basename(video_path))
                    conditions=[]
                    for i,roi_serie in rois_df.iterrows():
                        max_contrast=roi_serie["max_contrast"]
                        min_contrast=roi_serie["min_contrast"]
                        max_para_size=roi_serie["max_para_size"]
                        min_para_size=roi_serie["min_para_size"]
                        top_left=(roi_serie["xmin"],roi_serie["ymin"])
                        bottom_right=(roi_serie["xmax"],roi_serie["ymax"])
                        conditions.append(roi_serie["condition"])
                        cropped_video=circle_crop(bg_sub_video,top_left,bottom_right)
                        paramecia_counts=[]
                        for frame in cropped_video:
                            contrasted_frame=exposure.rescale_intensity(frame,(min_contrast,max_contrast))
                            ret, thresh_frame=cv2.threshold(contrasted_frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                            im2, contours, hierarchy=cv2.findContours(thresh_frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#find the contours of bright element on the black bg. returns 1) new image - dont use 2) contours (list of arrays=contours, elenents in the array = points of the contour 3) hierarchy - dont use
                            contour_areas=[cv2.contourArea(contour) for contour in contours]
                            contours_in_range=[contour for contour, area in zip(contours,contour_areas) if min_para_size <=area <=max_para_size]
                            paramecia_number=len(contours_in_range)
                            paramecia_counts.append(paramecia_number)
                        paramecia_counts=pd.Series(paramecia_counts)
                        rois_paramecia_counts.append(paramecia_counts)
                    rois_paramecia_counts=pd.DataFrame(rois_paramecia_counts).T
                    rois_paramecia_counts.columns=conditions
                    rois_paramecia_counts_filepath=os.path.join(paramecia_count_trial_folder,video_name+".csv")
                    rois_paramecia_counts.to_csv(rois_paramecia_counts_filepath,index=False)
                    print("Done "+trial_name+"/"+video_name)
