
# coding: utf-8

# In[1]:


import os
import cv2


# In[21]:


dataset_dir = '/scratch/datasets/IEMOCAP/'
sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
vid_loc = '/dialog/avi/DivX/'

datafile_loc = '/dialog/EmoEvaluation/'
segment_loc = "/scratch/user/pbhatt/IEMOCAP_vid_clips/"


# In[46]:

os.system('mkdir ' + '/scratch/users/pbhatt/IEMOCAP_vid_clips')

# For each session go though each video and segment them into clips based on the time given in text filesf
for sess in sessions:
    # extract names of all videos
    vid_full_names = [x for x in os.listdir(dataset_dir + sess + vid_loc) if x[-3:] == 'avi']
    all_names = [x[:-4] for x in vid_full_names]
    print "------------------------------------------------"
    print sess
#     print all_names
    
    # location to save the segmented video clips
    os.system('mkdir ' + segment_loc + sess)
    
    for name in all_names:
        vid_name = name + '.avi'
        txt_file_name = name + '.txt'
        
        # location of video file and text file
        vid_file = dataset_dir + sess + vid_loc + vid_name
        txt_file = dataset_dir + sess + datafile_loc + txt_file_name
        out_loc = segment_loc + sess + '/' # + respective_emotion_folder
        
        print "\nSegmentating ", vid_name
        
        with open(txt_file) as f:
            data = f.readlines()
        data = iter(data)
        
        # Extracting information from the text file and segmentation
        try:
            for line in data:
                if line != '\n':
                    continue;
                line = next(data)
                start_time = line.split()[0][1:]
                end_time = line.split()[2][:-1]
                segment_name = line.split()[3]
                emotion = line.split()[4]
                if emotion != 'xxx':
                    os.system('mkdir ' + out_loc + emotion)
#                     print line
                    cmd = 'ffmpeg -i ' + vid_file + ' -ss ' + start_time + ' -acodec copy -to '+ end_time + ' ' + out_loc + emotion + '/' + segment_name + '.avi'
#                     print cmd
                    print os.system(cmd),
        except(StopIteration):
            None
