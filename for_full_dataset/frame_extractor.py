
# coding: utf-8

# In[2]:


import os
import cv2


# In[37]:


sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
segment_loc = "/scratch/user/pbhatt/IEMOCAP_vid_clips/"
frames_loc = "/scratch/user/pbhatt/IEMOCAP_vid_frames/"
emotion_classes = ['ang', 'hap', 'neu']


# In[48]:
os.system("mkdir " + "/scratch/user/pbhatt/IEMOCAP_vid_frames")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
for sess in sessions:
    
    # make directory for each session
    os.system('mkdir ' + frames_loc + sess)
    for emo in emotion_classes:
        
        if os.path.isdir(segment_loc + sess + '/' + emo):
            video_files = [x for x in os.listdir(segment_loc + sess + '/' + emo) if x[-3:] == 'avi']
#             video_files = [x.split('_')[1][:-2] for x in video_files]
            # make directories for each emotion class inside all sessions
            out_loc = frames_loc + sess + '/' + emo
            os.system('mkdir ' + out_loc)
            
            for vid in video_files:
                if vid.split('_')[1][:-2] == 'script':
                    video = cv2.VideoCapture(segment_loc + sess + '/' + emo + '/' + vid)
                    print(vid, video.isOpened(),)
                    framerate = video.get(5)

                    while (video.isOpened()):
                        frameId = video.get(1)
                        success,image = video.read()

                        if(success != True):
                            break

                        # crop and resize the image
                        image = image[120:360, 50:290]
                        image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #                         imgplot = plt.imshow(image)
    #                         plt.show()

                        # save the frame
                        filename = out_loc + '/' + vid[:-4] + '_' + str(int(frameId)) + '.jpg'
                        cv2.imwrite(filename,image)
                    video.release()
                    print('-->done')
        else:
            print("Empty")

