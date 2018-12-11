
# coding: utf-8

# In[109]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os, subprocess
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight


# In[110]:


sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
frames_loc = "/Users/sharingan/Documents/IEMOCAP_vid_frames/"
emotion_classes = ['ang', 'hap', 'neu'] 


# In[111]:


def CNN_model():
    model = Sequential()
    # define CNN model
    model.add(Conv2D(32, (3, 3), activation = 'relu' ,input_shape = (50,50,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(len(emotion_classes), activation = 'softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model


# In[112]:


# generator returns an iterator whose each iteration returns a tuple with following two parts
# batch of images of target_size size of selected color_mode ---- in our case 32 images of size 50x50 with 1 channel(grayscale)
# ground truth in term of one hot encoding ----- in our case 32 one hot encodings

def generate_train(directory):
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(directory,
                                                        target_size=(50, 50),
                                                        batch_size=32,
                                                        class_mode='categorical',
                                                        color_mode = 'grayscale')
    return train_generator

def generate_test(directory):
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(directory,
                                                            target_size=(50, 50),
                                                            batch_size=32,
                                                            class_mode='categorical',
                                                            color_mode='grayscale')
    return test_generator


# In[113]:


kf = KFold(n_splits=len(sessions))
kf = kf.split(sessions)

# Calculating class weights
for train, test in kf:
    y_train = []
    for t in train:
        train_generator = generate_train(frames_loc + sessions[t])
        y_train.extend(train_generator.classes)
    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print class_weight_dict


# In[114]:


# # Code to print the representation of each class in each session
# kf = KFold(n_splits=len(sessions))
# kf = kf.split(sessions)
# for train, test in kf:
#     y_test = []
#     for t in test:
#         test_generator = generate_test(frames_loc + sessions[t])
#         y_test.extend(test_generator.classes)
#     print y_test.count(0),",",
#     print y_test.count(1),",",
#     print y_test.count(2)
# print test_generator.class_indices


# In[116]:


print test_generator.class_indices.keys()
print test_generator.class_indices


# In[117]:


# create training and testing set

kf = KFold(n_splits=len(sessions))
kf = kf.split(sessions)

# Each iteration represents one fold
for train, test in kf:
    print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    # New model for every fold
    model = CNN_model()
    
    # Calculating class weights
    y_train = []
    for t in train:
        train_generator = generate_train(frames_loc + sessions[t])
        y_train.extend(train_generator.classes)
    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
    class_weights = dict(enumerate(class_weights))
    
    # Validation data
    for t in test:
        test_generator = generate_test(frames_loc + sessions[t])
    
    # Fit training data
    for t in train:
        train_generator = generate_train(frames_loc + sessions[t])
        
        model.fit_generator(
            train_generator,
            steps_per_epoch=np.math.ceil(float(train_generator.samples)/float(train_generator.batch_size)),
            class_weight = class_weights,
            epochs=5,
#             validation_data=test_generator,
#             validation_steps=np.math.ceil(float(test_generator.samples)/float(test_generator.batch_size))
        )
        
#     class_labels = list(test_generator.class_indices.keys())
    predictions = model.predict_generator(test_generator, np.math.ceil(float(test_generator.samples)/float(test_generator.batch_size)))
    predicted_classes = np.argmax(predictions, axis=1)
    
    # results
    report = metrics.classification_report(test_generator.classes, predicted_classes, target_names=emotion_classes)
    
    # confusion matrix
    confusion_mat = metrics.confusion_matrix(test_generator.classes, predicted_classes)
    
    print report  
    print emotion_classes
    print confusion_mat
    
    with open("results.txt", "a") as f:
        print >> f, report
        print >> f, emotion_classes
        print >> f, confusion_mat

#     # making directories and storing training and testing samples
#     os.system("rm -rf train")
#     os.system("rm -rf test")
#     os.system("mkdir train")
#     os.system("mkdir test")
    
#     print "Creating TRAINING SET"
#     dest = " ./train/"
#     for dr in train:
#         src = frames_loc + selected_scripts[dr] + '/'
#         print os.system("rsync -a " + src + dest)
    
#     print "Creating TESTING SET"
#     dest = " ./test/"
#     for dr in test:
#         src = frames_loc + selected_scripts[dr] + '/'
#         print os.system("rsync -a " + src + dest)

