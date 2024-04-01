#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#converting jupyter notebook to python file
# get_ipython().system('jupyter nbconvert --to script data_augmentation_final.ipynb')


# In[8]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from math import sin, cos
import random
from pathlib import Path


def change_origin(frame):
  new_origin = np.mean(frame[:,:2], axis=0)
  # print(new_origin.shape)
  new_frame = np.copy(frame)
  new_frame[:,:2] -= new_origin
  return frame, new_frame 


# In[13]:


def visualize_rotation(func):

  def wrap(*args, **kwargs):
    original_frame, new_frame = func(*args, **kwargs)
    plt.scatter(original_frame[:, 0], original_frame[:, 1], c="blue", label="original_data")
    plt.scatter(new_frame[:, 0], new_frame[:, 1], c="red", label="new data")
    plt.scatter(0, 0, c="black")
    plt.annotate("origin", (0,0))
    plt.arrow(0, 0, original_frame[0][0], original_frame[0][1], length_includes_head=True, color="blue", head_width=0.03)
    plt.arrow(0, 0, new_frame[0][0], new_frame[0][1], length_includes_head=True, color="red", head_width=0.03)
    plt.legend()
    return original_frame, new_frame

  return wrap


# In[14]:


def visualize(frame):
    plt.scatter(frame[:, 0], frame[:, 1])


# In[15]:


@visualize_rotation
def rotate(frame, angle, origin=[0, 0]):
    normalized_frame = change_origin(frame)
    angle_rad = math.radians(angle)
    new_frame = np.copy(frame)
    rotation_matrix = np.array([[cos(angle_rad), -sin(angle_rad)], [sin(angle_rad), cos(angle_rad)]])
    for i in range(len(new_frame)):
        new_frame[i][:2] = np.matmul(rotation_matrix, new_frame[i][:2])
    return frame, new_frame

class VectorTransformation():

  def __init__(self, center_data=True, translation_x=0, translation_y=0, rotation_angle=0, 
               shear_x=0, shear_y=0, scaling_x=0, scaling_y=0, reflection=True, random=False):
    self.random = random
    self.center = center_data
    self.reflect = reflection

    self.tx_max = translation_x
    self.ty_max = translation_y
    self.angle_max = math.radians(rotation_angle)
    self.shx_max = shear_x
    self.shy_max = shear_y
    self.scale_x_max = scaling_x
    self.scale_y_max = scaling_y

    self.tx = translation_x
    self.ty = translation_y
    self.angle = math.radians(rotation_angle)
    self.shx = shear_x
    self.shy = shear_y
    self.scale_x= scaling_x
    self.scale_y = scaling_y

  def translation_matrix(self):
    return np.array([[1, 0, self.tx],
                    [0, 1, self.ty],
                    [0, 0, 1]])

  def rotation_matrix(self):
    return np.array([[cos(self.angle), -sin(self.angle), 0], 
                       [sin(self.angle), cos(self.angle), 0],
                       [0, 0, 1]])

  def shearx_matrix(self, sign=0):
    return np.array([[1, ((-1)**sign)*self.shx, 0],
                     [0, 1, 0], 
                     [0, 0, 1]])
    
  def sheary_matrix(self, sign=0):
    return np.array([[1, 0, 0],
                     [((-1)**sign)*self.shy, 1, 0],
                     [0, 0, 1]])
    
  def scaling_matrix(self):
    return np.array([[1+self.scale_x, 0, 0],
                     [0, 1+self.scale_y, 0], 
                     [0, 0, 1]])
    
  #reflection is incorrect
  def reflection_matrix(self):
    pass
    # return np.array([[(-1)**(int(self.reflect)), 0, 0],
    #                  [0, 1, 0], 
    #                  [0, 0, 1]])
    
  def change_origin(self, frame):
    new_origin = np.mean(frame[:,:2], axis=0)
    new_frame = np.copy(frame)
    new_frame[:,:2] -= new_origin
    return new_frame 

def add_to_class(Class):  
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper





@add_to_class(VectorTransformation)
def transform(self, frame, change_shear_direction=False):
    new_frame = np.copy(frame)
    # new_frame = np.reshape((1, 3))
    new_frame[:, 2] = 1

    if self.center:
            new_frame = self.change_origin(new_frame)

    for i in range(len(new_frame)):
        # new_frame[i][:3] = np.matmul(self.reflection_matrix(), new_frame[i][:3])
        # print(new_frame[i][:3])
        new_frame[i][:3] = np.matmul(self.translation_matrix(), new_frame[i][:3])
        if change_shear_direction:
          new_frame[i][:3] = np.matmul(self.shearx_matrix(int(new_frame[i][0]*new_frame[i][1]<0)), new_frame[i][:3])
          # print("shear x:", self.shearx_matrix(int(new_frame[i][0]*new_frame[i][1]<0)))
          # print("sign: ", int(new_frame[i][0]*new_frame[i][1]<0))
          new_frame[i][:3] = np.matmul(self.sheary_matrix(int(new_frame[i][0]*new_frame[i][1]<0)), new_frame[i][:3])
        else:
          new_frame[i][:3] = np.matmul(self.shearx_matrix(), new_frame[i][:3])
          # print("shear x:", self.shearx_matrix())
          new_frame[i][:3] = np.matmul(self.sheary_matrix(), new_frame[i][:3])

        # print(new_frame[i][:3])
        new_frame[i][:3] = np.matmul(self.rotation_matrix(), new_frame[i][:3])
        new_frame[i][:3] = np.matmul(self.scaling_matrix(), new_frame[i][:3])

    new_frame[:, 2] = frame[:, 2]
        
    return new_frame


# In[12]:


@add_to_class(VectorTransformation)
def visualize_change(self, original_frame, new_frame, plot=False):
    if plot:
      plt.plot(original_frame[:, 0], original_frame[:, 1], c="blue")
      plt.plot(new_frame[:, 0], new_frame[:, 1], c="red")
    plt.title(f"translation: ({self.tx:.3f}, {self.ty:.3f})\nrotation: {self.angle:.3f},\n shear: ({self.shx:.3f}, {self.shy:.3f})\n scaling: ({self.scale_x:.3f}, {self.scale_y:.3f})")
    plt.scatter(original_frame[:, 0], original_frame[:, 1], c="blue", label="original_data")
    plt.scatter(new_frame[:, 0], new_frame[:, 1], c="red", label="new data")
    plt.scatter(0, 0, c="black")
    plt.annotate("origin", (0,0))
    plt.arrow(0, 0, original_frame[0][0], original_frame[0][1], length_includes_head=True, color="blue", head_width=0.03)
    plt.arrow(0, 0, new_frame[0][0], new_frame[0][1], length_includes_head=True, color="red", head_width=0.03)
    plt.axhline(y=0, color="black")
    plt.axvline(x=0, color="black")
    plt.legend()
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([min(np.amin(original_frame[:, 0]), np.amin(new_frame[:, 0]))-1, max(np.amax(original_frame[:, 0]), np.amax(new_frame[:, 0]))+1])
    plt.ylim([min(np.amin(original_frame[:, 1]), np.amin(new_frame[:, 1]))-1, max(np.amax(original_frame[:, 1]), np.amax(new_frame[:, 1]))+1])


@add_to_class(VectorTransformation)
def set_random_variables(self):
  if self.random:
    self.tx = random.uniform(0, self.tx_max)
    self.ty = random.uniform(0, self.ty_max)
    self.angle = random.uniform(0, self.angle_max)
    self.shx = random.uniform(0, self.shx_max)
    self.shy = random.uniform(0, self.shy_max)
    self.scale_x= random.uniform(0, self.scale_x_max)
    self.scale_y = random.uniform(0, self.scale_y_max)


# In[14]:


import random 

@add_to_class(VectorTransformation)
def random_translation_matrix(self):
    return np.array([[1, 0, random.uniform(0, self.tx)],
                         [0, 1, self.ty],
                         [0, 0, 1]])

@add_to_class(VectorTransformation)
def random_rotation_matrix(self):
    return np.array([[cos(random.uniform(0, self.angle)), -sin(random.uniform(0,self.angle)), 0], 
                    [sin(random.uniform(0, self.angle)), cos(random.uniform(0, self.angle)), 0],
                    [0, 0, 1]])
  
@add_to_class(VectorTransformation)
def random_shearx_matrix(self):
    return np.array([[1, random.uniform(0, self.shx), 0],
                     [0, 1, 0], 
                     [0, 0, 1]])
    
@add_to_class(VectorTransformation)
def random_sheary_matrix(self):
  return np.array([[1, 0, 0],
                  [random.uniform(0, self.shy), 1, 0],
                  [0, 0, 1]])
    
@add_to_class(VectorTransformation)
def random_scaling_matrix(self):
  return np.array([[random.uniform(1, self.scale_x), 0, 0],
                  [0, random.uniform(1, self.scale_y), 0], 
                  [0, 0, 1]])


# In[15]:


@add_to_class(VectorTransformation)
def random_transform(self, frame):
    new_frame = np.copy(frame[:, :2])
    # new_frame = new_frame.reshape((1, 3))
    new_frame[:, 2] = 1

    #centering
    if self.center:
      new_frame = self.change_origin(frame)

    for i in range(len(new_frame)):
        new_frame[i][:3] = np.matmul(self.reflection_matrix(), new_frame[i][:3])
        new_frame[i][:3] = np.matmul(self.random_translation_matrix(), new_frame[i][:3])
        new_frame[i][:3] = np.matmul(self.random_rotation_matrix(), new_frame[i][:3])
        new_frame[i][:3] = np.matmul(self.random_shearx_matrix(), new_frame[i][:3])
        new_frame[i][:3] = np.matmul(self.random_sheary_matrix(), new_frame[i][:3])
        new_frame[i][:3] = np.matmul(self.random_scaling_matrix(), new_frame[i][:3])

    new_frame[:, 2] = frame[:, 2]
        
    return new_frame


# In[16]:


@add_to_class(VectorTransformation)
def get_new_data(self, frames: list([]), ratio, random=True):
  new_data = []
  # new_labels = []
  for j in range(ratio):
    new_video = []
    self.set_random_variables()
    for i in range(len(frames)):
      new_video.append(self.transform(frames[i]))
        # new_labels.append(labels[i])
    new_data.append(new_video)
  return new_data


def apply_transformation_on_array(transformation_func, arr):
    new_arr = []
    for frame in arr:
        new_frame = np.copy(frame)
        for i in range(frame.shape[0]):
            new_frame = transformation_func(new_frame)
          # new_frame[i][:3] = np.matmul(transformation_func(), new_frame[i][:3])
        new_arr.append(new_frame)
    return np.array(new_arr)

def apply_transformation_on_data(folder_path, file_path, name, vectorObject):
    df = pd.read_csv(file_path)
    label = df["class"].iloc[0]
    df = df.drop("class", axis=1)
    all_frames = df.to_numpy()
    print(all_frames.shape)
    num_coords = 21 + 21 + 33
    headerList = []
    for val in range(1, num_coords+1):
        headerList += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
    all_frames = all_frames.reshape((len(df), 75, 4))
    new_frames = np.array(vectorObject.get_new_data(all_frames, ratio=1))
    video = new_frames.reshape((-1, 300))
    df = pd.DataFrame(video)
    df.columns = headerList
    df.insert(0, "class", [label for i in range(video.shape[0])])
    new_file = df.to_csv(os.path.join(folder_path, name), index=False)
    return new_file

def apply_transformation_on_folder(folder_path):
# center_data=True, translation_x=0, translation_y=0, rotation_angle=0, shear_x=0, shear_y=0, scaling_x=0, scaling_y=0, reflection=True, random=False
    # for folder in os.listdir(os.path.join(folder_path)):
        # if folder != ".DS_Store":
    files = os.listdir(folder_path)
    max_trans = [0.5, 10.0, 0.2, 0.2]
    for transformation in range(0, len(max_trans)):
        trans = [max_trans[i] if i == transformation else 0 for i in range(0,len(max_trans))]
        print(trans)
        for file in files:
          vectorObject=VectorTransformation(center_data = False, translation_x = trans[0], translation_y = random.uniform(0, trans[0]), rotation_angle = trans[1], shear_x = 0, shear_y = 0, scaling_x = trans[2], scaling_y = trans[3], reflection = False, random=True)
          apply_transformation_on_data(folder_path, file_path=os.path.join(folder_path, file),
                                      name=os.path.splitext(file)[0] + "_transform" + ".csv", vectorObject=vectorObject)

                
def apply_projective_geo(folder_path):
    for folder in os.listdir(os.path.join(folder_path)):
        if folder != ".DS_Store":
            files = os.listdir(os.path.join(folder_path, folder))
            for file in files:
                xinlei_vinci(os.path.join(folder_path, folder, file), negativity = False)
                xinlei_vinci(os.path.join(folder_path, folder, file), negativity = True)


def translation_matrix(self):
    return np.array([[1, 0, self.tx],
                    [0, 1, self.ty],
                    [0, 0, 1]])

def rotation_matrix(self):
  return np.array([[cos(self.angle), -sin(self.angle), 0], 
                      [sin(self.angle), cos(self.angle), 0],
                      [0, 0, 1]])

def shearx_matrix(self, sign=0):
  return np.array([[1, ((-1)**sign)*self.shx, 0],
                    [0, 1, 0], 
                    [0, 0, 1]])
  
def sheary_matrix(self, sign=0):
  return np.array([[1, 0, 0],
                    [((-1)**sign)*self.shy, 1, 0],
                    [0, 0, 1]])
  
def scaling_matrix(self):
  return np.array([[1+self.scale_x, 0, 0],
                    [0, 1+self.scale_y, 0], 
                    [0, 0, 1]])
  
def reflection_matrix(self):
  return np.array([[(-1)**(int(self.reflect)), 0, 0],
                    [0, 1, 0], 
                    [0, 0, 1]])