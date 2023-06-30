#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#converting jupyter notebook to python file
get_ipython().system('jupyter nbconvert --to script data_augmentation_final.ipynb')


# In[8]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from math import sin, cos
import random
from pathlib import Path


# In[2]:


get_ipython().run_line_magic('run', 'data_visualization.ipynb')


# In[3]:


# Get all labels
names = []
for dir in os.listdir("/Users/aly/Documents/Programming/Apps/Machine Learning/ASL Converter/training_models/mediapipe/reformatting-the-data/data_25_labels"):
    if dir !=  '.DS_Store':
        names.append(dir)
names


# In[3]:


test_arr = np.array([[1,2,3,4],[5,6,7,8]], dtype=np.float32)
# test_arr[:,:2]
origin = np.mean(test_arr[:,:2], axis=0)
test_arr[:,:2] -= origin
origin, test_arr


# In[4]:


test_arr[:,2] = 1
test_arr


# In[5]:


test_arr[:, 1]


# In[6]:


plt.scatter(test_arr[:,0], test_arr[:, 1])
plt.scatter(0, 0, c="red")
plt.arrow(0,0,1,1, width=0.03)
plt.show()


# In[7]:


# df = pd.read_csv("/Users/aly/Documents/Programming/Apps/Machine Learning/ASL Converter/demo.csv")
df = pd.read_csv("/Users/aly/Documents/Programming/Apps/Machine Learning/ASL Converter/demo.csv")
df


# In[8]:


#test

frame = df.iloc[0].to_numpy()
frame = frame.reshape((75, 4))
frame[0]
# new_frame = rotate(frame, 5)
# print(new_frame[0])
# print(frame[0])


# In[9]:


df.to_numpy().shape


# In[10]:


all_frames = df.to_numpy().reshape((len(df), 75, 4))
all_frames[0][0]
all_frames.shape


# In[11]:


print(f"standard variation x: {np.std(all_frames[:, :, 0])}")
print(f"standard variation y: {np.std(all_frames[:, :, 1])}")
print(f"standard variation z: {np.std(all_frames[:, :, 2])}")


# In[12]:


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


# In[16]:


#rotation without changing origin
_, f1 = rotate(all_frames[0], 10)


# In[17]:


_, frame = change_origin(all_frames[0])
_, _=rotate(frame, 10)


# In[9]:


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
    
  def reflection_matrix(self):
    return np.array([[(-1)**(int(self.reflect)), 0, 0],
                     [0, 1, 0], 
                     [0, 0, 1]])
    
  def change_origin(self, frame):
    new_origin = np.mean(frame[:,:2], axis=0)
    new_frame = np.copy(frame)
    new_frame[:,:2] -= new_origin
    return new_frame 


# In[10]:


def add_to_class(Class):  
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper


# In[11]:


@add_to_class(VectorTransformation)
def transform(self, frame, change_shear_direction=False):
    new_frame = np.copy(frame)
    # new_frame = np.reshape((1, 3))
    new_frame[:, 2] = 1

    if self.center:
            new_frame = self.change_origin(new_frame)

    for i in range(len(new_frame)):
        new_frame[i][:3] = np.matmul(self.reflection_matrix(), new_frame[i][:3])
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


# In[22]:


data_aug = VectorTransformation(shear_x=0.2, reflection=False, center_data=True)
new_frame = data_aug.transform(all_frames[0])
# new_frame


# In[23]:


_, frame = change_origin(new_frame)
data_aug.visualize_change(all_frames[0], frame)


# ##random funcitons
# 

# In[13]:


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


  # return new_data, new_labels
  return new_data


# ##generating data

# In[17]:


data_augmentation = VectorTransformation(translation_x=0.1, translation_y=0.1, shear_x=0.1, shear_y=0.1, reflection=False,
                                         rotation_angle=20, center_data=False, random=True)

new_data = data_augmentation.get_new_data(all_frames, 4)
new_data = np.array(list(new_data))
new_data.shape


# In[29]:


print(all_frames[50][0], new_data[0][50][0])
all_frames[10][0], new_data[0][10][0]


# In[30]:


data_augmentation.visualize_change(all_frames[50], new_data[0][50])


# In[ ]:


# Evaluate which values are good for each transformation
# Maybe use Unsupervised learning and cluster the augmented data with the original data as centroids


# ##Visualization on shapes

# In[31]:


square = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=np.float32)
plt.plot(square[:, 0], square[:, 1])
plt.scatter(square[:, 0], square[:, 1])
plt.xlim([-0.5, 1.5])
plt.ylim([-0.5, 1.5])
plt.axhline(y=0, color="black")
plt.axvline(x=0, color="black")


# In[32]:


vt = VectorTransformation(translation_x=1, center_data=False, reflection=False)
new_square = vt.transform(square)
new_square


# In[33]:


plt.plot(new_square[:, 0], new_square[:, 1])
plt.xlim([-2, 3])
plt.ylim([-2, 3])
plt.axhline(y=0, color="black")
plt.axvline(x=0, color="black")


# In[34]:


vt.visualize_change(square, new_square, plot=True)


# In[35]:


vt.shx = 0.5
vt.shy = 1
vt.center = True
square3 = vt.transform(square[:-1])
square3


# In[36]:


square3 = np.append(square3, [square3[0]]).reshape(square.shape)
vt.visualize_change(square, square3, plot=True)


# In[37]:


vt2 = VectorTransformation(shear_x = 2, center_data=False, reflection=False)

r1 = np.array([[-1, 0, 1], [1, 0, 1], [1, 1, 1], [-1, 1, 1], [-1, 0, 1]], dtype=np.float32)

r2 = vt2.transform(r1[:-1], change_shear_direction=True)
r2 


# In[38]:


r2 = np.append(r2, [r2[0]]).reshape(r1.shape)
vt2.visualize_change(r1, r2, plot=True)


# In[39]:


data_augmentation = VectorTransformation(translation_x=0.1, translation_y=0.1, shear_x=0.1, shear_y=0.1, reflection=False,
                                         rotation_angle=20, center_data=False, random=True)
frame = df.iloc[0].to_numpy()
frame = frame.reshape((75, 4))

x = data_augmentation.get_new_data(frames=all_frames, ratio=3, random=True)
x = np.array(x)
x.shape


# In[40]:


data_augmentation = VectorTransformation(translation_x=0.1, translation_y=0.1, shear_x=0.1, shear_y=0.1, reflection=False,
                                         rotation_angle=20, center_data=False, random=True)
frame = df.iloc[0].to_numpy()
frame = frame.reshape((75, 4))

x = data_augmentation.get_new_data(frames=all_frames, ratio=3, random=True)
x = np.array(x)
all_frames.shape

x[0].reshape((-1, 300)).shape


# In[41]:


path_4_labels = "/Users/aly/Documents/Programming/Apps/Machine Learning/ASL Converter/data_augmentation/data_4_labels_augmentation"
data_augmentation = VectorTransformation(translation_x=0.1, translation_y=0.1, shear_x=0.1, shear_y=0.1, reflection=False,
                                         rotation_angle=20, center_data=False, random=True)
num_coords = 21 + 21 + 33
headerList = []
for val in range(1, num_coords+1):
    headerList += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
for folder in os.listdir(path_4_labels):
    if folder != ".DS_Store":
        files = os.listdir(os.path.join(path_4_labels, folder))
        for file in files:
            if file != ".DS_Store":
                file_path = (os.path.join(path_4_labels, folder, file))
                df = df.drop("class", axis=1)
                all_frames = df.to_numpy()
                print(all_frames.shape)
                all_frames = all_frames.reshape((len(df), 75, 4))
                new_frames = np.array(data_augmentation.get_new_data(frames=all_frames, ratio=3, random=True))
                for i in range(len(new_frames)):
                     # Rename the file
                    name = file + "_AUGMENTED_" + str(i) + ".csv"
                    video = new_frames[i].reshape((-1, 300))
                    pd.DataFrame(video).to_csv(os.path.join(path_4_labels, folder, name), header=headerList, index=False)


# In[42]:


data_augmentation = VectorTransformation(scale_x=0.2, scale_x=0.2, shear_x=0.1, shear_y=0.1, reflection=False,
                                         rotation_angle=20, center_data=False, random=True)
frame = df.iloc[0].to_numpy()
frame = frame.reshape((75, 4))

x = data_augmentation.get_new_data(frames=all_frames, ratio=3, random=True)
x = np.array(x)
x.shape


# In[18]:


def apply_transformation_on_array(transformation_func, arr):
    new_arr = []
    for frame in arr:
        new_frame = np.copy(frame)
        for i in range(frame.shape[0]):
            new_frame = transformation_func(new_frame)
          # new_frame[i][:3] = np.matmul(transformation_func(), new_frame[i][:3])
        new_arr.append(new_frame)
    return np.array(new_arr)




# In[30]:


get_ipython().run_line_magic('run', 'projective_geo.ipynb')


# In[20]:


def apply_transformation_on_data(folder_path, file_path, name, vectorObject, label):
    df = pd.read_csv(file_path)
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
    new_file = df.to_csv(os.path.join(folder_path, label, name), index=False)
    return new_file


# In[21]:


vectorObject=VectorTransformation(translation_x=-0.5, translation_y=-0.5, center_data=True, random=True)
def apply_transformation_on_folder(folder_path):
    for folder in os.listdir(os.path.join(folder_path)):
        if folder != ".DS_Store":
            files = os.listdir(os.path.join(folder_path, folder))
            for file in files:
                apply_transformation_on_data(folder_path, file_path=os.path.join(folder_path, folder, file),
                                             name=os.path.splitext(file)[0] + "_translate_0.5_left" + ".csv", vectorObject=vectorObject, label=folder)

                


# In[22]:


apply_transformation_on_folder("/Users/aly/Documents/Programming/Apps/Machine Learning/ASL Converter/data_augmentation/data_four_labels_augmentation/")


# In[31]:


def apply_projective_geo(folder_path):
    for folder in os.listdir(os.path.join(folder_path)):
        if folder != ".DS_Store":
            files = os.listdir(os.path.join(folder_path, folder))
            for file in files:
                xinlei_vinci(os.path.join(folder_path, folder, file), negativity = False)
                xinlei_vinci(os.path.join(folder_path, folder, file), negativity = True)


# In[32]:


apply_projective_geo("/Users/aly/Documents/Programming/Apps/Machine Learning/ASL Converter/data_augmentation/data_four_labels_augmentation/")


# In[ ]:


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

