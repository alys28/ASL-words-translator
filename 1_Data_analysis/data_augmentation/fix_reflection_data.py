#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#converting file to python script
get_ipython().system('jupyter nbconvert --to script fix-reflection-data.ipynb')


# In[1]:


import pandas as pd
import numpy as np
import os


# In[121]:


csv = pd.read_csv("demo_copy.csv")
csv


# In[137]:


data = csv.drop(columns="class")
data


# In[2]:


# Associate coordinates as pairs
# row = pose_row + right_hand_row + left_hand_row
# Pose: 1-33
# right hand: 34-54
# left hand: 55-75
def definePairs():
    pairs = [(2,5), (3, 6), (4,7), (8,9), (10,11), (12,13), (14, 15), (16,17), (18,19), (20, 21), (22, 23), (24, 25), (26, 27), (28, 29), (30, 31), (32, 33), (34, 55),
    (35, 56),
    (36, 57),
    (37, 58),
    (38, 59),
    (39, 60),
    (40, 61),
    (41, 62),
    (42, 63),
    (43, 64),
    (44, 65),
    (45, 66),
    (46, 67),
    (47, 68),
    (48, 69),
    (49, 70),
    (50, 71),
    (51, 72),
    (52, 73),
    (53, 74),
    (54, 75)
    ]

    return pairs


# TESTING THINGS OUT WITH PANDAS

# In[161]:


dic = {'A': [1, 4, 1, 4], 'B': [9, 2, 5, 3], 'C': [0, 0, 5, 3]}
df = pd.DataFrame(dic)


# In[162]:


df['B'] = [ 4, 4, 4, 0]
df['A'] = [ 4, 4, 4, 0]
df


# In[4]:


def swap_coordinates(data, pairs):
    i = 1
    new_data = data.copy()
    for (small_coord, big_coord) in pairs:
        # Invert coord between small and big
        small_x = data[f"x{small_coord}"]
        small_y = data[f"y{small_coord}"]
        small_z = data[f"z{small_coord}"]
        new_data[f"x{small_coord}"] = data[f"x{big_coord}"]
        new_data[f"y{small_coord}"] = data[f"y{big_coord}"]
        new_data[f"z{small_coord}"] = data[f"z{big_coord}"]
        new_data[f"x{big_coord}"] = small_x
        new_data[f"y{big_coord}"] = small_y
        new_data[f"z{big_coord}"] = small_z
        i += 1
    return new_data

# new_data = swap_coordinates(data, pairs)
# new_data


# In[178]:


# data["x1"] = data["y1"]
data


# In[93]:





# In[5]:


def save_file(df, name, path):
    new_file = df.to_csv(os.path.join(path, name), index=False)


# In[174]:


save_file(new_data, "demo_copy.csv", "C:/Users/malik/Desktop/ASL-Aly/ASL-words-translator/data_augmentation")


# In[11]:
#rewriting the function to accomodate new transformation file
def fixReflectionFinal(folderPath, pairs):

def fix_reflection(folder_path, pairs):
    for label in os.listdir(folder_path):
        print(label)
        if label != ".DS_Store":
            for file in os.listdir(os.path.join(folder_path, label)):
                if file != ".DS_Store":
                    if not("reflection" in file):
                        csv = pd.read_csv(os.path.join(folder_path, label, file))
                        # data = csv.drop(columns="class")
                        new_data = swap_coordinates(csv, pairs)
                        save_file(new_data, name=os.path.splitext(file)[0] + "_REFLECTION_" + ".csv", path=os.path.join(folder_path, label))
fix_reflection("/Users/aly/Documents/Programming/Apps/Machine Learning/ASL Converter/data_augmentation/data_four_labels_augmentation", definePairs)            


# In[13]:


# delete "old reflection files"
folder_path = "/Users/aly/Documents/Programming/Apps/Machine Learning/ASL Converter/data_augmentation/data_four_labels_augmentation"
for label in os.listdir(folder_path):
        if label != ".DS_Store":
                for file in os.listdir(os.path.join(folder_path, label)):
                        if file != ".DS_Store":
                                if ("reflection" in file):
                                        os.remove(os.path.join(folder_path, label, file))


# In[183]:





# In[ ]:




