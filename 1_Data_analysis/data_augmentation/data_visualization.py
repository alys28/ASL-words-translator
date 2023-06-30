#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('jupyter nbconvert --to script data_visualization.ipynb')


# In[1]:


import pandas as pd
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import statistics


# In[2]:


#loading the data

def load_data(file_name):
    df = pd.read_csv(file_name)
    df.head()

    x = []
    y = []
    z = []
    v = []

    #iterate through every row
    for row in range(len(df.index)):
        #iterate through every column
        for col in df:
            if(col[0]=='x'):
                #*i am putting it into floats, the values remain with decimals
                #!error!, by x.append(df[i]), i am appending the WHOLE column to the list.
                    #i just want the first element
                x.append(df[col].iloc[row])
            
            if(col[0]=='y'):
                y.append(df[col].iloc[row])
            
            if(col[0] == 'z'):
                z.append(df[col].iloc[row])
                
            if(col[0]=='v'):
                v.append((df[col].iloc[row]))

    # print(x)
    # print(y)
    # print(z)
    #transform the pandas series into python list
    return [x,y,z,v]

# x = load_data("init_demo.csv")[0]
# # print(x)
# type(x)

# print(x)
# print(y)


# In[3]:


# Plotting
def scatter_3d(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y, z)
    plt.show()


# In[4]:


#plotting x and y
def scatter_2d(x,y, title = "random title"):
    plt.scatter(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(title)
    plt.show()


# In[5]:


def remove_values_from_list(the_list, val):
  return [value for value in the_list if value != val]

#!don't use the average. use the median! (since there are weird points)

def find_center(new_x):
    #find the median in a list
    final_x = remove_values_from_list(new_x, 0)

  
  #~ error: sort will sort the entire data, outside of the scope, while sorted() will only sort within the function
    # sort(x)
    # vs 
    # x.sorted()
    sorted(final_x)


    # print("final_x is ",final_x)
    # return final_x[int(len(final_x)/2)]
    # print(statistics.median (final_x))
    return statistics.mean(final_x)

# file_name = "init_demo.csv"
# x = load_data(file_name)[0]
# print("HI")
# print("HI", find_center(x))


# In[26]:


file_name = '1E8k8gI_xYk3420.csv'

#running:
x = load_data(file_name)[0]
y = load_data(file_name)[1]
z = load_data(file_name)[2]

#!initial 
scatter_2d(x,y, title="No Augmentation Applied")
scatter_3d(x,y,z)

#!transformed
# file_name = 'final_demo.csv'
# scatter_2d(file_name)
# scatter_3d(file_name)


# In[30]:


# #running:
file_name_transformed = '1E8k8gI_xYk3420_rotation_15.csv'
x = load_data(file_name_transformed)[0]
y = load_data(file_name_transformed)[1]
z = load_data(file_name_transformed)[2]

# #!initial 
scatter_2d(x,y, title="Projective Geometry")
scatter_3d(x,y,z)


# In[ ]:




