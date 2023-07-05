import pandas as pd
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import statistics

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

    return [x,y,z,v]


# Plotting
def scatter_3d(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y, z)
    plt.show()

#plotting x and y
def scatter_2d(x,y, title = "random title"):
    plt.scatter(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(title)
    plt.show()

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

    return statistics.mean(final_x)
    
def visualize(filename):
    #visualizes the first line of the file
    x = load_data(filename)[0]
    y = load_data(filename)[1]
    z = load_data(filename)[2]

    scatter_2d(x,y,title = "First visualization")
    scatter_3d(x,y,z)