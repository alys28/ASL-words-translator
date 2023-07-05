#I am using a shorcut, which is nbimporter, in order to import the functions of another notebook in the following python file
import nbimporter
import os
import pandas as pd

from projective_geo import xinlei_vinci
from data_augmentation_final import apply_transformation_on_folder
from fix_reflection_data import swap_coordinates, definePairs, save_file
from data_visualization import visualize


def applyProjectiveGeo(currFilePath):
    xinlei_vinci(currFilePath, negativity = False)
    xinlei_vinci(currFilePath, negativity = True)

def applyReflection(currFilePath, currDir, file):
    df = pd.read_csv(currFilePath)
    new_df = swap_coordinates(df, definePairs())
    save_file(df = new_df, name = file, path = currDir)

def applySarinaTransform(currFileDir):
    apply_transformation_on_folder(currFileDir)

def visualizeData(currFilePath):
    visualize(currFilePath)

def applyTransformations(index, currFilePath, currDir, file):
    print("index")
    if index == 0:
        applyProjectiveGeo(currFilePath)
    if index == 1:
        applyReflection(currFilePath, currDir, file)
    if index == 2:
        applySarinaTransform(currDir)


def transformMe(DATADIR, NUMTRANS):
    for index in range(2,NUMTRANS):
        #running through all the files
        FOLDERS = ['train', 'test']
        for folder in FOLDERS:
            currDir = os.path.join(DATADIR, folder)
            labels = os.listdir(currDir)

            for label in labels:
                #augmenting the files
                if folder != ".DS_Store":
                    currDir = os.path.join(currDir, label)
                    files = os.listdir(currDir)
                    # print(files)

                    for file in files:
                        currfilePath = os.path.join(currDir, file)
                        applyTransformations(index, currfilePath, currDir, file)
                        # visualize(currfilePath)

#following value should be changed
DATADIR = r"D:\Personnel\Other learning\Programming\Personal_projects\ASL_Language_translation\1_Data_analysis\data_augmentation\Data_augmentation\new_files"
#number of transformations that we have.
NUMTRANS = 3

os.chdir(DATADIR)
print(os.getcwd())

transformMe(DATADIR, NUMTRANS)
# applySarinaTransform(DATADIR + r"\train")