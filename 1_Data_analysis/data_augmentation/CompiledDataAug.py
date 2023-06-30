#I am using a shorcut, which is nbimporter, in order to import the functions of another notebook in the following python file
import nbimporter
import os
import pandas as pd

from projective_geo import xinlei_vinci
from fix_reflection_data import swap_coordinates, definePairs, save_file

def applyProjectiveGeo(currFilePath):
    xinlei_vinci(currFilePath, negativity = False)
    xinlei_vinci(currFilePath, negativity = True)

def applyReflection(currFilePath, currDir, file):
    df = pd.read_csv(currFilePath)
    new_df = swap_coordinates(df, definePairs())
    save_file(df = new_df, name = file, path = currDir)

def applyTransformations(index, currFilePath, currDir, file):
    if index ==0:
        applyProjectiveGeo(currFilePath)
    if index == 1:
        applyReflection(currFilePath, currDir, file)


def transformMe(DATADIR):
    for index in range(0,2):
        #running through all the files
        FOLDERS = ['train', 'test']
        for folder in FOLDERS:
            currDir = os.path.join(DATADIR, folder)
            labels = os.listdir(currDir)

            for label in labels:
                #augmenting the files
                currDir = os.path.join(currDir, label)
                files = os.listdir(currDir)
                # print(files)

                for file in files:
                    currfile = os.path.join(currDir, file)
                    applytransformations(index, currfile, currDir, file)

#following value should be changed
DATADIR = "D:\Personnel\Other learning\Programming\Personal_projects\ASL_Language_translation\1_Data_analysis\data_augmentation\Data_augmentation\new_files"
os.chdir(DATADIR)
print(os.getcwd())
# DATADIR = "D:/Personnel/Other learning/Programming/Personal_projects/ASL_Language_translation/000_Database/data_25_labels_augmentation/"
transformMe(DATADIR)