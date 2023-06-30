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
    new_df = swap_coordinates(csv, definePairs())
    save_file(df = new_df, name = file, path = currDir)

def applyTransformations(currFilePath, currDir, file):
    applyProjectiveGeo(currFilePath)
    applyReflection(currFilePath, currDir, file)

def transformMe(DATADIR):
    #running through all the files
    FOLDERS = ['train', 'test']
    labels = []
    for folder in FOLDERS:
        currDir = os.path.join(DATADIR, folder)
        features = os.listdir(currDir)

        for feature in features:
            #augmenting the files
            currDir = os.path.join(currDir, feature)
            files = os.listdir(currDir)
            print(files)

            for file in files:
                currFile = os.path.join(currDir, file)
                applyTransformations(currFile, currDir, file)

#following value should be changed
DATADIR = "D:/Personnel/Other learning/Programming/Personal_projects/ASL_Language_translation/000_Database/data_25_labels_augmentation/"
transformMe(DATADIR)