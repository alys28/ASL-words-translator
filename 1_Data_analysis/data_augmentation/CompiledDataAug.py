#I am using a shorcut, which is nbimporter, in order to import the functions of another notebook in the following python file
# import nbimporter
import os
import pandas as pd

from projective_geo import xinlei_vinci
from data_augmentation_final import apply_transformation_on_folder
from fix_reflection_data import swap_coordinates, definePairs, save_file
from data_visualization import visualize


def applyProjectiveGeo(currDir):
    files = os.listdir(currDir)
    for file in files:
        currfilePath = os.path.join(currDir, file)
        print(currfilePath)
        xinlei_vinci(currfilePath, negativity = False)
        xinlei_vinci(currfilePath, negativity = True)

def applyReflection(currDir):
    files = os.listdir(currDir)
    for file in files:
        currfilePath = os.path.join(currDir, file)
        print(currfilePath)

        df = pd.read_csv(currfilePath)
        new_df = swap_coordinates(df, definePairs())
        save_file(df = new_df, name = file, path = currDir)
    

def applySarinaTransform(currFileDir):
    apply_transformation_on_folder(currFileDir)

def visualizeData(currDir):
    files = os.listdir(currDir)
    for file in files:
        currfilePath = os.path.join(currDir, file)
        visualize(currfilePath)

    # visualize("/Tmp/linxinle/Programming/ASL-words-translator/000_Database/25LabelsData/test/book/0UsjUE-TXns0.csv")
    # visualize("/Tmp/linxinle/Programming/ASL-words-translator/1_Data_analysis/data_augmentation/Data_augmentation/new_files/train/label/9bosgmeAAuo7031.csv")

def applyTransformations(index, currDir):
    print("Transformation number: ", index)
    if index == 0:
        applyProjectiveGeo(currDir)
    if index == 1:
        applyReflection(currDir)
    if index == 2:
        applySarinaTransform(currDir)


def transformMe(DATADIR, NUMTRANS):
    
    for index in range(0,NUMTRANS):
        #running through all the files
        FOLDERS = ['train', 'test']
        for folder in FOLDERS:
            currDir = os.path.join(DATADIR, folder)
            labels = os.listdir(currDir)

            for label in labels:
                files = os.listdir(os.path.join(currDir, label))

                #augmenting the files
                currLabelDir = os.path.join(currDir, label)
                applyTransformations(index, currLabelDir)
                

                #visualizing the files
                # visualizeData(os.path.join(currDir, label))

#following value should be changed
# DATADIR = r"D:\Personnel\Other learning\Programming\Personal_projects\ASL_Language_translation\1_Data_analysis\data_augmentation\Data_augmentation\new_files"
# DATADIR = r"/Tmp/linxinle/Programming/ASL-words-translator/1_Data_analysis/data_augmentation/Data_augmentation/new_files"#number of transformations that we have.
DATADIR = r"/Tmp/linxinle/Programming/ASL-words-translator/000_Database/Aug25LabelsData"
# DATADIR = r"/Tmp/linxinle/Programming"
NUMTRANS = 3

os.chdir(DATADIR)
print(os.getcwd())

transformMe(DATADIR, NUMTRANS)
# applySarinaTransform(DATADIR + r"\train")
# visualizeData(DATADIR)