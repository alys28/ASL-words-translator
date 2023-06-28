#I am using a shorcut, which is nbimporter, in order to import the functions of another notebook in the following python file
import nbimporter
import os



def main():
    #running through all the files
    FOLDERS = ['train', 'test']
    DATADIR = "D:/Personnel/Other learning/Programming/Personal_projects/ASL_Language_translation/000_Database/data_25_labels_augmentation/"
    labels = []
    for folder in FOLDERS:
        labels = os.listdir(os.path.join(DATADIR, folder))

    for label in labels:
        #augmenting the files
        listdir =   

main()