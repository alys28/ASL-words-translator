import os
import json
#Total files: 2570

def iterate_files(files, data=[], path="MS-ASL/training_video_data"):
    dataIdx = 0
    total = 0
    for file in files:
        title = os.path.splitext(file)[0]
        for i in range(dataIdx, len(data)):
            if data[i]["file"] == title:
                total += 1
                # dataIdx = i
                break # match found, no changes to be made
            elif data[i]["org_text"] == title:
                # os.rename(os.path.join(path, file), os.path.join(path, data[i]["file"]+".mp4"))
                print("Filename changed: ", file)
                
                # dataIdx = i
                # print(i)
                break
        else:
            pass
            # total+=1
            print(file + " ------> "+ str(dataIdx))
    print(total)
    print(len(files))

    

if __name__ == "__main__":
    # path = "MS-ASL/training_video_data_OLD"
    files = os.listdir(path)
    
    filePath = "MS-ASL/MSASL_train.json"
    jsonFile = open(filePath)
    data = json.load(jsonFile)
    # sort file by date created (more efficient timing)
    files.sort(key=lambda fn: os.path.getmtime(os.path.join(path, fn)))
    print("WEIRD FILES:")
    iterate_files(files, data, path)