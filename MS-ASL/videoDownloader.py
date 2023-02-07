import json
import os
import pytube
from pytube import YouTube
from urllib.parse import urlparse
from urllib.parse import parse_qs





def setupData(filePath):
    data = json.load(file)
    return data
   
def getVideosFromURL(data, index, path):
    current_url = data[index]["url"]
    previous_url = data[index - 1]["url"]
    # if urls, are the same, skip current url
    if current_url != previous_url:
        downloadVideo(current_url, path, index)




def getVideoID(url):
    parsed_url = urlparse(url)
    id = parse_qs(parsed_url.query)['v'][0]
    return id 
def downloadVideo(url, path, index):
    try: 
        yt = YouTube(url)
        out_file = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download(path)
        os.rename(out_file, os.path.join(path,f"{getVideoID(url)}.mp4"))
    
        print(f"Downloaded: {url}, index: {index}")
    except Exception as e:
        print(f"DOWNLOAD FAILED (Video Unavailable) -----> index: {index}, REASON: ", e)


if __name__ == "__main__":
    # filePath = "MS-ASL/MSASL_val.json"
    # file  = open(filePath)
    # data = setupData(filePath)
    # for i in range(315, len(data)):
    #     getVideosFromURL(data, i, "/Users/aly/Documents/Programming/Apps/Machine Learning/ASL Converter/MS-ASL/validation_video_data")
    # file.close()
    print(len(os.listdir("/Users/aly/Documents/Programming/Apps/Machine Learning/ASL Converter/MS-ASL/validation_video_data")))
