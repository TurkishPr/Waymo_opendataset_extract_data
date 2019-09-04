
import os, sys
import imp
import tensorflow as tf
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
from PIL import Image
import binascii
import time

# TODO: Change this to your own setting
os.environ['PYTHONPATH']='/env/python:~/github/waymo-open-dataset'
m=imp.find_module(r'C:\Users\Joseph Kim\Desktop\Personal\Waymo\waymo-od\waymo_open_dataset', ['.'])
# m=imp.find_module('waymo_open_dataset', [r'C:\Users\Joseph Kim\Desktop\Personal\Waymo\waymo-od\'])
# print(str(m[0])+ str(m[1])+ str(m[2])) None C:\Users\Joseph Kim\Desktop\Personal\Waymo\waymo-od\waymo_open_dataset ('', '', 5)
imp.load_module('waymo_open_dataset', m[0], m[1], m[2])

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2 as open_dataset2

tf.compat.v1.enable_eager_execution()
count =0

def image_show(data, name, layout, cmap=None):
  """Show an image."""
  plt.subplot(*layout)
  plt.imshow(tf.image.decode_jpeg(data), cmap=cmap)
  plt.title(name)
  plt.grid(False)
  plt.axis('off')



#  /$$$$$$$$ /$$   /$$ /$$$$$$$$ /$$$$$$$   /$$$$$$   /$$$$$$  /$$$$$$$$       /$$$$$$ /$$      /$$  /$$$$$$ 
# | $$_____/| $$  / $$|__  $$__/| $$__  $$ /$$__  $$ /$$__  $$|__  $$__/      |_  $$_/| $$$    /$$$ /$$__  $$
# | $$      |  $$/ $$/   | $$   | $$  \ $$| $$  \ $$| $$  \__/   | $$           | $$  | $$$$  /$$$$| $$  \__/
# | $$$$$    \  $$$$/    | $$   | $$$$$$$/| $$$$$$$$| $$         | $$           | $$  | $$ $$/$$ $$| $$ /$$$$
# | $$__/     >$$  $$    | $$   | $$__  $$| $$__  $$| $$         | $$           | $$  | $$  $$$| $$| $$|_  $$
# | $$       /$$/\  $$   | $$   | $$  \ $$| $$  | $$| $$    $$   | $$           | $$  | $$\  $ | $$| $$  \ $$
# | $$$$$$$$| $$  \ $$   | $$   | $$  | $$| $$  | $$|  $$$$$$/   | $$          /$$$$$$| $$ \/  | $$|  $$$$$$/
# |________/|__/  |__/   |__/   |__/  |__/|__/  |__/ \______/    |__/         |______/|__/     |__/ \______/ 
                                                                                                           

# filepath = [r'C:\Users\Joseph Kim\Desktop\training_0000\segment-11119453952284076633_1369_940_1389_940_with_camera_labels.tfrecord']
# FILENAME = filepath

# create list of tfrecord filepaths
folder_path = r"C:\Users\Joseph Kim\Desktop\training_0002" #dir containing the folder with tfrecords to convert/extract
records = []
records = os.listdir(folder_path)
index =0
for record in records:
    records[index] = folder_path + "\\"+ record
    index+=1

dataset = tf.data.TFRecordDataset(records, compression_type='')
frame_count =0
file_num=0
first_time =1

if not os.path.exists(folder_path+"_extract"):
    os.mkdir(folder_path+"_extract")

for data in dataset: #records 명시된 파일들을 TFRecordDataset함수를 통해 읽고 거기서 data하나를 가져옴. 하나의 segment=dataset. data=frame. frame = 5images 한프레임=이미지 5장.
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy())) #텐서는 .numpy() 메서드(method)를 호출하여 넘파이 배열로 변환할 수 있음
                                                   #bytearray(반복가능한객체): 반복 가능한 객체로 바이트 배열 객체를 생성
    if not os.path.exists(folder_path+"_extract\\"+str(frame.context.name) ):
        os.mkdir(folder_path+"_extract\\"+str(frame.context.name) )

    if not os.path.exists(folder_path+"_extract\\"+str(frame.context.name) +"\\general_specs.txt"):
        f1 = open(folder_path+"_extract\\"+str(frame.context.name) +"\\general_specs.txt", 'w')
        f1.write("time_of_day: "+str(frame.context.stats.time_of_day))
        f1.write("\nlocation: "+str(frame.context.stats.location))
        f1.write("\nweather: "+str(frame.context.stats.weather))
      
        for index, camera in enumerate(frame.context.camera_calibrations):
            f1.write("\n\nCamera: " + open_dataset.CameraName.Name.Name(camera.name))
            f1.write("\nwidth: " + str(camera.width)+"\n")
            f1.write("height: " + str(camera.height)+"\n\n")
        f1.close()
        

    #5개의 카메라중, 각각의 이미지에서 보이는 물체들의 박스 좌표와 TYPE, 그리고 각 물체의 id 프린트
    if not os.path.exists(folder_path+"_extract\\"+str(frame.context.name)+"\\frame_data.txt"):
        f = open(folder_path+"_extract\\"+str(frame.context.name) +"\\frame_data.txt", 'a+') #to write frame data to txt file
        # print("\n\nLABLE INFO PER IMAGE")
        for index, cam_labels in enumerate(frame.camera_labels):
            f.write("Camera: " +open_dataset.CameraName.Name.Name(cam_labels.name)+"\n")
            for index2, label in enumerate(cam_labels.labels):
                f.write("type: ")
                if label.type ==0:
                  f.write("unknown\n")
                if label.type ==1:
                  f.write("vehicle\n")              
                if label.type ==2:
                  f.write("pedestrian\n")
                if label.type ==3:
                  f.write("sign\n")         
                if label.type ==4:
                  f.write("cyclist\n") 
                f.write("id: " + label.id+"\n")
                f.write(str(label.box)+"\n")
        f.close()
print("\n***finished fraame data and general spec info extraction***\n")

for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))

    if first_time:
        old_name=str(frame.context.name)
        first_time=0
            
    if old_name!=str(frame.context.name):
      frame_count=0
      file_num+=1

    for index, images in enumerate(frame.images):
        if frame_count==0 and not os.path.exists(folder_path+"_extract\\"+str(frame.context.name) ):
            os.mkdir(folder_path+"_extract\\"+str(frame.context.name) )

        with open(folder_path+"_extract\\"+str(frame.context.name)  +'\\'+str(frame_count//5) + '_'+str(index+1) +'.jpg', 'wb') as image_file:
            image_file.write(images.image)
        '''see images by uncommenting the 2 lines below'''    
        #  image_show(images.image, open_dataset.CameraName.Name.Name(images.name), [2, 3, index+1]) #displays image
    # plt.show() #displays image 
        frame_count+=1

        old_name = str(frame.context.name)


print("IMAGE EXTRACTION ENDED")
time.sleep(5)

#  /$$$$$$$$ /$$   /$$ /$$$$$$$ 
# | $$_____/| $$$ | $$| $$__  $$
# | $$      | $$$$| $$| $$  \ $$
# | $$$$$   | $$ $$ $$| $$  | $$
# | $$__/   | $$  $$$$| $$  | $$
# | $$      | $$\  $$$| $$  | $$
# | $$$$$$$$| $$ \  $$| $$$$$$$/
# |________/|__/  \__/|_______/ 
