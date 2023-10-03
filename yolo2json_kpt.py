import os
import json
import numpy as np
from PIL import Image
import glob

def get_coco_json_format():
    coco_format = {
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }
    return coco_format

def create_images_format(filename, img_width, img_height, image_id):
    images = {
        "file_name": filename,
        "id": image_id, # image id, id cannotdo not repeat
        "width": img_width,
        "height": img_height
    }
    return images

def create_categories_format(category_dict):
    category_list = []
    for key, value in category_dict.items():
        category = {
            "supercategory": key,
            "id": value,
            "name": key,
            #TODO: add "keypoints": the name of the point serial number
            #TODO: add "skeleton": skeleton composed of points, not necessary for training
        }
        category_list.append(category)
    return category_list

def create_annotations_format(data, image_id, category_id, annotation_id):
    bbox = data[annotation_id][1]
    keypoints = data[annotation_id][2]
    annotations = {
        "category_id": category_id, # The category to which the instance belongs
        "num_keypoints": 5, # the number of marked points of the instance
        "bbox": bbox, # location of detection box,format is x, y, w, h
        # N*3 list of x, y, v.
        "keypoints": keypoints,
        "id": annotation_id, # the id of the instance, id cannot repeat
        "image_id": image_id # The id of the image where the instance is located, repeatable. This represents the presence of multiple objects on a single image
        #TODO: add "iscrowd": covered or not, when the value is 0, it will participate in training
        #TODO: add "area": # the area occupied by the instance, can be simply taken as w * h. Note that when the value is 0, it will be skipped, and if it is too small, it will be ignored in eval
    }

def parse_labels(filetxt, img_width, img_height):
    
    data = []
    
    with open(filetxt, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        
        class_id = int(line[0])
        x_center = float(line[1]) * img_width 
        y_center = float(line[2]) * img_height
        bbox_width = float(line[3]) * img_width
        bbox_height = float(line[4]) * img_height
        x_min = x_center - bbox_width / 2
        y_min = y_center - bbox_height / 2
        
        bbox = [x_min, y_min, bbox_width, bbox_height]
        
        # keypoints: x, y, v
        keypoints = []
        for i in line[5:]:
            keypoints.append(int(i) if i.isdigit() else float(i))
        indices_x = [0, 3, 6, 9, 12]
        indices_y = [1, 4, 7, 10, 13]
        for x, y in zip(indices_x, indices_y):
                keypoints[x] *= img_width
                keypoints[y] *= img_height
        
        obj = [class_id, bbox, keypoints]
        data.append(obj)
    return data

# Label ids of the dataset
category_ids = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motor": 3,
    "license_plate":4,
    "bus": 5,
    "truck": 6,
    "face": 7,
    "motor_per": 8
}

def yolo2coco(imagefolder):
    # This id will be automatically increased as we go
    annotation_id = 0
    image_id = 0
    annotations = []
    images = []
    
    for kpt_img in glob.glob(imagefolder + "*.jpg") + glob.glob(imagefolder + "*.png"):
        open_kpt_img = Image.open(kpt_img).convert("RGB")
        img_width, img_height = open_kpt_img.size
        
        # image
        file_name = os.path.basename(kpt_img)
        image = create_images_format(file_name, img_width, img_height, image_id)
        images.append(image)
        
        # label
        # labelfile = kpt_img.replace("images", "labels").rsplit('.', 1)[0] + ".txt"
        # data = parse_labels(labelfile, img_width, img_height)
        # for i,obj in enumerate(data):
        image_id += 1

    print(images)

yolo2coco('/Users/thanhhuyen/Dev/yolo2json/dataset_kpt/images/')
        
        
            
        
                
        
        
        
        
    
    


    
