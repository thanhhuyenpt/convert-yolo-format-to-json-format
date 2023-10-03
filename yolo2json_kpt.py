import os
import json
import numpy as np
from PIL import Image

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
    bbox = data[category_id]
    

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


        
        
            
        
                
        
        
        
        
    
    


    
