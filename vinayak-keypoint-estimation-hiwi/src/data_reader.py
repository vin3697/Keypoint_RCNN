import json
import cv2
import numpy as np
from operator import itemgetter
import os

class datareader:
    def __init__(self, PATH_TO_DATAFILE, PATH_TO_RAW_DATA):

        with open(PATH_TO_DATAFILE, "r") as f:
            self.mydata = json.load(f)

    def __getitem__(self, indx: int):
        """
        Get item of index and return img and annotation

        Args:
            indx (int): ID of Image as in labelstudio
        """
        data_requested = self.mydata[indx]
        data_file_upload = data_requested["file_upload"].split("-")[1]
        img = cv2.imread(PATH_TO_RAW_DATA + "/" + data_file_upload)

        #grabing the bounding box and keypoint from json file
        annotations_all = data_requested["annotations"][0]["result"]
        returnAnno = []
        returnBox = []
        for anno in annotations_all:
            if anno["type"] == "keypointlabels":
                returnAnno.append(anno)
            if anno["type"] == "rectanglelabels":
                returnBox.append(anno)
        return img, returnAnno, returnBox ,data_file_upload

    #get the keypoints and store them for visualizing purpose
    def annotateImg(self, img, annotations):
        
        keypoints_name = []
        for anno in annotations:
            x_percentage = anno["value"]["x"] / 100.0
            y_percentage = anno["value"]["y"] / 100.0
            width = anno["original_width"]
            height = anno["original_height"]
            x_center = int(x_percentage * width)
            y_center = int(y_percentage * height)
            label = anno["value"]["keypointlabels"][0]

            #added "if" statment to make the sorting easy!
            if label == "node10":
                label = "node91"

            cv2.putText(img, label, (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            img = cv2.circle(img, (x_center, y_center), 10, [0, 0, 255], -1)
            #print("Annotating img .. {} {}".format(x_center, y_center))
            #grabing all keypoints and then sorting out according to the node value
            keypoints_name.append([x_center,y_center,label])
        return img , keypoints_name
    

    #getting the bounding box of ROI
    def BBox(self, img, box):
        bbox = []
        for anno in box:
            
            x_percentage = anno["value"]["x"] /100
            y_percentage = anno["value"]["y"] /100
            width_percentage = anno["value"]["width"] /100
            height_percentage = anno["value"]["height"] /100

            img_width = anno["original_width"]
            img_height = anno["original_height"]            
 
            x_top_corner = round(x_percentage*img_width)
            y_top_corner = round(y_percentage*img_height)
            width    = round(width_percentage*img_width)
            height   = round(height_percentage*img_height)
            top_left_corner = (x_top_corner, y_top_corner )
            bottom_right_corner = (x_top_corner + width, y_top_corner + height)
            #img = cv2.circle(img, (x_center, y_center), 10, [0, 0, 255], -1) 

            img = cv2.rectangle(img, bottom_right_corner, top_left_corner, (0, 255, 0), 3)
            bbox.append([top_left_corner[0],top_left_corner[1],bottom_right_corner[0],bottom_right_corner[1]])
        #get the edges correctly and return it back to main fucntion!
        return img , bbox
    


if __name__ == "__main__":
    PATH_TO_DATAFILE = "C:/Users/vinay/detection/dataset/anno_all_img.json"
    PATH_TO_RAW_DATA = "C:/Users/vinay/detection/dataset/images"
    dataR = datareader(PATH_TO_DATAFILE, PATH_TO_RAW_DATA)


    # 56 is the numbers of total images (train+test dataset!)
    for i in range(56):
   
        img, annotation, box ,data_file_upload = dataR[i]
    
        #call the anotateImg fucntion
        img ,keypoints_name = dataR.annotateImg(img, annotation)

        #sorting keypoints according to the name
        sorted_keypoints_name = sorted(keypoints_name, key= itemgetter(2))
        print("Sorted keypoints:", sorted_keypoints_name)
        for kp in keypoints_name:
            kp[2] = 1
        print("Keypoints in format:: x_c, y_c , visibility", keypoints_name)

        img, bbox = dataR.BBox(img, box)
        annotation_file_path = 'C:/Users/vinay/detection/dataset/train/annotations'
        data_file_upload = data_file_upload.split(".")[0]
        annotation_file = os.path.join(annotation_file_path,data_file_upload+'.json')
        annotations = {}
        annotations['bboxes'], annotations['keypoints'] = bbox, sorted_keypoints_name
        with open(annotation_file,"w") as file_data:
            json.dump(annotations, file_data)

        #print("bounding box for img:: top left corner: x1, y1 || bottom right corner: x2, y2", bbox)
        # display the final image!
        cv2.imshow("imreader", img)
        cv2.waitKey(0)
