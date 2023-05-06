# keypointrcnn model from pytorch!

import os, json, cv2, numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

# https://github.com/pytorch/vision/tree/main/references/detection
import sys
sys.path.append('C:/Users/vinay/detection/lib')
import transforms, utils, engine, train
from utils import collate_fn
from engine import train_one_epoch, evaluate


keypoints_classes_ids2names = {0:"node1", 1:"node2", 2:"node3", 3:"node4", 4:"node5", 5:"node6",
    6:"node7", 7:"node8", 8:"node9", 9:"node91"}

class ClassDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs_files = sorted(os.listdir(os.path.join(root, "images")))
        self.annotations_files = sorted(os.listdir(os.path.join(root, "annotations")))

    def __getitem__(self, idx: int):
        
       
        img_path = os.path.join(self.root, "images", self.imgs_files[idx])
        annotations_path = os.path.join(self.root, "annotations", self.annotations_files[idx])

        img_original = cv2.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB) 

        with open(annotations_path) as f:
            data = json.load(f)
            bboxes_original = data['bboxes']
            keypoints_original = data['keypoints']
            #object is wire harness
            bboxes_labels_original = ['wire harness' for _ in bboxes_original]            

        #storing the data in tensor
        bboxes_original = torch.as_tensor(bboxes_original, dtype=torch.float32)
        target_original = {}
        target_original["boxes"] = bboxes_original
        target_original["labels"] = torch.as_tensor([1 for _ in bboxes_original] , dtype=torch.int64)
        target_original["image_id"] = torch.tensor([idx])
        target_original["area"] = (bboxes_original[:, 3] - bboxes_original[:, 1]) * (bboxes_original[:, 2] - bboxes_original[:, 0])
        target_original["iscrowd"] = torch.zeros(len(bboxes_original), dtype=torch.int64)
        target_original["keypoints"] = torch.as_tensor([keypoints_original], dtype=torch.float32)        
        img_original = F.to_tensor(img_original)


        return img_original, target_original

    def __len__(self):
        return len(self.imgs_files)


def visualize(image_original=None, bboxes_original=None, keypoints_original=None, text_option=True):
    
    #visualizing the data from tensor type
    image_copy = image_original.copy()
    
    for bbox in bboxes_original:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image_original = cv2.rectangle(image_copy, start_point, end_point, (0,255,0), 2)
    
    for kps in keypoints_original:
        for idx, kp in enumerate(kps):
            image_original = cv2.circle(image_original, tuple(kp), 1, (255,0,0), 10)
            if text_option:
                image_original = cv2.putText(image_copy, " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3, cv2.LINE_AA)
    
    #cv2.imshow("reader", image_original)
    #wait_key = cv2.waitKey(0)
    #if wait_key == 27:
    #    cv2.destroyAllWindows()
    return image_original
    



if __name__ == "__main__":
    KEYPOINTS_FOLDER_TRAIN = 'C:/Users/vinay/detection/dataset/train'
    dataset = ClassDataset(KEYPOINTS_FOLDER_TRAIN)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    iterator = iter(data_loader)
    batch = next(iterator)
    print("Original targets:\n", batch[1], "\n\n")


    #visualize the image with the data from batch!
    image_original = (batch[0][0].permute(1,2,0).numpy() * 255).astype(np.uint8)
    bboxes_original = batch[1][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()

    keypoints_original = []
    for kps in batch[1][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
        keypoints_original.append([kp[:2] for kp in kps])
    #print(keypoints_original)

    #randomly visualize a image! make shuffle false not to do random
    visualize(image_original, bboxes_original, keypoints_original)
