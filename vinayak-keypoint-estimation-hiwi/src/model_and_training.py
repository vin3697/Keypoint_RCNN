# Batch size learning rate whatever loss function etc. Hyperparams


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
from data_in_tensor import *

# Keypoint RCNN Model which returns an instance of a keypoint detection model based on the ResNet-50 FPN architecture
def get_model(num_keypoints, weights_path=None):
    

    """ 
    Anchor Gernarator is an instance from class 'torchvision' library: Generates a set of anchors,
    at different scales and aspect ratios (used in region proposal network https://medium.com/egen/region-proposal-network-rpn-backbone-of-faster-r-cnn-4a744a38d7f9)
    An anchor is essentially a box with a fixed shape and size that is placed at different positions over the image.
    """
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))

    # an instance of keypointrcnn_resnet50_fpn model from the torchvision.models.detection module
    """
    #ARGUMENTS

    1. pretrained: If set to True, the model will be initialized with weights pre-trained on the COCO dataset. 
    In this case, it is set to False, meaning we will train the model from scratch.
    2. In torchvision.models.detection.keypointrcnn_resnet50_fpn, the pretrained_backbone argument allows you to specify whether or not to use a pre-trained backbone model. 
    When pretrained_backbone=True, the ResNet50 backbone model is initialized with weights pre-trained on the ImageNet dataset.
    This can be useful because the pre-trained model has already learned a lot of useful features that can be helpful for object detection.
    5.  A function that generates a set of anchor boxes for each spatial location of the feature map produced by the Region Proposal Network (RPN) in the model. 
    """
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=True,
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes = 2, # Background is the first class, object is the second class
                                                                   rpn_anchor_generator=anchor_generator)

    #if weights_path:
    #    state_dict = torch.load(weights_path)
    #    model.load_state_dict(state_dict)         
    return model

if __name__=="__main__":


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    KEYPOINTS_FOLDER_TRAIN = 'C:/Users/vinay/detection/dataset/train'
    KEYPOINTS_FOLDER_TEST = 'C:/Users/vinay/detection/dataset/test'

    dataset_train = ClassDataset(KEYPOINTS_FOLDER_TRAIN)
    dataset_test = ClassDataset(KEYPOINTS_FOLDER_TEST)

    #more description about batch_size and collate_fn is in README.md
    data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = get_model(num_keypoints = 10)
    model.to(device)


    # creates a list of parameters that require gradients. The model.parameters() method returns an iterator over all the parameters in the model,
    # and p.requires_grad is a boolean flag that indicates whether a parameter requires gradients or not.
    params = [p for p in model.parameters() if p.requires_grad]
   
    optimizer = torch.optim.SGD(params, lr=0.001)
    #optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.3)

    num_epochs = 8

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=1000)
        #lr_scheduler.step()
        evaluate(model, data_loader_test, device)
        
    # Save model weights after training
    torch.save(model.state_dict(), 'C:/Users/vinay/detection/dataset/weights/keypointsrcnn_weights.pth')



    #Visualizing model predictions
    iterator = iter(data_loader_test)
    
    #
    number_of_test_images = 11
    for i in range(number_of_test_images):
        images, targets = next(iterator)
        images = list(image.to(device) for image in images)

        with torch.no_grad():
            model.to(device)
            model.eval()
            output = model(images)

        print("Predictions: \n", output)

        #visualize predictions
        image = (images[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
        scores = output[0]['scores'].detach().cpu().numpy()

        high_scores_idxs = np.where(scores > 0.65)[0].tolist() # Indexes of boxes with scores > 0.7
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)

        # Below, in output[0]['keypoints'][high_scores_idxs][post_nms_idxs] and output[0]['boxes'][high_scores_idxs][post_nms_idxs]
        # Firstly, we choose only those objects, which have score above predefined threshold. This is done with choosing elements with [high_scores_idxs] indexes
        # Secondly, we choose only those objects, which are left after NMS is applied. This is done with choosing elements with [post_nms_idxs] indexes

        keypoints = []
        for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            keypoints.append([list(map(int, kp[:2])) for kp in kps])

        print(keypoints)
        bboxes = []
        for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            bboxes.append(list(map(int, bbox.tolist())))
            
        visualize(image, bboxes, keypoints)
