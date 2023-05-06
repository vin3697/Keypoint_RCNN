

# Keypoint detection model with Pytorch

Keypoint RCNN based DLO node detection
The original implementation is [Mask R-CNN](https://arxiv.org/abs/1703.06870)


# Credit
- Keypoint RCNN tutorial: https://medium.com/@alexppppp/how-to-train-a-custom-keypoint-detection-model-with-pytorch-d9af90e111da
- Pytorch detection utils: https://github.com/pytorch/vision/tree/main/references/detection

#Bounding Boxes
The bounding boxes in KeyPoint RCNN refer to the bounding boxes that surround the whole object. The keypoints are associated with these bounding boxes and the model learns to predict the keypoints within these bounding boxes. Therefore, there is only one bounding box per object instance and it is associated with all the keypoints belonging to that object instance.
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

Anchor Gernarator is an instance from class 'torchvision' library: Generates a set of anchors,
at different scales and aspect ratios (used in region proposal network https://medium.com/egen/region-proposal-network-rpn-backbone-of-faster-r-cnn-4a744a38d7f9)
An anchor is essentially a box with a fixed shape and size that is placed at different positions over the image.
"""
"""
#ARGUMENTS
1. pretrained: If set to True, the model will be initialized with weights pre-trained on the COCO dataset. 
In this case, it is set to False, meaning we will train the model from scratch.
2. In torchvision.models.detection.keypointrcnn_resnet50_fpn, the pretrained_backbone argument allows you to specify whether or not to use a pre-trained backbone model. 
When pretrained_backbone=True, the ResNet50 backbone model is initialized with weights pre-trained on the ImageNet dataset.
This can be useful because the pre-trained model has already learned a lot of useful features that can be helpful for object detection.
5.  A function that generates a set of anchor boxes for each spatial location of the feature map produced by the Region Proposal Network (RPN) in the model. 
"""


model_and_training.py
* Batch size:
A larger batch size can speed up training since the model is updated less frequently, and the updates are made in a more accurate direction. However, larger batch sizes require more memory to store the intermediate computations, which can limit the size of the model that can be trained on a given hardware. Additionally, larger batch sizes may cause the model to converge to a less accurate solution, as the stochastic nature of the gradient descent algorithm is reduced.

On the other hand, smaller batch sizes can lead to a more accurate solution since the model is updated more frequently, but this comes at the cost of slower training due to more frequent updates. Smaller batch sizes also require less memory, which can allow for larger models to be trained on a given hardware.

* collate_fn: (feeding of batch data to the NN in structured way)

The default collate_fn in PyTorch assumes that each sample is a tuple consisting of tensors, and it concatenates the tensors along the first dimension to create the batch.

- It takes a list of samples as input
- It creates a batch by stacking the images along a new dimension (the batch dimension)
- It returns a dictionary that contains the following fields:
    - images: the batch of images
    - targets: a list of dictionaries, where each dictionary represents the bounding box annotations for the objects in the corresponding image in the batch.


* Optimizers
Stochastic Gradient Descent (SGD): torch.optim.SGD()
Adaptive Moment Estimation (Adam): torch.optim.Adam()
Adaptive Gradient Algorithm (Adagrad): torch.optim.Adagrad()
Adaptive Delta (Adadelta): torch.optim.Adadelta()
Adaptive Moment Estimation with RMSprop (RMSprop): torch.optim.RMSprop()


* train_one_epoch()
 train_one_epoch() function is used for training a model for one epoch. This is useful when you want more control over the training process, such as monitoring the loss at every batch or updating the learning rate after a certain number of iterations.