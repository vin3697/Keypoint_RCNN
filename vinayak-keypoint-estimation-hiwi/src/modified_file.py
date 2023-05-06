
from data_in_tensor import *
import matplotlib.pyplot as plt

result_folder_path = "C:/Users/vinay/detection/dataset/result_imgs"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def ground_truth_img(iterator_ground_img):
    batch = next(iterator_ground_img)
    print("Original targets:\n", batch[1], "\n\n")

    #visualize the image with the data from batch!
    image_original = (batch[0][0].permute(1,2,0).numpy() * 255).astype(np.uint8)
    bboxes_original = batch[1][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()

    keypoints_original = []
    for kps in batch[1][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
        keypoints_original.append([kp[:2] for kp in kps])
    #print(keypoints_original)

    #randomly visualize a image! make shuffle false not to do random
    return visualize(image_original, bboxes_original, keypoints_original)


def predicted_img(iterator):
    images, targets = next(iterator)
    images = list(image.to(device) for image in images)
    model = get_model(num_keypoints)
    model.to(device)
    with torch.no_grad():
        model.to(device)
        model.eval()
        output = model(images)
    print("Predictions: \n", output)
    #visualize predictions
    image = (images[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
    scores = output[0]['scores'].detach().cpu().numpy()
    high_scores_idxs = np.where(scores > 0.65)[0].tolist() # Indexes of boxes with scores > 0.65
    #TODO: save the scores and IoU value
    post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)
    # NMS leaves the boxes with the highest confidence score (the best candidates) and removes other boxes, which partially overlap the best candidates. 
    # To define the degree of this overlapping, we will set the threshold for Intersection over Union (IoU) equal 0.3.
    # Below, in output[0]['keypoints'][high_scores_idxs][post_nms_idxs] and output[0]['boxes'][high_scores_idxs][post_nms_idxs]
    # Firstly, we choose only those objects, which have score above predefined threshold. This is done with choosing elements with [high_scores_idxs] indexes
    # Secondly, we choose only those objects, which are left after NMS is applied. This is done with choosing elements with [post_nms_idxs] indexes
    keypoints = []
    for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        keypoints.append([list(map(int, kp[:2])) for kp in kps])
    #print(keypoints)
    bboxes = []
    for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        bboxes.append(list(map(int, bbox.tolist())))
        
    return visualize(image, bboxes, keypoints)

def get_model(num_keypoints, weights_path=None):
    
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))

    # an instance of keypointrcnn_resnet50_fpn model from the torchvision.models.detection module
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=True,
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes = 2, # Background is the first class, object is the second class
                                                                   rpn_anchor_generator=anchor_generator)

    #if weights_path:
    #    state_dict = torch.load(weights_path)
    #    model.load_state_dict(state_dict)         
    return model

def trainModel(KEYPOINTS_FOLDER_TRAIN, KEYPOINTS_FOLDER_TEST, LOG_FOLDER, Weight_Folder, batch_size_test, batch_size_train, learning_rate, epochs, num_keypoints):

    #print(device)
    dataset_train = ClassDataset(KEYPOINTS_FOLDER_TRAIN)
    dataset_test = ClassDataset(KEYPOINTS_FOLDER_TEST)

    data_loader_train = DataLoader(dataset_train, batch_size_train, shuffle=True, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size_test, shuffle=False, collate_fn=collate_fn)
    model = get_model(num_keypoints)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    # The params variable contains a list of all model parameters that require gradients.
    optimizer = torch.optim.SGD(params, learning_rate)

    loss_list =[]
    for epoch in range(epochs):
        loss = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=1000)
        loss_list.append(loss) # append loss to the list after each epoch
        #lr_scheduler.step()
        evaluate(model, data_loader_test, device)
    # save the loss list to a file
    #with open(os.path.join(LOG_FOLDER, 'loss.json'), 'w') as f:
    #    json.dump(loss_list, f)

    # Save model weights after training
    torch.save(model.state_dict(),Weight_Folder)

    # Save the hyperparameters to a file
    #TODO : saving hyperparameters according to different batch size and learning rate.
    hyperparams = {
    "num_epochs": epochs,
    "learning_rate": learning_rate,
    "batch_size_test": batch_size_test,
    "batch_size_train" : batch_size_train,
    "optimizer": "SGD",
    #"scheduler": "StepLR",
    #"step_size": 5,
    #"gamma": 0.1,
    }
    with open(os.path.join(LOG_FOLDER, 'hyperparameters.json'), "w") as f:
        json.dump(hyperparams, f)

    #Compare the Ground truth and Predicted images : QUalitative result!

    #Predicted Image
    iterator_predicted_img  = iter(data_loader_test)
    number_of_test_images = 11
    
    #Ground Truth Images
    data_set = ClassDataset(KEYPOINTS_FOLDER_TEST)
    data_loader =DataLoader(data_set, batch_size_test, shuffle=False, collate_fn=collate_fn)
    iterator_ground_img = iter(data_loader)

    for i in range(number_of_test_images):
        predicted_image = predicted_img(iterator_predicted_img)
        ground_truth_image = ground_truth_img(iterator_ground_img)
        # create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        # plot the predicted image
        axs[0].imshow(predicted_image)
        axs[0].set_title('Predicted Image')
        
        # plot the ground truth image
        axs[1].imshow(ground_truth_image)
        axs[1].set_title('Ground Truth Image')
        #TODO club all images in single file
        #TODO : to save images add a count variable just for seek of different image names so dont get overwrite while trainModels!
        # save the figure to a folder
        fig.savefig(os.path.join(result_folder_path, f"image_{i}.png"))

        #TODO : For Quantitative results: OKS

    return None



if __name__=="__main__":

    KEYPOINTS_FOLDER_TRAIN = 'C:/Users/vinay/detection/dataset/train'
    KEYPOINTS_FOLDER_TEST = 'C:/Users/vinay/detection/dataset/test'
    LOG_FOLDER ='C:/Users/vinay/detection/dataset/log'
    Weight_Folder = 'C:/Users/vinay/detection/dataset/weights/keypointsrcnn_weights.pth'
    batch_size_train = 1
    batch_size_test = 1
    learning_rate = 0.001
    epochs = 1
    num_keypoints = 10
    #TODO: create list for batch size and learning rate and loop over them 
    trainModel(KEYPOINTS_FOLDER_TRAIN, KEYPOINTS_FOLDER_TEST, LOG_FOLDER, Weight_Folder, batch_size_test, batch_size_train, learning_rate, epochs , num_keypoints)