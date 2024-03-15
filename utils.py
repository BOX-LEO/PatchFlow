import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
# save the model
def save_model(model, filename):
    # ex. filename = 'visa_capsules.pt'
    cwd = os.getcwd()
    model_path = os.path.join(cwd, 'models')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    filename = os.path.join(model_path, filename)
    torch.save(model.state_dict(), filename)


# load the model
def load_model(model, filename):
    # ex. filename = 'visa_capsules.pt'
    if type(model) != nn.DataParallel:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model_path = os.path.join(os.getcwd(), 'models', filename)

    model.load_state_dict(torch.load(model_path))
    return model



# save heatmap on top of the image
def save_heatmap(image, anomaly_map, path,):
    # image: numpy array, shape (H, W, C)
    # heatmap: numpy array, shape (H, W)

    # add heatmap by 1 and change to heatmap
    anomaly_map = anomaly_map + 1
    # print max and min of heatmap

    heatmap = cv2.applyColorMap(np.uint8(255 * anomaly_map), cv2.COLORMAP_JET)

    # convert image to uint8
    image = np.uint8(255 * image)
    # change to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # add heatmap to the image
    image = cv2.addWeighted(heatmap, 0.2, image, 0.8, 0)

    # save the image
    cv2.imwrite(path, image)



def export_heatmap(model, test_loader,eval_loader, folder):
    current_dir = os.getcwd()
    if not os.path.exists(os.path.join(current_dir, 'result')):
        os.makedirs(os.path.join(current_dir, 'result'))
    if not os.path.exists(os.path.join(current_dir,'result', folder)):
        os.makedirs(os.path.join(current_dir,'result', folder))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    j=0
    for (test_image,_),(og_image,_) in zip(test_loader,eval_loader):
        # data, target = data.to(device), target.to(device)
        anomaly_maps = model(test_image)
        for i in range(anomaly_maps.shape[0]):
            # reshape image to (H, W, C)
            save_heatmap(og_image[i].permute(1, 2, 0).cpu().detach().numpy(), anomaly_maps[i].squeeze().cpu().detach().numpy(), os.path.join(current_dir, 'result', folder, str(j) + '.png'))
            j+=1
    print('Heatmap exported to: ', os.path.join(current_dir,'result', folder))


