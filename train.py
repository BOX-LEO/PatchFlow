import torch
import torch.nn as nn
import torch.optim as optim
import time
from loss import Loss
from metrics import image_AUROC, pixel_AUROC


def train_model(model, train_loader, epoch):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criteria = Loss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    start = time.time()
    print('Training Start')
    for i in range(epoch):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            z_dist, jacobians = model(data)
            loss = criteria(z_dist, jacobians)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # TODO: Add evaluation step and pick the best model
        #  1. randomly sample image from train_loader
        #  2. add anomaly to the image
        #  3. calculate image_AUROC and pixel_AUROC
        #  4. save the model if it has the best performance


        print('Epoch: {} Loss: {:.6f}'.format(i, train_loss / len(train_loader.dataset)))
    # print training time in minutes
    print('Training time: {} minutes'.format((time.time() - start) / 60))
    return model
