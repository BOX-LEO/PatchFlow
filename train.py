import torch
import torch.nn as nn
import torch.optim as optim
import time
from loss import Loss
from metrics import image_AUROC, pixel_AUROC
from utils import save_model


def train_model(model, train_loader, val_image_loader, val_mask_loader, epoch, data_name='mvtec', category='capsules'):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criteria = Loss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    max_pixel_auroc = 0
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
        print('Epoch: {} Loss: {:.6f}'.format(i, train_loss / len(train_loader.dataset)))

        # validation
        model.eval()
        pa = pixel_AUROC(model, val_image_loader, val_mask_loader)
        if pa > max_pixel_auroc:
            max_pixel_auroc = pa
            output = 'model_{}_{}.pt'.format(data_name, category)
            save_model(model, output)
            print('New highest pixel AUROC: {:.6f}'.format(pa))

    # print training time in minutes
    print('Training time: {} minutes'.format((time.time() - start) / 60))
    return output
