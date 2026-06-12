from utils import *
from train import train_model
from model import PatchflowModel
from data import get_dataset, get_mask_dataset
from metrics import image_AUROC, pixel_AUROC


input_size = (768, 768)

datapath = '/home/aivision/Documents/PatchFlow-main/'
category = 'bottle'
data_name = 'mvtec'
batch_size = 8
epoch = 1

# training images + image-level test images
train_loader, test_loader = get_dataset(datapath, category, batch_size=batch_size, data_name=data_name)
# ground-truth masks + the matching anomalous test images (used for validation)
mask_loader, mask_image_loader = get_mask_dataset(datapath, category, batch_size=batch_size, data_name=data_name)

model = PatchflowModel(input_size, scale=3, flow_feature_dim=128, flow_steps=1)

# train_model validates each epoch and saves the best checkpoint, returning its path
best_model_path = train_model(model, train_loader, mask_image_loader, mask_loader, epoch,
                              data_name=data_name, category=category)
print('Best model saved to:', best_model_path)

# Evaluate the best model
model = load_model(model, best_model_path)
image_AUROC(model, test_loader)
pixel_AUROC(model, mask_image_loader, mask_loader)
