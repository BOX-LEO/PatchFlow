from utils import *
from train import train_model
from model import PatchflowModel
from data import get_dataset, get_dataloader
from evaluate import image_auroc, pixel_AUROC


input_size = (768, 768)

datapath = '/home/dao2/defect_detection/VisA/visa_pytorch'
category = 'capsules'

train_dataset, test_dataset, mask_dataset = get_dataset(datapath,category)
train_loader, test_loader, mask_loader = get_dataloader(train_dataset, test_dataset, batch_size=16)

model = PatchflowModel(input_size, scale=3, flow_feature_dim=128, flow_steps=1)
#
# model = train_model(model, train_loader, 10)
# save_model(model, 'patchflow.pt')

model = load_model(model, 'patchflow.pt')


image_auroc(model, test_loader)

