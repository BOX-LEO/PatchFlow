from utils import *
from model import PatchflowModel
from data import get_dataset, get_mask_dataset, get_validation_dataset, get_evaluation_dataset
from metrics import image_AUROC, pixel_AUROC, get_mask_anomalymap, pixel_AUPRO

input_size = (768, 768)
eval_bs = 16  # batch size

datapath = '/home/dao2/defect_detection/VisA/visa_pytorch'
data_name = 'visa'
category = 'capsules'
for data_name in ['visa']:
    if data_name == 'mvtec':
        datapath = '/home/dao2/defect_detection/mvtec_anomaly_detection_v'
        categories = ['screw', 'bottle', 'metal_nut', 'pill', 'toothbrush', 'transistor', 'zipper', 'carpet', 'grid',
                      'leather', 'tile', 'wood', 'cable', 'capsule', 'hazelnut']
    elif data_name == 'visa':
        datapath = '/home/dao2/defect_detection/VisA/visa_pytorch'
        categories = os.listdir(datapath)
        categories= ['macaroni2']
    model = PatchflowModel(input_size, scale=3, flow_feature_dim=128, flow_steps=1)
    for category in categories:



        best_model_path = 'model_{}_{}.pt'.format(data_name, category)

        # load the model
        model = load_model(model, best_model_path)


        # evaluate the model
        print('Evaluating on {} dataset, category: {}'.format(data_name, category))

        # get image level AUROC
        _, test_loader = get_dataset(datapath, category, batch_size=eval_bs, data_name=data_name)
        image_AUROC(model, test_loader)
        del test_loader

        # get pixel level AUROC
        mask_loader, mask_image_loader = get_mask_dataset(datapath, category, batch_size=eval_bs, data_name=data_name)
        pixel_AUROC(model, mask_image_loader, mask_loader)
        del mask_loader


        path = '{}_{}'.format(data_name, category)
        # visualize the anomaly map
        og_image_loader = get_evaluation_dataset(datapath, category, batch_size=eval_bs, data_name=data_name)
        export_heatmap(model,mask_image_loader,og_image_loader,path)


# # get pixel level AUPRO
# # get all masks in a list
# mask_list = []
# for mask, _ in mask_loader:
#     mask_list.append(mask)
# mask = torch.cat(mask_list, dim=0)
# print(mask.shape)
# mask,anomaly_map = get_mask_anomalymap(model,mask_loader, mask_image_loader)
# mask = mask.cpu().detach().numpy()
# anomaly_map = anomaly_map.cpu().detach().numpy()
# pixel_AUPRO(anomaly_map,mask)