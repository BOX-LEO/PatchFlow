import torch
import numpy as np
from torchmetrics import AUROC
from skimage.measure import label, regionprops
from sklearn.metrics import auc
import torch.nn as nn
import time


def image_AUROC(model, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    im_auroc = AUROC(task='binary')

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        anomaly_maps = model(data)
        # reshape the anomaly_maps and get the maximum value
        anomaly_maps = anomaly_maps.view(anomaly_maps.shape[0], -1)
        anomaly_scores = torch.max(anomaly_maps, dim=-1).values
        print(anomaly_scores)
        print(target)
        im_auroc.update(anomaly_scores, target)

    print('Image AUROC: {:.6f}'.format(im_auroc.compute()))
    return im_auroc.compute()



def pixel_AUROC(model, test_loader, mask_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    pixel_auroc = AUROC(task='binary')

    for (data, target),(mask,_) in zip(test_loader,mask_loader):
        # get the data with target == 1


        data, target = data.to(device), target.to(device)
        mask = mask.to(device)
        anomaly_maps = model(data)
        # match the shape of anomaly_maps to mask
        # get the shape of mask
        H, W = mask.shape[-2:]
        # linearly interpolate the anomaly_maps to the shape of mask
        anomaly_maps = torch.nn.functional.interpolate(anomaly_maps, size=(H, W), mode='bilinear', align_corners=False)

        #turn the anomaly_maps and mask into 1D tensor
        # mask = mask.view(-1)
        # anomaly_maps = anomaly_maps.view(-1)

        print(mask.shape)
        print(anomaly_maps.shape)

        # anomaly_maps= anomaly_maps.view(anomaly_maps.shape[0], -1)
        # mask = mask.view(mask.shape[0], -1)
        pixel_auroc.update(anomaly_maps, mask)


    print('Pixel AUROC: {:.6f}'.format(pixel_auroc.compute()))
    # return pixel_auroc.compute().item()





def get_mask_anomalymap(model,mask_loader, mask_image_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    mask_list = []
    for mask, _ in mask_loader:
        mask_list.append(mask)
    mask_list = torch.cat(mask_list, dim=0)
    # squeeze the mask_list
    mask_list = mask_list.squeeze(1)
    # mask datatype to bool
    mask_list = mask_list.bool()
    print(mask_list.shape)

    H, W = mask.shape[-2:]
    anomaly_map_list = []
    for images, _ in mask_image_loader:
        anomaly_map = model(images.to(device))
        anomaly_map = torch.nn.functional.interpolate(anomaly_map, size=(H, W), mode='bilinear', align_corners=False)
        anomaly_map_list.append(anomaly_map)
    anomaly_map_list = torch.cat(anomaly_map_list, dim=0)
    anomaly_map_list = anomaly_map_list.squeeze(1)
    print(mask_list.shape)
    return mask_list, anomaly_map_list



def pixel_AUPRO(anomaly_maps, mask):
    # calculate segmentation AUPRO
    # from https://github.com/YoungGod/DFR:
    def rescale(x):
        return (x - x.min()) / (x.max() - x.min())

    max_step = 1000
    expect_fpr = 0.3  # default 30%
    max_th = anomaly_maps.max()
    min_th = anomaly_maps.min()
    delta = (max_th - min_th) / max_step
    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(anomaly_maps, dtype=bool)

    for step in range(max_step):
        start = time.time()
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[anomaly_maps <= thred] = 0
        binary_score_maps[anomaly_maps > thred] = 1
        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):  # for i th image
            # pro (per region level)
            label_map = label(mask[i], connectivity=2)
            props = regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox  # find the bounding box of an anomaly region
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = mask[i][x_min:x_max, y_min:y_max]   # bug!
                cropped_mask = prop.filled_image  # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], mask[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], mask[i]).astype(np.float32).sum()
            if mask[i].any() > 0:  # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        # print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        gt_masks_neg = ~mask
        fpr = np.logical_and(gt_masks_neg, binary_score_maps).sum() / gt_masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)
        print('Step: {} Time: {:.6f}'.format(step, time.time() - start))
    # as array
    # threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    # pros_std = np.array(pros_std)
    fprs = np.array(fprs)
    # ious_mean = np.array(ious_mean)
    # ious_std = np.array(ious_std)
    # # best per image iou
    # best_miou = ious_mean.max()
    # print(f"Best IOU: {best_miou:.4f}")
    # default 30% fpr vs pro, pro_auc
    idx = fprs <= expect_fpr  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    seg_pro_auc = auc(fprs_selected, pros_mean_selected)
    print('Pixel AUROC: {:.6f}'.format(seg_pro_auc))
#


