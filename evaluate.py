import torch
from torchmetrics import AUROC


def image_auroc(model, test_loader):
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
    model.to(device)
    model.eval()
    pixel_auroc = AUROC(task='binary')

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        mask = next(iter(mask_loader))[0].to(device)
        anomaly_maps = model(data)
        # match the shape of anomaly_maps to mask
        # get the shape of mask
        H, W = mask.shape[-2:]
        # linearly interpolate the anomaly_maps to the shape of mask
        anomaly_maps = torch.nn.functional.interpolate(anomaly_maps, size=(H, W), mode='bilinear', align_corners=False)


        anomaly_scores= anomaly_maps.view(anomaly_maps.shape[0], -1)
        mask = mask.view(mask.shape[0], -1)
        pixel_auroc.update(anomaly_scores, mask)


    print('Pixel AUROC: {:.6f}'.format(pixel_auroc.compute()))
    return pixel_auroc.compute()


# TODO: Add AUPRO for both image and pixel level


