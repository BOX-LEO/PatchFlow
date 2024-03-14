import torch
import matplotlib.pyplot as plt
import os
import torch.nn as nn

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
def save_heatmap(image, heatmap, path):
    plt.imshow(image)
    plt.imshow(heatmap, alpha=0.6)
    plt.savefig(path)
    plt.close()


def export_heatmap(model, test_loader, folder):
    current_dir = os.getcwd()
    if not os.path.exists(os.path.join(current_dir, folder)):
        os.makedirs(os.path.join(current_dir, folder))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        anomaly_maps = model(data)
        for i in range(anomaly_maps.shape[0]):
            save_heatmap(data[i].cpu().permute(1, 2, 0), anomaly_maps[i].cpu(), os.path.join(current_dir, folder, str(i) + '.png'))
    print('Heatmap exported to: ', os.path.join(current_dir, folder))
