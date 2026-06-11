import timm
import torch
from FrEIA.framework import SequenceINN
from FrEIA.modules import AllInOneBlock
from torch import Tensor, nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
import math
from anomaly_map import AnomalyMap
from loss import Loss


class FeatureExtractor(nn.Module):
    def __init__(self, image_size=768, use_layer=(19, 26, 35)):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b5')
        self.image_size = image_size
        self.layers = use_layer

    def forward(self, x):
        x = self.feature_extractor._swish(self.feature_extractor._bn0(self.feature_extractor._conv_stem(x)))
        # Blocks
        features = []
        for idx, block in enumerate(self.feature_extractor._blocks):
            drop_connect_rate = self.feature_extractor._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.feature_extractor._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in self.layers:
                features.append(x)
            if idx == max(self.layers):
                return features


class SwinFeatureExtractor(nn.Module):
    def __init__(self, use_layer=(1, 2, 3), pre_trained=True):
        super(SwinFeatureExtractor, self).__init__()
        self.feature_extractor = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=pre_trained)
        self.layers = use_layer

    def forward(self, x):
        feature1 = self.feature_extractor.patch_embed(x)
        feature1 = self.feature_extractor.layers[0](feature1)
        feature2 = self.feature_extractor.layers[1](feature1)
        feature3 = self.feature_extractor.layers[2](feature2)
        feature4 = self.feature_extractor.layers[3](feature3)
        features = []
        if 1 in self.layers:
            feature1 = feature1.permute(0, 3, 1, 2)
            feature1 = nn.LayerNorm(feature1.shape[1:]).to('cuda')(feature1)
            features.append(feature1)
        if 2 in self.layers:
            feature2 = feature2.permute(0, 3, 1, 2)
            feature2 = nn.LayerNorm(feature2.shape[1:]).to('cuda')(feature2)
            features.append(feature2)
        if 3 in self.layers:
            feature3 = feature3.permute(0, 3, 1, 2)
            feature3 = nn.LayerNorm(feature3.shape[1:]).to('cuda')(feature3)
            features.append(feature3)
        if 4 in self.layers:
            feature4 = feature4.permute(0, 3, 1, 2)
            feature4 = nn.LayerNorm(feature4.shape[1:]).to('cuda')(feature4)
            features.append(feature4)
        return features


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, image_size=768, use_layer=(19, 26, 35), scale=3):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b5')
        self.scale = scale
        self.image_size = image_size
        self.use_layer = use_layer

    def eff_ext(self, x):
        feats = []
        x = self.feature_extractor._swish(self.feature_extractor._bn0(self.feature_extractor._conv_stem(x)))
        # Blocks
        for idx, block in enumerate(self.feature_extractor._blocks):
            drop_connect_rate = self.feature_extractor._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.feature_extractor._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in self.use_layer:
                feats.append(x)
            if idx == max(self.use_layer):
                return feats

    def forward(self, x):
        y = list()
        for s in range(self.scale):
            feat_s = F.interpolate(x, size=(self.image_size // (2 ** s), self.image_size // (2 ** s))) if s > 0 else x
            feat_s = self.eff_ext(feat_s)

            y += feat_s
        return y


class ResFeatureExtractor(nn.Module):
    def __init__(self, use_layer=(3, 4,)):
        super(ResFeatureExtractor, self).__init__()
        self.feature_extractor = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.laver_num = use_layer

    def forward(self, x):
        y = list()
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)
        x = self.feature_extractor.maxpool(x)
        x = self.feature_extractor.layer1(x)
        if 1 in self.laver_num:
            y.append(x)
        x = self.feature_extractor.layer2(x)
        if 2 in self.laver_num:
            y.append(x)
        x = self.feature_extractor.layer3(x)
        if 3 in self.laver_num:
            y.append(x)
        x = self.feature_extractor.layer4(x)
        if 4 in self.laver_num:
            y.append(x)

        return y


class ResMultiScaleFeatureExtractor(nn.Module):
    def __init__(self, layer_num=(4,), scales=3):
        super(ResMultiScaleFeatureExtractor, self).__init__()
        self.feature_extractor = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        self.laver_num = layer_num
        self.scales = scales

    def _extract(self, x):
        y = list()
        x = self.feature_extractor.backbone.conv1(x)
        x = self.feature_extractor.backbone.bn1(x)
        x = self.feature_extractor.backbone.relu(x)
        x = self.feature_extractor.backbone.maxpool(x)
        x = self.feature_extractor.backbone.layer1(x)
        if 1 in self.laver_num:
            y.append(x)
        x = self.feature_extractor.backbone.layer2(x)
        if 2 in self.laver_num:
            y.append(x)
        x = self.feature_extractor.backbone.layer3(x)
        if 3 in self.laver_num:
            y.append(x)
        x = self.feature_extractor.backbone.layer4(x)
        if 4 in self.laver_num:
            y.append(x)

        return y

    def forward(self, x):
        y = list()
        for i in range(self.scales):
            features = self._extract(x)
            y += features
            x = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        return y


class GetPatchFeature(nn.Module):
    def __init__(self, target_feature_size=(48, 48), patch_size=3):
        super(GetPatchFeature, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=patch_size, stride=1,
                                    padding=patch_size // 2)  # feature Hight and Width don't change
        self.upsample = nn.Upsample(size=target_feature_size, mode='bilinear', align_corners=True)

    def forward(self, features):
        patch_features = []
        for f in features:
            f = self.avgpool(f)
            f = self.upsample(f)
            patch_features.append(f)

        patch_features = torch.cat(patch_features, dim=1)
        return patch_features


class FeatureFuser(nn.Module):
    def __init__(self, layers=3, patch_size=3, target_feature_size=(96, 96)):
        super(FeatureFuser, self).__init__()
        self.layers = layers
        self.avgpool = nn.AvgPool2d(kernel_size=patch_size, stride=1,
                                    padding=patch_size // 2)  # feature size dont change
        self.upsample = nn.Upsample(size=target_feature_size, mode='bilinear', align_corners=True)

    def forward(self, features):
        patch_features = []
        for i, f in enumerate(features):
            n = i % self.layers
            f = self.avgpool(f)
            f = self.upsample(f)
            if i // self.layers > 0:
                patch_features[n] += f
            else:
                patch_features.append(f)

        patch_features = torch.cat(patch_features, dim=1)
        return patch_features


class FeatureAdaptor(nn.Module):
    def __init__(self, dim_in=608, dim_out=512):
        super(FeatureAdaptor, self).__init__()
        self.dim_out = dim_out
        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=1)
        # self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_si/

    def forward(self, x):
        x = self.conv1(x)

        return x


def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P


class Flow(nn.Module):

    def __init__(self, input_dims=(1024, 48, 48,), n_blocks=8):
        super(Flow, self).__init__()
        self.flow = SequenceINN(*input_dims)

        def subnet_fc(dims_in, dims_out):
            '''Return a feed-forward subnetwork, to be used in the coupling blocks below'''
            return nn.Sequential(nn.Conv2d(dims_in, dims_out, kernel_size=1, padding='same'), nn.BatchNorm2d(dims_out),
                                 nn.LeakyReLU(0.2), )

        for k in range(n_blocks):
            self.flow.append(AllInOneBlock, subnet_constructor=subnet_fc)

    def forward(self, x, rev=False):
        if not rev:
            return self.flow(x)
        else:
            return self.flow(x, rev=True)


class FlowBottleneck(nn.Module):

    def __init__(self, input_dims=(1024, 48, 48,), n_blocks=8, hidden_dim=256):
        super(FlowBottleneck, self).__init__()
        self.flow = SequenceINN(*input_dims)
        self.hidden_dim = hidden_dim

        def subnet_fc(dims_in, dims_out):
            '''Return a feed-forward subnetwork, to be used in the coupling blocks below'''
            return nn.Sequential(nn.Conv2d(dims_in, self.hidden_dim, kernel_size=3, padding='same'),
                                 # nn.BatchNorm2d(self.hidden_dim),
                                 # nn.LeakyReLU(0.2),
                                 # nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding='same'),
                                 # nn.BatchNorm2d(self.hidden_dim),
                                 # nn.LeakyReLU(0.2),
                                 nn.Conv2d(self.hidden_dim, dims_out, kernel_size=3, padding='same'),
                                 nn.BatchNorm2d(dims_out),
                                 nn.LeakyReLU(0.2), )

        for k in range(n_blocks):
            self.flow.append(AllInOneBlock, subnet_constructor=subnet_fc)

    def forward(self, x, rev=False):
        if not rev:
            return self.flow(x)
        else:
            return self.flow(x, rev=True)


class PatchflowModel(nn.Module):
    """PatchflowModel.

    Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows.

    Args:
        input_size (tuple[int, int]): Model input size.
        backbone (str): Backbone CNN network
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
        flow_steps (int, optional): Flow steps.
        hidden_ratio (float, optional): Ratio to calculate hidden var channels. Defaults to 1.0.

    Raises:
        ValueError: When the backbone is not supported.
    """

    def __init__(
            self,
            input_size,  # image size HxW
            backbone: str = 'efficientnet-b5',
            pre_trained: bool = True,
            flow_steps: int = 1,
            flow_feature_dim: int = 128,
            scale: int = 3,
            crop_size=(672, 672),

    ) -> None:
        super().__init__()
        self.input_size = input_size
        im_size = crop_size[0] if crop_size else input_size[0]
        self.feature_extractor = MultiScaleFeatureExtractor(image_size=im_size, use_layer=(12, 19, 35), scale=scale)
        # self.feature_extractor = ResFeatureExtractor(use_layer=(2,3))
        # self.feature_extractor = ResMultiScaleFeatureExtractor(layer_num=(3,4), scales=2)
        # self.feature_extractor = SwinFeatureExtractor(use_layer=(2,3,4), pre_trained=True)
        self.patcher = GetPatchFeature(target_feature_size=(im_size // 8, im_size // 8), patch_size=3)
        # self.patcher = FeatureFuser(layers=3, patch_size=3, target_feature_size=(96, 96))
        self.feature_adaptor = FeatureAdaptor(dim_in=496 * scale, dim_out=flow_feature_dim)
        self.graph = FlowBottleneck(input_dims=[flow_feature_dim, im_size // 8, im_size // 8], n_blocks=flow_steps,
                                    hidden_dim=128)

        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False  # freeze the feature extractor
        #
        self.anomaly_map_generator = AnomalyMap(input_size=input_size, crop_size=crop_size)

        # print the number of parameters and trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        print('Total parameters:', total_params)
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Total trainable parameters:', total_trainable_params)

    #
    def forward(self, input_tensor: Tensor, rev=False):
        """Forward-Pass the input to the FastFlow Model.

        Args:
            input_tensor (Tensor): Input tensor.

        Returns:
            Tensor | list[Tensor] | tuple[list[Tensor]]: During training, return
                (hidden_variables, log-of-the-jacobian-determinants).
                During the validation/test, return the anomaly map.
        """
        if not rev:

            self.feature_extractor.eval()

            # if isinstance(self.feature_extractor, SwinTransformer):
            #     features = self._get_swin_feature(input_tensor)
            features = self.feature_extractor(input_tensor)
            features = self.patcher(features)
            features = self.feature_adaptor(features)
            hidden_variables, log_jacobians = self.graph(features)

            return_val = (hidden_variables, log_jacobians)

            if not self.training:
                return_val = self.anomaly_map_generator(hidden_variables)
                # save the anomaly map as row data
                # anomaly_map = return_val.cpu().detach().numpy()
                # anomaly_map = np.squeeze(anomaly_map)

            return return_val
        else:

            self.feature_extractor.eval()

            feature_inv, jac_inv = self.graph(input_tensor, rev=True)

            return_val = (feature_inv, jac_inv)

            return return_val


if __name__ == '__main__':
    input_batch = torch.rand(4, 3, 768, 768)
    # map to[0,1]
    input_batch = input_batch - torch.min(input_batch)
    # normalize with imagenet stats
    input_batch[:, 0, :, :] = (input_batch[:, 0, :, :] - 0.485) / 0.229
    input_batch[:, 1, :, :] = (input_batch[:, 1, :, :] - 0.456) / 0.224
    input_batch[:, 2, :, :] = (input_batch[:, 2, :, :] - 0.406) / 0.225

    print(torch.max(input_batch), torch.min(input_batch))
    model = PatchflowModel(input_size=(768, 768), pre_trained=True, flow_steps=1)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)
    # model.eval()
    # feature1, feature2, feature3 = FeatureExtractor()(input_batch)
    # print(torch.max(feature1), torch.max(feature2), torch.max(feature3))
    # print(torch.min(feature1), torch.min(feature2), torch.min(feature3))
    #
    output, jac = model(input_batch)
    print(output.shape)
    print(jac.shape)
    print(type(output))
    # inv_output, inv_jac = model(output, rev=True)
    # print(torch.max(inv_output[0] - feature1))
    # print(torch.max(inv_output[1] - feature2))
    # print(torch.max(inv_output[2] - feature3))

    # print('feature span')
    # print(torch.max(feature1), torch.max(feature2), torch.max(feature3))
    # print(torch.min(feature1), torch.min(feature2), torch.min(feature3))

    # feature1 = torch.rand(4,192,  96, 96)
    # feature2 = torch.rand(4, 384, 48, 48)
    # feature3 = torch.rand(4, 768, 24, 24)

    # flow = CrossScaleFlow([3, 384, 384], 8, 2.0, 1024)
    # output, jac = flow([feature1, feature2, feature3])
    # print(torch.max(output[0]))
    # print(torch.min(output[0]))
    # print(torch.max(output[1]))
    # print(torch.min(outpput[2]))
    #     # print('#################')
    #     # inv_output, jac = flow(output, rev=True)
    #     # print(torch.max(feature1))
    #     # print(torch.min(feature2))
    #     # print(torch.max(inv_output[0] - feature1))
    #     # print(torch.max(inv_output[1] - feature2))
    #     # print(torch.max(inv_output[2] - feature3))ut[1]))
    # print(torch.max(output[2]))
    # print(torch.min(out

    loss = Loss()
    print(loss(output, jac))

    # # print swin_large
    # feature_extractor = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=True)
    # print(feature_extractor)
