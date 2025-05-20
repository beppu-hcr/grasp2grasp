import torch
import torch.nn as nn

from network.pvcnn.models.utils import create_mlp_components, create_pointnet_components

__all__ = ['PVCNN']


class PVCNN(nn.Module):
    blocks = ((64, 1, 32), (64, 2, 16), (128, 1, 16), (1024, 1, None))

    def __init__(self, extra_feature_channels=0, width_multiplier=1.0, voxel_resolution_multiplier=1):
        super().__init__()
        self.in_channels = extra_feature_channels + 3

        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks, in_channels=self.in_channels, with_se=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.point_features = nn.ModuleList(layers)

        layers, channels_cloud = create_mlp_components(
            in_channels=channels_point, out_channels=[256, 128],
            classifier=False, dim=1, width_multiplier=width_multiplier)
        self.cloud_features = nn.Sequential(*layers)

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['features']

        coords = inputs[:, :3, :]
        out_features_list = []
        for i in range(len(self.point_features)):
            inputs, _ = self.point_features[i]((inputs, coords))
            out_features_list.append(inputs)
        # inputs: num_batches * 1024 * num_points -> num_batches * 1024 -> num_batches * 128
        inputs = self.cloud_features(inputs.max(dim=-1, keepdim=False).values)
        out_features_list.append(inputs.unsqueeze(-1).repeat([1, 1, coords.size(-1)]))
        out_features = torch.cat(out_features_list, dim=1)
        out_features = torch.max(out_features, dim=-1)[0]
        out_features = out_features.view(out_features.size(0), -1)
        return out_features, None, None

if __name__ == '__main__':
    points = torch.randn(2, 3, 2012).cuda()
    print(points.size())
    pvcnn = PVCNN(extra_feature_channels=0).cuda()
    print('params {}M'.format(sum(p.numel() for p in pvcnn.parameters()) / 1000000.0))
    out, _, _ = pvcnn(points)
    print(out.size())
