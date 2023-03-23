import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss


class PW_ATM(nn.Module):
    def __init__(self, bn_decay=None):
        super(PW_ATM, self).__init__()
        self.conv0 = nn.Conv2d(1, 64, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, point_cloud, is_training=True):
        # point_cloud: [B, C, N]
        batch_size = point_cloud.size(0)
        num_point = point_cloud.size(2)
        point_cloud = point_cloud.permute(0, 2, 1)  # point_cloud: [B, N, C]

        # Extract z-coordinates
        point_cloud = point_cloud[:, :, 2].unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N]

        net = self.relu(self.bn0(self.conv0(point_cloud)))  # [B, 64, 1, N]
        net = self.relu(self.bn1(self.conv1(net)))  # [B, 256, 1, N]

        # Global feature vector
        global_feature = net.max(dim=-1, keepdim=True)[0]  # [B, 256, 1, 1]
        global_feature = global_feature.repeat(1, 1, 1, num_point)  # [B, 256, 1, N]

        # Concatenate with local features
        net = torch.cat([net, global_feature], dim=1)  # [B, 512, 1, N]

        net = self.relu(self.bn2(self.conv2(net)))   # [B, 128, 1, N]
        net = self.sigmoid(self.bn3(self.conv3(net)))  # [B, 1, 1, N]

        transform = net.squeeze(dim=-1).squeeze(dim=-1)  # [B, 1, 1, N]
        transform = transform.permute(0, 3, 2, 1).squeeze(-1)
        print(transform.size())

        return transform  # transform [B,N,1] 比例系数


if __name__ == '__main__':
    import torch

    # model = get_model(13)
    model = PW_ATM()
    xyz = torch.rand(6, 9, 1024)
    (model(xyz))
