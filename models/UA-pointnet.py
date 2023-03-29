import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        self.PA = PW_ATM()  # 数据对齐
        self.FG = SetAbstraction()  # 提取器
        self.F1 = FeatureProgation(num_classes)  # 分类器F1
        self.F2 = FeatureProgation(num_classes)  # 分类器F2

    def forward(self, Source_xyz, Target_xyz):
        assert Source_xyz.shape == Target_xyz.shape

        # 处理Source_xyz
        l0_points_1_S, l4_points_S = self.FG(Source_xyz)
        F1_pred_S = self.F1(l0_points_1_S, l4_points_S)
        F2_pred_S = self.F2(l0_points_1_S, l4_points_S)

        # 处理Target_xyz
        transform = self.PA(Target_xyz)
        Target_xyz[:, 2, :] = Target_xyz[:, 2, :] * transform[:, :, 0]
        l0_points_1_T, l4_points_T = self.FG(Source_xyz)
        F1_pred_T = self.F1(l0_points_1_T, l4_points_T)
        F2_pred_T = self.F2(l0_points_1_T, l4_points_T)

        return F1_pred_S, F2_pred_S, F1_pred_T, F2_pred_T  # 用来计算CE1和CE2 # 用来计算ADV


class SetAbstraction(nn.Module):
    def __init__(self):
        super(SetAbstraction, self).__init__()
        # 提取器
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        # 提取
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l3_points_1 = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points_1 = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points_1)
        l1_points_1 = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points_1)
        l0_points_1 = self.fp1(l0_xyz, l1_xyz, None, l1_points_1)
        return l0_points_1, l4_points


class FeatureProgation(nn.Module):
    def __init__(self, num_classes):
        super(FeatureProgation, self).__init__()
        # 分类器

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, l0_points_1, l4_points):
        x_1 = self.drop1(F.relu(self.bn1(self.conv1(l0_points_1))))
        x_1 = F.log_softmax(self.conv2(x_1), dim=1)
        x_1 = x_1.permute(0, 2, 1)

        return x_1, l4_points


class PW_ATM(nn.Module):
    def __init__(self):
        super(PW_ATM, self).__init__()
        self.conv0 = nn.Conv2d(1, 64, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(64, 256, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, point_cloud):
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

        net = self.relu(self.bn2(self.conv2(net)))  # [B, 128, 1, N]
        net = self.sigmoid(self.bn3(self.conv3(net)))  # [B, 1, 1, N]

        transform = net.squeeze(dim=-1).squeeze(dim=-1)  # [B, 1, 1, N]
        transform = transform.permute(0, 3, 2, 1).squeeze(-1)

        return transform  # transform [B,N,1] 比例系数


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss


class EMD_loss(nn.Module):  # emd_loss
    def __init__(self):
        super(EMD_loss, self).__init__()

    def forward(self, Target_Z, Source_Z):
        norm_squared = torch.sum(torch.pow(Target_Z - Source_Z, 2))
        norm = torch.sqrt(norm_squared)
        emd_loss = torch.min(norm)

        return emd_loss


class ADV_loss(nn.Module):  # ADV_loss
    def __init__(self):
        super(ADV_loss, self).__init__()

    def forward(self, F1_pred, F2_pred):
        adv_loss = torch.sum(torch.abs(F1_pred - F2_pred)) / (F1_pred.size[0] * F1_pred.size[1])

        return adv_loss


if __name__ == '__main__':
    import torch

    model = get_model(13)
    # print(list(model.pw_atm.parameters()))
    # # model = pw_atm()
    xyz1 = torch.rand(6, 9, 1024)
    xyz2 = torch.rand(6, 9, 1024)
    (model(xyz1, xyz2))
    print(model)
