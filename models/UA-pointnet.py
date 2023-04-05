import pdb

import torch.nn as nn
import torch.nn.functional as F
import torch
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        self.PA = PW_ATM()  # 数据对齐
        self.FG = SetAbstractionandFeatureProgation()  # 提取器
        self.F1 = Classifier(num_classes)  # 分类器F1
        self.F2 = Classifier(num_classes)  # 分类器F2

    def forward(self, Source_xyz, Target_xyz, step='Step1'):
        if step == 'Step1':
            # 处理Source_xyz
            l0_points_1_S, l4_points_S = self.FG(Source_xyz)
            F1_pred_S = self.F1(l0_points_1_S, l4_points_S)
            F2_pred_S = self.F2(l0_points_1_S, l4_points_S)
            return F1_pred_S, F2_pred_S  # 用来计算CE1和CE2
        elif step == 'Step2':
            transform = self.PA(Target_xyz)
            # Target = transform_Function.apply(Target_xyz, transform)
            # Target_xyz[:, 2, :] = Target_xyz[:, 2, :] * transform.squeeze()
            # Target_z = Target_xyz[:, 2, :]
            return transform  # 用来计算EMD

        elif step == 'Step3':
            # 处理Source_xyz
            l0_points_1_S, l4_points_S = self.FG(Source_xyz)
            F1_pred_S = self.F1(l0_points_1_S, l4_points_S)
            F2_pred_S = self.F2(l0_points_1_S, l4_points_S)
            # 处理Target_xyz
            transform = self.PA(Target_xyz)
            # Target = transform_Function.apply(Target_xyz, transform)
            # Target_xyz[:, 2, :] = Target_xyz[:, 2, :] * transform.squeeze()

            l0_points_1_T, l4_points_T = self.FG(transform)
            F1_pred_T = self.F1(l0_points_1_T, l4_points_T)

            F2_pred_T = self.F2(l0_points_1_T, l4_points_T)
            # print(F1_pred_T)
            # pdb.set_trace()

            return F1_pred_S, F2_pred_S, F1_pred_T, F2_pred_T
            # 用来计算ADV  # 用来计算EMD
        elif step == 'Step4':
            # 处理Source_xyz
            l0_points_1_S, l4_points_S = self.FG(Source_xyz)
            F1_pred_S = self.F1(l0_points_1_S, l4_points_S)
            F2_pred_S = self.F2(l0_points_1_S, l4_points_S)
            # 处理Target_xyz
            transform = self.PA(Target_xyz)
            # Target = transform_Function.apply(Target_xyz, transform)
            # Target_xyz[:, 2, :] = Target_xyz[:, 2, :] * transform.squeeze()

            l0_points_1_T, l4_points_T = self.FG(transform)
            F1_pred_T = self.F1(l0_points_1_T, l4_points_T)
            F2_pred_T = self.F2(l0_points_1_T, l4_points_T)
            return F1_pred_S, F2_pred_S, F1_pred_T, F2_pred_T
        elif step == 'test':
            # 处理Target_xyz
            transform = self.PA(Target_xyz)
            # Target = transform_Function.apply(Target_xyz, transform)
            # Target_xyz[:, 2, :] = Target_xyz[:, 2, :] * transform.squeeze()
            l0_points_1_T, l4_points_T = self.FG(transform)
            F1_pred_T = self.F1(l0_points_1_T, l4_points_T)
            F2_pred_T = self.F2(l0_points_1_T, l4_points_T)

            return F1_pred_T, F2_pred_T


class SetAbstractionandFeatureProgation(nn.Module):
    def __init__(self):
        super(SetAbstractionandFeatureProgation, self).__init__()
        # 提取器
        self.sa1 = PointNetSetAbstraction(1024, 0.5, 32, 9 + 3, [32, 32, 64], False, True)
        self.sa2 = PointNetSetAbstraction(256, 1.0, 32, 64 + 3, [64, 64, 128], False, True)
        self.sa3 = PointNetSetAbstraction(64, 2.0, 32, 128 + 3, [128, 128, 256], False, True)
        self.sa4 = PointNetSetAbstraction(16, 4.0, 32, 256 + 3, [256, 256, 512], False, True)
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


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        # 分类器

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, l0_points_1, l4_points):
        x_1 = self.drop1(F.relu(self.bn1(self.conv1(l0_points_1))))
        x_1 = F.log_softmax(self.conv2(x_1), dim=1)
        x_1 = x_1.permute(0, 2, 1)

        return x_1


class PW_ATM(nn.Module):
    def __init__(self):
        super(PW_ATM, self).__init__()
        self.conv0 = nn.Conv2d(1, 64, kernel_size=1, bias=True)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 256, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, point_cloud):
        # point_cloud: [B, C, N]
        batch_size = point_cloud.size(0)
        num_point = point_cloud.size(2)
        point_cloud = point_cloud.permute(0, 2, 1)  # point_cloud: [B, N, C]
        orgin_point = point_cloud.clone()

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
        # transform = transform.permute(0,2,1)
        orgin_point[:, :, 2] *= transform[:, :, 0]
        orgin_point = orgin_point.permute(0, 2, 1)

        # return transform  # transform [B,1,N] 比例系数
        return orgin_point


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred1, pred2, target, weight):
        total_loss1 = F.nll_loss(pred1, target, weight=weight)
        total_loss2 = F.nll_loss(pred2, target, weight=weight)
        return total_loss1 + total_loss2


class EMD_loss(nn.Module):  # emd_loss
    def __init__(self):
        super(EMD_loss, self).__init__()

    def forward(self, Target_Z, Source_Z):
        emd_loss = torch.min(torch.norm(Target_Z - Source_Z, dim=1) / 2)
        # emd_loss = torch.min(torch.sum(torch.sqrt(torch.pow(Target_Z - Source_Z, 2)/2)))
        # emd_loss = torch.norm(Target_Z-Source_Z, dim=1)/2

        return emd_loss


class ADV_loss(nn.Module):  # ADV_loss
    def __init__(self):
        super(ADV_loss, self).__init__()

    def forward(self, F1_pred, F2_pred):
        adv_loss = torch.sum(torch.abs(torch.softmax(F1_pred, dim=1) - torch.softmax(F2_pred, dim=1)) / (
                F1_pred.size(0) * F1_pred.size(1)))

        return adv_loss


class transform_Function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, B):
        result = A.clone()
        result[:, 2, :] *= B[:, :, 0]
        ctx.save_for_backward(result, B)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = grad_output.clone()
        grad_A[:, 2, :] *= B[:, :, 0]
        grad_B = torch.sum((grad_output * A[:, 2, :].unsqueeze(1)).permute(0, 2, 1), dim=2, keepdim=True)
        return grad_A, grad_B


if __name__ == '__main__':
    model = get_model(13)
    # print(list(model.pw_atm.parameters()))
    # # model = pw_atm()
    xyz1 = torch.rand(6, 9, 1024)
    xyz2 = torch.rand(6, 9, 1024)
    # (model(xyz1, xyz2))
    # for m in model.modules():
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #         print(m.bias)
    print(model)
