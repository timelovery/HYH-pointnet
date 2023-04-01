import os
import random
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset


class TrainDataLoader(Dataset):
    def __init__(self, Source_root='trainval_fullarea', Target_root='trainval_fullarea', num_point=4096,
                 block_size=1.0,
                 sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        Source_areas = sorted(os.listdir(Source_root))
        Source_areas = [area for area in Source_areas if '.npy' in area]  # 所有的文件的输入格式应该为npy
        Target_areas = sorted(os.listdir(Target_root))
        Target_areas = [area for area in Target_areas if '.npy' in area]  # 所有的文件的输入格式应该为npy

        self.Source_points, self.Source_labels = [], []
        self.Source_coord_min, self.Source_coord_max = [], []
        self.Target_points = []  # Target 数据没有 labels
        self.Target_coord_min, self.Target_coord_max = [], []

        Source_num_point_all = []
        Target_num_point_all = []
        Source_labelweights = np.zeros(9)

        for Source_area_name in tqdm(Source_areas, total=len(Source_areas)):
            area_path = os.path.join(Source_root, Source_area_name)
            area_data = np.load(area_path)  # xyzrgbl, N*7
            points, labels = area_data[:, 0:6], area_data[:, 6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(10))
            Source_labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.Source_points.append(points), self.Source_labels.append(labels)
            self.Source_coord_min.append(coord_min), self.Source_coord_max.append(coord_max)
            Source_num_point_all.append(labels.size)
        for Target_area_name in tqdm(Target_areas, total=len(Target_areas)):
            area_path = os.path.join(Target_root, Target_area_name)
            area_data = np.load(area_path)  # xyzrgb, N*6
            points = area_data  # xyzrgb, N*6;
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.Target_points.append(points),
            self.Target_coord_min.append(coord_min), self.Target_coord_max.append(coord_max)
            Target_num_point_all.append(points.shape[0])

        Source_labelweights = Source_labelweights.astype(np.float32)
        Source_labelweights = Source_labelweights / np.sum(Source_labelweights)
        self.Source_labelweights = np.power(np.amax(Source_labelweights) / Source_labelweights, 1 / 3.0)

        # source
        Source_sample_prob = Source_num_point_all / np.sum(Source_num_point_all)
        Source_num_iter = int(np.sum(Source_num_point_all) * sample_rate / num_point)  # 创建迭代器
        Source_area_idxs = []
        for index in range(len(Source_areas)):
            Source_area_idxs.extend([index] * int(round(Source_sample_prob[index] * Source_num_iter)))
        # target
        Target_sample_prob = Target_num_point_all / np.sum(Target_num_point_all)
        Target_sum_iter = int(np.sum(Target_num_point_all) * sample_rate / num_point)
        Target_area_idxs = []
        for index in range(len(Target_areas)):
            Target_area_idxs.extend([index] * int(round(Target_sample_prob[index] * Target_sum_iter)))

        # 实现数据量对齐
        if len(Source_area_idxs) > len(Target_area_idxs):
            for i in range(len(Target_area_idxs), len(Source_area_idxs)):
                Target_area_idxs.append(random.choice(Target_area_idxs))
        else:
            for i in range(len(Source_area_idxs), len(Target_area_idxs)):
                Source_area_idxs.append(random.choice(Source_area_idxs))
        self.Source_area_idxs = np.array(Source_area_idxs)
        self.Target_area_idxs = np.array(Target_area_idxs)
        print("Totally {} samples in {} set.".format(len(self.Source_area_idxs), "Source_area"))
        print("Totally {} samples in {} set.".format(len(self.Target_area_idxs), "Target_area"))

    def __getitem__(self, idx):
        Source_area_idx = self.Source_area_idxs[idx]
        Target_area_idx = self.Target_area_idxs[idx]
        Source_points = self.Source_points[Source_area_idx]
        Source_labels = self.Source_labels[Source_area_idx]
        Target_points = self.Target_points[Target_area_idx]
        Source_N_points = Source_points.shape[0]
        Target_N_points = Target_points.shape[0]

        Source_selected_point_idx, Source_center = self.point_idxs(Source_points, Source_N_points)  # 选择源场景数据
        Target_selected_point_idx, Target_center = self.point_idxs(Target_points, Target_N_points)  # 选择目标场景数据
        Source_current_points = self.normalize(Source_points, Source_selected_point_idx,
                                               Source_area_idx, Source_center, self.Source_coord_max)
        Target_current_points = self.normalize(Target_points, Target_selected_point_idx,
                                               Target_area_idx, Target_center, self.Target_coord_max)

        Source_current_labels = Source_labels[Source_selected_point_idx]
        if self.transform is not None:
            Source_current_points, Source_current_labels = self.transform(Source_current_points, Source_current_labels)
        return Source_current_points, Source_current_labels, Target_current_points

    def point_idxs(self, points, N_points):
        while True:
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where(
                (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (
                        points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break
        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)
        return selected_point_idxs, center

    def normalize(self, points, selected_point_idxs, room_idx, center, coord_max):
        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        return current_points

    def __len__(self):
        return len(self.Source_area_idxs)


class TestDataLoader(Dataset):
    def __init__(self, Test_root='trainval_fullarea', num_point=4096, block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        Source_areas = sorted(os.listdir(Test_root))
        Source_areas = [area for area in Source_areas if '.npy' in area]  # 所有的文件的输入格式应该为npy

        self.Source_points, self.Source_labels = [], []
        self.Source_coord_min, self.Source_coord_max = [], []

        Source_num_point_all = []
        Source_labelweights = np.zeros(9)

        for Source_area_name in tqdm(Source_areas, total=len(Source_areas)):
            area_path = os.path.join(Test_root, Source_area_name)
            area_data = np.load(area_path)  # xyzrgbl, N*7
            points, labels = area_data[:, 0:6], area_data[:, 6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(10))
            Source_labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.Source_points.append(points), self.Source_labels.append(labels)
            self.Source_coord_min.append(coord_min), self.Source_coord_max.append(coord_max)
            Source_num_point_all.append(labels.size)

        Source_labelweights = Source_labelweights.astype(np.float32)
        Source_labelweights = Source_labelweights / np.sum(Source_labelweights)
        self.Source_labelweights = np.power(np.amax(Source_labelweights) / Source_labelweights, 1 / 3.0)

        # source
        Source_sample_prob = Source_num_point_all / np.sum(Source_num_point_all)
        Source_num_iter = int(np.sum(Source_num_point_all) * sample_rate / num_point)  # 创建迭代器
        Source_area_idxs = []
        for index in range(len(Source_areas)):
            Source_area_idxs.extend([index] * int(round(Source_sample_prob[index] * Source_num_iter)))

        self.Source_area_idxs = np.array(Source_area_idxs)
        print("Totally {} samples in {} set.".format(len(self.Source_area_idxs), "Source_area"))

    def __getitem__(self, idx):
        Source_area_idx = self.Source_area_idxs[idx]
        Source_points = self.Source_points[Source_area_idx]
        Source_labels = self.Source_labels[Source_area_idx]
        Source_N_points = Source_points.shape[0]

        Source_selected_point_idx, Source_center = self.point_idxs(Source_points, Source_N_points)  # 选择源场景数据
        Source_current_points = self.normalize(Source_points, Source_selected_point_idx,
                                               Source_area_idx, Source_center, self.Source_coord_max)

        Source_current_labels = Source_labels[Source_selected_point_idx]
        if self.transform is not None:
            Source_current_points, Source_current_labels = self.transform(Source_current_points, Source_current_labels)
        return Source_current_points, Source_current_labels

    def point_idxs(self, points, N_points):
        while True:
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where(
                (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (
                        points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break
        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)
        return selected_point_idxs, center

    def normalize(self, points, selected_point_idxs, room_idx, center, coord_max):
        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        return current_points

    def __len__(self):
        return len(self.Source_area_idxs)


class testsetWholeScene():
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, stride=0.5, block_size=1.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.stride = stride
        self.scene_points_num = []
        self.file_list = sorted(os.listdir(root))
        self.scene_points_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:, :6]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_room, index_room = np.array([]), np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (
                                points[:, 1] >= s_y - self.padding) & (
                            points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        index_room = index_room.reshape((-1, self.block_points))
        return data_room, index_room

    def __len__(self):
        return len(self.scene_points_list)
