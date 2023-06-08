"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.DataLoader import testsetWholeScene
from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['Artificialterrain', 'Naturalterrain', 'Highvegetation', 'Lowvegetation', 'Buildings',
           'Hardscape', 'Scanningartifacts', 'Automobile', 'Nopointsmarked']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    # parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3,
                        help='aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            # if weight[b, n] != 0 and not np.isinf(weight[b, n]):
            #     vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
            vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1  # TODO：忽略了权重的影响，暂不清楚会不会影响结果
    return vote_label_pool


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 9
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    root = 'data/testset/'

    TEST_DATASET_WHOLE_SCENE = testsetWholeScene(root, block_points=NUM_POINT)
    log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        log_string('---- EVALUATION WHOLE SCENE----')

        for batch_idx in range(num_batches):
            print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
            try:
                if args.visual:
                    fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
                whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
                vote_label_pool = np.zeros((whole_scene_data.shape[0], NUM_CLASSES))
                for _ in tqdm(range(args.num_votes), total=args.num_votes):
                    scene_data, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
                    num_blocks = scene_data.shape[0]
                    s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
                    batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))
                    batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                    for sbatch in range(s_batch_num):
                        start_idx = sbatch * BATCH_SIZE
                        end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                        real_batch_size = end_idx - start_idx
                        batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]  # [B,N,C]
                        batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]  # [B,N]
                        batch_data[:, :, 3:6] /= 1.0
                        torch_data = torch.Tensor(batch_data)
                        torch_data = torch_data.float().cuda()
                        torch_data = torch_data.transpose(2, 1)
                        seg_pred, _ = classifier(torch_data, torch_data, step='test')
                        batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                        vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                                   batch_pred_label[0:real_batch_size, ...],
                                                   1)  # TODO：需要解决batch_smpw问题，
                pred_label = np.argmax(vote_label_pool, 1)
                print('----------------------------')
                filename = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
                with open(filename, 'w') as pl_save:
                    for i in pred_label:
                        pl_save.write(str(int(i)) + '\n')
                    pl_save.close()
                for i in range(whole_scene_data.shape[0]):
                    color = g_label2color[pred_label[i]]
                    if args.visual:
                        fout.write('v %f %f %f %d %d %d\n' % (
                            whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1],
                            color[2]))
                if args.visual:
                    fout.close()
            except Exception as e:
                print(f" %s failed at {str(e)}" % (scene_id[batch_idx]) )
        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
