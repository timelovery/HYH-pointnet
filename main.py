import os

os.system('python train_ua_pointnet.py --model UA-pointnet --batch_size 24 --log_dir UA-point  --epoch 32')
os.system('python test_ua_pointnet.py --log_dir UA-point --batch_size 24  --visual')