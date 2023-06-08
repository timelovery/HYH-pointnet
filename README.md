
## 本文创建了SUPnetw网络实现利用有限开放基准测试与无语义标签实际城市竣工测绘三维场景数据共同进行训练模型，提高城市三维场景点云语义分割的性能
# SUPnet网络结构如下：

![SUPnet.png](..%2F..%2F%E6%AF%95%E4%B8%9A%E8%AE%BA%E6%96%87-%E5%9F%BA%E4%BA%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84%E5%AE%A4%E5%A4%96%E7%82%B9%E4%BA%91%E4%B8%89%E7%BB%B4%E5%9C%BA%E6%99%AF%E5%88%86%E5%89%B2%2F%E5%B0%8F%E8%AE%BA%E6%96%872%EF%BC%88%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%EF%BC%8C%E5%9F%8E%E5%B8%82%E5%9C%BA%E6%99%AF%E7%82%B9%E4%BA%91%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2%EF%BC%89%2F%E6%8A%95%E7%A8%BF%EF%BC%88%E6%B5%8B%E7%BB%98%E7%A7%91%E5%AD%A6%E6%8A%80%E6%9C%AF%E5%AD%A6%E6%8A%A5%EF%BC%89%2F%E5%9B%BE%E7%89%87%2FSUPnet.png)

SUPnet由特征提取器和分类器以及数据对齐模块共同构建MCD。
具体步骤如下：
1）源场景数据流经特征提取器，然后流经两个分类器，利用 cross-entropy_loss 训练特
征提取器以及两个分类器在源场景中的语义分割性能，同时复制一份源场景数据用以进行步
骤 2）PW-ATM 训练。\
2）目标数据流经 PW-ATM 模块，利用 EMD_loss 来训练 PW-ATM 模块的转换能
力。\
3）目标数据经过 PW-ATM 转换后，流入提取器和两个分类器，利用 ADV_loss 来最
大化分类器差异。在这一步中，冻结特征提取模块和数据对齐模块，仅更新两个分类器的参
数。\
4）目标数据经过 PW-ATM 转换后，流入提取器和两个分类器，利用 ADV_loss 来最
小化分类器差异。在这一步中，冻结两个分类器，更新特征提取器和数据对齐模块的参数。


本项目的data下面应包含四个文件：
.\Source_Scene_Point_Clouds\
.\Target_Scene_Point_Clouds\
.\Validationset\
.\testset


    python train_SUPnet.py --model SUPnet --batch_size 12 --log_dir SUPnet  --epoch 32
    python test_SUPnet.py --log_dir SUPnet --visual

# SUPnet对实际城市竣工数据语义分割结果如下：

![结果.png](..%2F..%2F%E6%AF%95%E4%B8%9A%E8%AE%BA%E6%96%87-%E5%9F%BA%E4%BA%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84%E5%AE%A4%E5%A4%96%E7%82%B9%E4%BA%91%E4%B8%89%E7%BB%B4%E5%9C%BA%E6%99%AF%E5%88%86%E5%89%B2%2F%E5%B0%8F%E8%AE%BA%E6%96%872%EF%BC%88%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%EF%BC%8C%E5%9F%8E%E5%B8%82%E5%9C%BA%E6%99%AF%E7%82%B9%E4%BA%91%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2%EF%BC%89%2F%E6%8A%95%E7%A8%BF%EF%BC%88%E6%B5%8B%E7%BB%98%E7%A7%91%E5%AD%A6%E6%8A%80%E6%9C%AF%E5%AD%A6%E6%8A%A5%EF%BC%89%2F%E5%9B%BE%E7%89%87%2F%E7%BB%93%E6%9E%9C.png)

其中，a列和c列为PointNet++语义分割结果，b列和d列为SUPnet语义分割结果。

表格 网络测试集语义分割准确率

PointNet++ acc（%）
SUPNet acc （%）
实验区A
34.5
89.3
实验区B
37.3
85.1
实验区D
12.8
88.7
实验区F
32.7
90.8
平均
29.3
88.5


## 感谢下面论文提供的思路和代码
* [PointConv: Deep Convolutional Networks on 3D Point Clouds, CVPR'19](https://github.com/Young98CN/pointconv_pytorch)
* [On Isometry Robustness of Deep 3D Point Cloud Models under Adversarial Attacks, CVPR'20](https://github.com/skywalker6174/3d-isometry-robust)
* [Label-Efficient Learning on Point Clouds using Approximate Convex Decompositions, ECCV'20](https://github.com/matheusgadelha/PointCloudLearningACD)
* [PCT: Point Cloud Transformer](https://github.com/MenghaoGuo/PCT)
* [Point Sampling Net: Fast Subsampling and Local Grouping for Deep Learning on Point Cloud](https://github.com/psn-anonymous/PointSamplingNet)
* [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
* [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://readpaper.com/pdf-annotate/note?pdfId=630730691941388288&noteId=1692589319410262272)
* [Unsupervised scene adaptation for semantic segmentation of urban mobile laser scanning point clouds](https://readpaper.com/pdf-annotate/note?pdfId=4731812533506146305&noteId=1705717256133143808)