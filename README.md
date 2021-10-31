# SCADN
This is the implementation of paper “Learning Semantic Context from Normal Samplesfor Unsupervised Anomaly Detection”.
# Dataset
Dataset should be stored in `./dataset/xxx`.

The Mvtec AD dataset could be download at : https://www.mvtec.com/company/research/datasets/mvtec-ad/
# Environment
Python = 3.6.8

Pytorch = 1.1.0

opencv = 3.4.2

mmcv = 0.4.0

scikit-image = 0.15.0

# Testing

On Mvtec AD dataset:

Download checkpoints form https://pan.baidu.com/s/1TEwc_C9HPs6GOrRA1z_HfA code:9s47.

Then unzip file and move it to `./ckpt`.

Run this comand to get result of `${class_name}` such as bottle etc.

`python test.py configs/config_mvtec.py --subset ${class_name}`
