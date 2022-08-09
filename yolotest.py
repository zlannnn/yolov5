import MFTesterC
import numpy as np
import torch

yolo_path = '/home/kylin/deploy/mfmodeltesterv2/install/kylin/lib/libmfmodeltesterv2.so'
mf_yolo = MFTesterC.MFTesterC(yolo_path)
print('load library finished!')
yaml_path = '/home/kylin/work/dataset_yolov5/CCSFF_yolov5#N1#640_sparseX16_int8_single_core_fase_b2_mode_false_context1_batch_parallel_false_0/CCSFF_yolov5#N1#640_sparseX16_int8_single_core_fase_b2_mode_false_context1_batch_parallel_false_0_7a6a82d/chip_runtime.yaml'
ret = mf_yolo.init(yaml_path.encode('utf-8'), 0)
myout = np.ones([1,1,1,1,25200,32], dtype=np.uint8)

im = torch.randn(1,640,640,3)
im = im.numpy().view(dtype=np.uint8)
print(im.shape)
mf_yolo.append_param(im, im.nbytes)
mf_yolo.inference(myout, myout.nbytes)
print(myout)
