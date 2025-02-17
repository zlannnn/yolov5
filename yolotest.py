from MFTesterC import YoloRuntime
import numpy as np
import torch

lib_path = '/home/kylin/deploy/mfmodeltesterv2/install/kylin/lib/libmfmodeltesterv2.so'
mf_yolo = YoloRuntime(mfmodeltesterv2_lib_path=lib_path)
print('load library finished!')
model_path = '/home/kylin/work/dataset_yolov5/CCSFF_yolov5#N1#640_sparseX16_int8_single_core_fase_b2_mode_false_context1_batch_parallel_false_0/CCSFF_yolov5#N1#640_sparseX16_int8_single_core_fase_b2_mode_false_context1_batch_parallel_false_0_7a6a82d/chip_runtime.yaml'
mf_yolo.load_model(model_path=model_path)
mf_yolo.get_output_info(yaml_path=model_path)
myout = np.ones([1,1,1,1,25200,32], dtype=np.uint8)
im = torch.randn(1,640,640,3)
data = mf_yolo.preprocess(im)
print(im.shape)
data = mf_yolo.moffett_inference(data)
myout = mf_yolo.postprocess(data)
print(myout)
