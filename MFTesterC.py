import numpy as np
import torch
from ctypes import *
from numpy.ctypeslib import ndpointer
import yaml

def uint16tofloat32(data):
    data = np.left_shift(data.astype(np.int32), 16).view(np.float32)
    return data


def np_float2np_bf16(arr):
    ''' Convert a numpy array of float to a numpy array
    of bf16 in uint16'''
    orig = arr.view('<u4')
    bias = np.bitwise_and(np.right_shift(orig, 16), 1) + 0x7FFF
    new_arr = np.right_shift(orig + bias, 16).astype('uint16')
    if isinstance(new_arr, np.uint16):
        return new_arr
    else:
        return np.where(new_arr == 32768, 0, new_arr)


def wrap_function(lib, funcname, argtypes, restype):
    """Simplify wrapping ctypes functions"""
    func = lib.__getattr__(funcname)
    func.argtypes = argtypes
    func.restype = restype
    return func


class MFTesterC:
    def __init__(self, mfmodeltesterv2_lib_path):
        self.mfmodeltesterv2 = CDLL(mfmodeltesterv2_lib_path)

        # int init(char * yaml_path);
        self.init = wrap_function(self.mfmodeltesterv2, 'model_tester_init', [c_char_p, c_int], c_int)

        self.append_param = wrap_function(self.mfmodeltesterv2, 'model_tester_append_uint8_input', [ndpointer(c_uint8, flags="C_CONTIGUOUS"), c_ulong], c_int)

        self.inference = wrap_function(self.mfmodeltesterv2, 'model_tester_inference', [ndpointer(c_uint8, flags="C_CONTIGUOUS"), c_ulong],
                                       c_int)

        self.destroy = wrap_function(self.mfmodeltesterv2, 'model_tester_destroy', None, None)
        
        self.core = 1
    
    def load_model(self, model_path, device_id=0):
        ret = self.init(model_path.encode('utf-8'), device_id)
        #self.get_output_info(model_path)
        
    def get_output_info(self, yaml_path):
        dtype_map = {
            "bf16": np.uint16,
            "int8": np.int8
            }
        data = yaml.safe_load(open(yaml_path, "r"))
        self.output_shape = data["model_output"][0]["shape"]
        self.dtype = dtype_map[data["model_output"][0]["dtype"]]

        
    def preprocess(self, *args, **kwargs):
        raise NotImplementedError
    
    
    def moffett_inference(self, data):
        output_data = np.ones(shape=self.output_shape, dtype=self.dtype).view(np.uint8)
        for input_data in data:
            self.append_param(input_data, input_data.nbytes)
        self.inference(output_data, output_data.nbytes)
        return output_data
    
    
    def postprocess(self, *args, **kwargs):
        raise NotImplementedError
        
    def forward(self, data):
        data = self.preprocess(data)
        output = self.moffett_inference(data)
        output = self.postprocess(output, batch_size)
        return output
        
        
class YoloRuntime(MFTesterC):
    def __init__(self, mfmodeltesterv2_lib_path=None):
        super(YoloRuntime, self).__init__(mfmodeltesterv2_lib_path=mfmodeltesterv2_lib_path)
        
    def preprocess(self, data):
        data = torch.round(torch.clip(data.float(), -128, 127).float()).int()
        data = data.permute(0,2,3,1).contiguous()
        data = data.numpy().view(dtype=np.uint8)
        return data

    def postprocess(self, data):
        data = np.expand_dims(data.squeeze()[:,:9], axis=0)
        data = uint16tofloat32(data)
        data = torch.from_numpy(data)
        return data
        
        
