from ctypes import *
from numpy.ctypeslib import ndpointer


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