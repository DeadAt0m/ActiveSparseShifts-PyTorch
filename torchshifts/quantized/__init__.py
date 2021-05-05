import torch
import copy
tv = torch.__version__


if tv >= '1.8':
    tndm_prop_list = torch.quantization.quantization_mappings.get_default_qconfig_propagation_list()
    tndm_mapping = torch.quantization.quantization_mappings.get_default_static_quant_module_mappings()
    tndm_qat_mapping = torch.quantization.quantization_mappings.get_default_qat_module_mappings()
    tndm_fuse_mapping = copy.deepcopy(torch.quantization.fuser_method_mappings.DEFAULT_OP_LIST_TO_FUSER_METHOD)
elif tv >= '1.7':
    tndm_prop_list = copy.deepcopy(torch.quantization.quantization_mappings.get_qconfig_propagation_list())
    tndm_mapping = copy.deepcopy(torch.quantization.quantization_mappings.get_static_quant_module_mappings())
    tndm_qat_mapping = copy.deepcopy(torch.quantization.quantization_mappings.get_qat_module_mappings())
    tndm_fuse_mapping = copy.deepcopy(torch.quantization.fuser_method_mappings.OP_LIST_TO_FUSER_METHOD)
else:
    raise RuntimeError('The torch earlier than 1.7 is not supported. Please Update!')


from .modules import Shift1d, Shift2d, Shift3d, new_quant_mapping
quant_mapping = {**tndm_mapping, **new_quant_mapping}

