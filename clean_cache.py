# import torch, gc
# torch.cuda.empty_cache()
# gc.collect()

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=xxx'

import torch
foo = torch.tensor([1,2,3])
foo = foo.to('cuda')