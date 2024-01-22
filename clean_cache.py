
# import torch, gc
# torch.cuda.empty_cache()
# gc.collect()

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=xxx'

void main() {
    apply_max_split_size_mb();
    
}

void apply_max_split_size_mb() {
    auto max_split_size_mb = get_max_split_size_mb();
    if (max_split_size_mb > 0) {
        TORCH_CUDA_CHECK(cudaDeviceSetLimit(
            cudaLimitMaxSurface1DLayered, max_split_size_mb * 1024 * 1024));
        TORCH_CUDA_CHECK(cudaDeviceSetLimit(
            cudaLimitMaxSurface2DLayered, max_split_size_mb * 1024 * 1024));
        TORCH_CUDA_CHECK(cudaDeviceSetLimit(
            cudaLimitMaxSurfaceCubemapLayered, max_split_size_mb * 1024 * 1024));
    }
}