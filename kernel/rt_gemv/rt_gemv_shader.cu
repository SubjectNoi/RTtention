#include <optix.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "rt_gemv.hpp"

// nvcc -o rt_gemv_shader.optixir --optix-ir -I ${OptiX_INSTALL_DIR}/include/ -I ${OptiX_INSTALL_DIR}/SDK/ -I ${OptiX_INSTALL_DIR}/SDK/support -I ${OptiX_INSTALL_DIR}/SDK/build -I /home/zhliu/workspace/RTtention/kernel/rt_gemv/include rt_gemv_shader.cu

extern "C" {
    __constant__ Params params;
}

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const float3 direction = make_float3(0.0f, 0.0f, 1.0f);
    const RayGenData* rgData = (RayGenData*)optixGetSbtDataPointer();
    int ray_idx = idx.x * dim.y * dim.z + idx.y * dim.z + idx.z ;
    const float3 origin = rgData->ray_origin[ray_idx];
    printf("Ray:%d : (%6.3f, %6.3f, %6.3f)\n", ray_idx, origin.x, origin.y, origin.z);
    optixTrace(params.handle, 
               origin, 
               direction, 
               0.0f, 
               1.0f, 
               0.0f, 
               OptixVisibilityMask(255), 
               OPTIX_RAY_FLAG_NONE, 
               0, 
               0, 
               0
              );
}

extern "C" __global__ void __miss__ms() {

}

extern "C" __global__ void __anyhit__ah() {

}

extern "C" __global__ void __closesthit__ch() {

}