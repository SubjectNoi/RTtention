#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include <torch/extension.h>
#include "stdio.h"
#include <iostream>
#include <fstream>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include "mma.h"
#include <random>
#include <optix.h>
#include <optix_host.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
// #include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
// #include <sutil/sutil.h>
#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include "rt_gemv.hpp"

#define SPACE 64
#define ENTRY 256
#define RADIUS 0.5


template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
    << message << "\n";
}

class rt_pipe {
public:
    OptixDeviceContext              context             = nullptr;
    OptixDeviceProperty             property            = {};
    CUcontext                       cuCtx               = 0;
    OptixDeviceContextOptions       options             = {};

    OptixTraversableHandle          gas_handle;
    CUdeviceptr                     d_gas_output_buffer;
    CUdeviceptr                     d_params;
    OptixAccelBuildOptions          accel_options       = {};
    char                            log[2048];

	OptixPipeline                   pipeline            = nullptr;
    OptixShaderBindingTable         sbt                 = {};

    float*                          d_ray_origin;
    Params                          params;
    float*                          d_hit_record;

    CUstream                        stream;
public:
	rt_pipe() {
        std::cout << "a" << std::endl;
		options.logCallbackFunction = &context_log_cb;
        std::cout << "b" << std::endl;
        options.logCallbackLevel = 4;
        std::cout << "c" << std::endl;
        CUDA_CHECK(cudaFree(0));
        std::cout << "d" << std::endl;
        OPTIX_CHECK(optixInit());
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
        std::cout << "e" << std::endl;
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        std::cout << "f" << std::endl;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
        std::cout << "g" << std::endl;
        CUDA_CHECK(cudaStreamCreate(&stream));
        std::cout << "h" << std::endl;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ray_origin), sizeof(float3) * 1 * SPACE));
        std::cout << "i" << std::endl;
	}

	void build_index_from_codebook(half* codebook, float* centers, float* radius);

    void set_ray_origin(float* ray_origin, int num_rays);

    void run();
};

// Codebook: (SPACE=64, ENTRY=256, 2)
void rt_pipe::build_index_from_codebook(half* d_codebook, float* centers, float* radius) {

    std::cout << "Begin" << std::endl;
    // float3 *centers = new float3 [SPACE * ENTRY];
    // float  *radius  = new float  [SPACE * ENTRY];
    // for (int level = 0; level < SPACE; level++) {
    //     for (int entry = 0; entry < ENTRY; entry++) {
    //         int idx = level * ENTRY + entry;
    //         float x = __half2float(codebook[level * ENTRY * 2 + entry * 2 + 0]);
    //         float y = __half2float(codebook[level * ENTRY * 2 + entry * 2 + 1]);
    //         float z = level * 2 + 1; // 1, 3, 5, 7, ..., 127
    //         centers[idx] = make_float3(x, y, z);
    //         radius[idx] = sqrt(RADIUS * RADIUS + x * x + y * y);
    //     }
    // }

    std::cout << "0" << std::endl;
    CUdeviceptr d_centers;
    CUdeviceptr d_radius;
    d_centers = reinterpret_cast<CUdeviceptr>(centers);
    d_radius = reinterpret_cast<CUdeviceptr>(radius);
    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_centers), SPACE * ENTRY * sizeof(float3)));
    // CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_centers), centers, SPACE * ENTRY * sizeof(float3), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_radius), SPACE * ENTRY * sizeof(float)));
    // CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_radius), radius, SPACE * ENTRY * sizeof(float), cudaMemcpyHostToDevice));

    uint32_t sphere_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    OptixBuildInput sphere_input = {};
    sphere_input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    sphere_input.sphereArray.vertexBuffers = &d_centers;
    sphere_input.sphereArray.numVertices = SPACE * ENTRY;
    sphere_input.sphereArray.radiusBuffers = &d_radius;
    sphere_input.sphereArray.flags = sphere_input_flags;
    sphere_input.sphereArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &sphere_input, 1, &gas_buffer_sizes));
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));
    CUdeviceptr d_gas_output_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));
    OPTIX_CHECK(optixAccelBuild(context, 0, &accel_options, &sphere_input, 1, d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes, d_gas_output_buffer, gas_buffer_sizes.outputSizeInBytes, &gas_handle, nullptr, 0));    
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
    std::cout << "1" << std::endl;

    OptixModule module = nullptr;
    OptixModule shpere_module;
    OptixModuleCompileOptions module_compile_options = {};
    OptixPipelineCompileOptions pipeline_compile_options = {};    
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 3;
    pipeline_compile_options.numAttributeValues = 3;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;

    std::string input_shader;
    std::ifstream file(std::string("/home/wennitao/workspace/RTtention/kernel/rt_gemv/rt_gemv_shader.optixir"), std::ios::binary);
    if (file.good()) {
        std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
        input_shader.assign(buffer.begin(), buffer.end());
    }
    size_t shader_binary_size = input_shader.size();
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreate(context, &module_compile_options, &pipeline_compile_options, input_shader.c_str(), shader_binary_size, log, &sizeof_log, &module));
    
    OptixBuiltinISOptions builtin_is_options = {};
    builtin_is_options.usesMotionBlur = false;    
    builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
    OPTIX_CHECK_LOG(optixBuiltinISModuleGet(context, &module_compile_options, &pipeline_compile_options, &builtin_is_options, &shpere_module));

    std::cout << "2" << std::endl;
    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;

    OptixProgramGroupOptions program_group_options = {};
    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, 
                                            &raygen_prog_group_desc, 
                                            1, 
                                            &program_group_options,
                                            log,
                                            &sizeof_log,
                                            &raygen_prog_group
                                            ));
    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, 
                                            &miss_prog_group_desc, 
                                            1, 
                                            &program_group_options, 
                                            log, 
                                            &sizeof_log, 
                                            &miss_prog_group
                                            ));
    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH = module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hitgroup_prog_group_desc.hitgroup.moduleAH = module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
    hitgroup_prog_group_desc.hitgroup.moduleIS = shpere_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = nullptr;
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, 
                                            &hitgroup_prog_group_desc, 
                                            1, 
                                            &program_group_options, 
                                            log, 
                                            &sizeof_log, 
                                            &hitgroup_prog_group
                                            ));
    const uint32_t max_trace_depth = 1;
    OptixProgramGroup program_groups[] = {raygen_prog_group, miss_prog_group, hitgroup_prog_group};
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(context, 
                                        &pipeline_compile_options, 
                                        &pipeline_link_options, 
                                        program_groups, 
                                        sizeof(program_groups) / sizeof(program_groups[0]), 
                                        log, 
                                        &sizeof_log, 
                                        &pipeline
                                        ));
    OptixStackSizes stack_sizes = {};
    for (auto& prog_group: program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));
    }

    std::cout << "3" << std::endl;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_satck_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, 
                                            max_trace_depth, 
                                            0, 
                                            0, 
                                            &direct_callable_stack_size_from_traversal, 
                                            &direct_callable_stack_size_from_state, 
                                            &continuation_satck_size
                                            ));
    OPTIX_CHECK(optixPipelineSetStackSize(pipeline, 
                                          direct_callable_stack_size_from_traversal, 
                                            direct_callable_stack_size_from_state, 
                                            continuation_satck_size, 
                                            1
                                            ));
    CUdeviceptr raygen_record;
    size_t raygen_record_size = sizeof(RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
    RayGenSbtRecord rg_sbt;
    rg_sbt.data.ray_origin = d_ray_origin;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
    sbt.raygenRecord = raygen_record;

    CUdeviceptr miss_record;
    size_t miss_record_size = sizeof(MissSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
    MissSbtRecord ms_sbt;    
    ms_sbt.data = {0.0f, 0.0f, 0.0f};
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(miss_record), &ms_sbt, miss_record_size, cudaMemcpyHostToDevice));
    sbt.missRecordBase = miss_record;
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.missRecordCount = 1;  

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit_record), sizeof(float) * SPACE * ENTRY));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_hit_record), 0, sizeof(float) * SPACE * ENTRY));
    CUdeviceptr hitgroup_record;
    size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
    HitGroupSbtRecord hg_sbt;
    hg_sbt.data.hit_record = d_hit_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroup_record), &hg_sbt, hitgroup_record_size, cudaMemcpyHostToDevice));
    sbt.hitgroupRecordBase = hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    sbt.hitgroupRecordCount = 1; 

    std::cout << "4" << std::endl;
    params.handle = gas_handle;
    params.radius = 0.5;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(params)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params), &params, sizeof(params), cudaMemcpyHostToDevice));
}

void rt_pipe::set_ray_origin(float* ray_origin, int num_rays) {
    // CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_ray_origin), host_ray_origin, sizeof(float3) * 1 * num_rays, cudaMemcpyHostToDevice));
    d_ray_origin = ray_origin;
}

void rt_pipe::run() {
    OPTIX_CHECK(optixLaunch(pipeline, stream, d_params, sizeof(Params), &sbt, SPACE, 1, 1));
}

torch::Tensor rt_gemv(
    torch::Tensor input,
    torch::Tensor quantized_w,
    torch::Tensor codebook, 
    torch::Tensor centers, 
    torch::Tensor radius, 
    torch::Tensor origins
)
{
	auto head_dim = input.size(1);
	auto seq_len  = quantized_w.size(0);
	auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);

	half* codebook_ptr = reinterpret_cast<half*>(codebook.data_ptr<at::Half>());
    half* input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());
    float* centers_ptr = centers.data_ptr<float>();
    float* radius_ptr = radius.data_ptr<float>();
    float* origins_ptr = origins.data_ptr<float>();

    std::cout << "Work Begins" << std::endl;
	rt_pipe pipe;
	pipe.build_index_from_codebook(codebook_ptr, centers_ptr, radius_ptr);
    std::cout << "Pipeline built" << std::endl;
    // Set ray origin with input
    // float3 *h_ray_origin = new float3 [SPACE];
    // for (int ray = 0; ray < SPACE; ray++) {
    //     float x = __half2float(input_ptr[ray * 2 + 0]);
    //     float y = __half2float(input_ptr[ray * 2 + 1]);
    //     float z = ray * 2;
    // }
    // pipe.set_ray_origin(h_ray_origin, SPACE);
    pipe.set_ray_origin(origins_ptr, SPACE);  
    std::cout << "Origin set" << std::endl;

    // Launch ray tracing pipeline
    pipe.run();
    CUDA_SYNC_CHECK();
    std::cout << "Launched" << std::endl;

    torch::Tensor o = torch::full({1, seq_len}, 0, options);
	return o;
}
