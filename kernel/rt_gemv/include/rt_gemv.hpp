#pragma once

struct Params {
    OptixTraversableHandle handle;
    float radius ;
};

struct RayGenData {
    float3* ray_origin;
};

struct MissData {
    float r, g, b;
};

struct HitGroupData {
    float *hit_record; // 64 * 256 ?
};