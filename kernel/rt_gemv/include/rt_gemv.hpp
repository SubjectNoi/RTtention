#pragma once

struct Params {
    OptixTraversableHandle handle;
    float radius ;
};

struct RayGenData {
    float* ray_origin;
};

struct MissData {
    float r, g, b;
};

struct HitGroupData {
    float *hit_record; // 64 * 256 ?
};