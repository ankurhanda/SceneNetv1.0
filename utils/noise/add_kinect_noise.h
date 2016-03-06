#pragma once

#include <iosfwd>
#include <vector_types.h>
#include <stdio.h>
#include <string.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <sys/time.h>
#include <iu/iucore.h>

#include <cuda.h>
#include <cuda_runtime.h>
//#include "cu_math_stuff.h"

//#include <cutil_math.h>

#include "/usr/local/cuda/samples/common/inc/helper_math.h"


using namespace std;

namespace noise{

void uploadTexture2CUDA(float* depth,
                        unsigned int pitchf1,
                        unsigned int width,
                        unsigned int height);

void  warpImage(float* depth_new,
                unsigned int stridef1,
                float2* tex_coords,
                unsigned int stridef2,
                unsigned int width,
                unsigned int height);


void launch_add_kinect_noise(float4* points3D, float4* normals3D,
                             float4* noisy_points, const unsigned int stridef4,
                             const unsigned int height,
                             const float focal_length,
                             const float theta_1,
                             const float theta_2,
                             float z1,
                             float z2,
                             float z3);

void launch_get_z_coordinate_only(float4* vertex_with_noise,
                                  const unsigned int stridef4,
                                  const unsigned int width,
                                  const unsigned int height,
                                  float* noisy_depth,
                                  const unsigned int stridef1
                                  );
void launch_colour_from_normals(float4* normals,
                                float4* colour,
                                const unsigned int stridef4,
                                const unsigned int height);

void  launch_convert_depth2png(float* noisy_depth,
                               const unsigned int stridef1,
                               u_int16_t* noisy_depth_png,
                               const unsigned int strideu16,
                               const unsigned int width,
                               const unsigned int height);

void convertVerts2Depth(iu::ImageGpu_32f_C1 *depth, iu::ImageGpu_32f_C4 *vertex, float2 pp, float2 fl);


void generate_smooth_noise(iu::ImageGpu_32f_C1 *smoothNoise,
                           iu::ImageGpu_32f_C1 *baseNoise,
                           const float samplePeriod,
                           const float sampleFrequency,
                           const unsigned int width,
                           const unsigned int height);

void generate_perlin_noise(iu::ImageGpu_32f_C1* perlinNoise,
                           iu::ImageGpu_32f_C1* baseNoise,
                           const float _amplitude,
                           const float persistance,
                           const int octaveCount);

void add_noise2vertex(iu::ImageGpu_32f_C4* vertex,
                      iu::ImageGpu_32f_C4* normals,
                      iu::ImageGpu_32f_C4* vertex_with_noise,
                      iu::ImageGpu_32f_C1* perlinNoise);


void gaussian_shifts(float2* tex_coods,
                     const unsigned int stridef2,
                     const unsigned int height,
                     const float _sigma);

void add_depth_noise_barronCVPR2013(float* noisy_depth_copy,
                               const int stridef1,
                               const int height);

void ComputeVertexFromDepth(float* depth,
                            const unsigned int stridef1,
                            float4* vertex,
                            const unsigned int stridef4,
                            const unsigned int width,
                            const unsigned int height,
                            const float2 fl,
                            const float2 pp,
                            const float near_plane,
                            const float far_plane);

void ComputeNormalsFromVertex(float4* normal,
                              float4* vertex,
                              const unsigned int stridef4,
                              const unsigned int width,
                              const unsigned int height);

void ComputeDepthFromVertex(float4* vertex,
                            const unsigned int stridef4,
                            float* depth,
                            const unsigned int stridef1,
                            const unsigned int width,
                            const unsigned int height,
                            const float2 fl,
                            const float2 pp);
}
