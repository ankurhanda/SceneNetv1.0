#ifndef _AUX_MATH_H_
#define _AUX_MATH_H_

#undef isfinite
#undef isnan

#include <vector_types.h>
#include <cuda_runtime.h>
#include <iostream>
#include <assert.h>
#include <boost/math/common_factor.hpp>
#include <TooN/se3.h>
#include "cumath.h"

#include "/usr/local/cuda/samples/common/inc/helper_math.h"

namespace aux_math{

//template<typename To, typename Ti>
//__global__ void KernBilateralFilter(
//    Image<To> dOut, const Image<Ti> dIn, float gs, float gr, int size, Ti minval
//) {
//    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
//    const uint y = blockIdx.y*blockDim.y + threadIdx.y;

//    if( dOut.InBounds(x,y)) {
//        const Ti p = dIn(x,y);
//        float sum = 0;
//        float sumw = 0;

//        if( p >= minval) {
//            for(int r = -size; r <= size; ++r ) {
//                for(int c = -size; c <= size; ++c ) {
//                    const Ti q = dIn.GetWithClampedRange(x+c, y+r);
//                    if(q >= minval) {
//                        const float sd2 = r*r + c*c;
//                        const float id = p-q;
//                        const float id2 = id*id;
//                        const float sw = __expf(-(sd2) / (2 * gs * gs));
//                        const float iw = __expf(-(id2) / (2 * gr * gr));
//                        const float w = sw*iw;
//                        sumw += w;
//                        sum += w * q;
//                    }
//                }
//            }
//        }

//        const To outval = (To)(sum / sumw);
//        dOut(x,y) = outval;
//    }
//}

//template<typename To, typename Ti>
//void BilateralFilter(
//    Image<To> dOut, const Image<Ti> dIn, float gs, float gr, uint size, Ti minval
//) {
//    dim3 blockDim, gridDim;
//    InitDimFromOutputImageOver(blockDim,gridDim, dOut);
//    KernBilateralFilter<To,Ti><<<gridDim,blockDim>>>(dOut, dIn, gs, gr, size, minval);
//}

//template void BilateralFilter(Image<float>, const Image<float>, float, float, uint, float);

void filterBilateral(float* d_image_in, float* d_image_out,
                     const unsigned int stridef1, float sigma_s,
                     float sigma_r, int2 img_size, int2 filter_radius, float minVal);



inline void convertTooN2CUDAMat( TooN::SE3<>& T_wc,
                                 Mat<float,3,4>& T_wcMat)
{
    /// Copy the matrix
    TooN::Matrix<3>Rwc_TooN = T_wc.get_rotation().get_matrix();
    TooN::Vector<3>twc_TooN = T_wc.get_translation();

    /// Convert TooN matrix to CUDA compatiable matrix!
    for(int r=0 ; r < 3; r++ )
    {
        for(int c=0; c < 3 ; c++)
        {
            T_wcMat(r,c) = Rwc_TooN(r,c);
        }
    }
    for(int r = 0 ; r < 3 ; r++ )
        T_wcMat(r,3) = twc_TooN[r];
}

void CopyNormals(float4* predNormals,
                 float4* inNormals,
                 const unsigned int stridef4,
                 const unsigned int width,
                 const unsigned int height);

void InterpolateNormals(float4* inNormals,
                        const unsigned stridef4in,
                        float4* outNormals,
                        const unsigned stridef4out,
                        const unsigned int widthIn,
                        const unsigned int heightIn,
                        const unsigned int widthOut,
                        const unsigned int heightOut);


void ComputeNormalsFromVertex(float4* normal,
                              float4* vertex,
                              const unsigned int stridef4,
                              const unsigned int width,
                              const unsigned int height);

void ComputeNormalsFromVertexRobust(float4* normal,
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


void downSampleDepth(float *depth_in,
                     const unsigned int stridefin,
                     float *depth_out,
                     const unsigned int stridefout,
                     const unsigned int width_in,
                     const unsigned int height_in,
                     const unsigned int width_out,
                     const unsigned int height_out,
                     const float n_plane,
                     const float f_plane);


void RotatePoints(float4* vertex_src,
                  float4* vertex_dest,
                  TooN::SE3<>T_lr,
                  const unsigned int stridef4,
                  const unsigned int width,
                  const unsigned int height);

inline TooN::SE3<> ObtainRelativePose_w( TooN::SE3<>& T_w_cpov, TooN::SE3<>& T_cgl_cpov)
{
    return (T_w_cpov.inverse() * T_cgl_cpov).inverse();
}

inline TooN::SE3<> ChangeBasis ( TooN::SE3<>& T_wc, TooN::Matrix<4> T_change_basis)
{
    TooN::Matrix<4>T_wc_tooN = TooN::Identity(4);

    TooN::Matrix<3>Rwc_TooN = T_wc.get_rotation().get_matrix();
    TooN::Vector<3>twc_TooN = T_wc.get_translation();

    for(int r=0 ; r < 3; r++ )
    {
        for(int c=0; c < 3 ; c++)
        {
            T_wc_tooN(r,c) = Rwc_TooN(r,c);
        }
    }
    for(int r = 0 ; r < 3 ; r++ )
        T_wc_tooN(r,3) = twc_TooN[r];


    TooN::Matrix<4> T_wc_changed = T_change_basis*T_wc_tooN*T_change_basis;

    std::cout << "T_wc_changed = " << T_wc_changed << std::endl;

    TooN::Vector<3> trans = TooN::makeVector(T_wc_changed(0,3),
                                             T_wc_changed(1,3),
                                             T_wc_changed(2,3));

    TooN::SO3<> Rotation  = TooN::SO3<>(T_wc_changed.slice<0,0,3,3>());
    return TooN::SE3<>(Rotation,trans);

}

/// Wrapper for getting the 3D positions
}


#endif
