#include <stdio.h>
#include <vector_types.h>
#include <cuda_runtime.h>
#include "/usr/local/cuda/samples/common/inc/helper_math.h"
#include "aux_math.h"

#include "cumath.h"
#include "MatUtils.h"
#include <TooN/se3.h>

namespace aux_math{

__global__ void cuRotatePoints(float4* vertex_src,
                               float4* vertex_dest,
                               Mat<float,3,4>Tlr,
                               const unsigned int stridef4,
                               const unsigned int width,
                               const unsigned int height)
{

    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if ( x < width && y < height )
    {
        float4 pr = vertex_src[y*stridef4+x];

        float3 transform_pr = Tlr*pr;

        vertex_dest[y*stridef4+x] = make_float4(transform_pr.x,
                                                transform_pr.y,
                                                transform_pr.z,1.0f);
    }

}


void RotatePoints(float4* vertex_src,
                  float4* vertex_dest,
                  TooN::SE3<>T_lr,
                  const unsigned int stridef4,
                  const unsigned int width,
                  const unsigned int height)
{


    const int blockWidthx = 32;
    const int blockWidthy = 32;

    const dim3 dimBlock(boost::math::gcd<unsigned>( width,blockWidthx),
                        boost::math::gcd<unsigned>(height,blockWidthy),
                        1);

    const dim3 dimGrid(width  / dimBlock.x,
                       height / dimBlock.y,
                       1);

    Mat<float,3,4>T_lrMat;

    TooN::Matrix<3>Rlr_TooN = T_lr.get_rotation().get_matrix();
    TooN::Vector<3>tlr_TooN = T_lr.get_translation();

    for(int r=0 ; r < 3; r++ )
    {
        for(int c=0; c < 3 ; c++)
        {
            T_lrMat(r,c) = Rlr_TooN(r,c);
        }
    }

    for(int r=0 ; r < 3; r++)
        T_lrMat(r,3) = tlr_TooN[r];


    cuRotatePoints<<<dimGrid,dimBlock>>>(vertex_src,
                                         vertex_dest,
                                         T_lrMat,
                                         stridef4,
                                         width,
                                         height);

}


__global__ void cufilterBilateral(float* d_image_in,
                                  float* d_image_out,
                                  const unsigned int stridef1,
                                  float sigma_s,
                                  float sigma_r,
                                  int width,
                                  int height,
                                  int2 filter_radius,
                                  float minVal)
{

    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float gs = sigma_s;
    float gr = sigma_r;

    if ( x < width && y < height )
    {
        float p = d_image_in[y*stridef1+x];
        float sum = 0;
        float sumw = 0;

        if( p >= minVal)
        {
            for(int r = -filter_radius.y; r <= filter_radius.y; r++ )
            {
                for(int c = -filter_radius.x; c <= filter_radius.x; c++ )
                {
                    if ( x+c >= 2 && x+c < width-2 && y+r >= 2 && y+r <height-2)
                    {
                        const float q = d_image_in[(y+r)*stridef1+(x+c)];

                        if(q >= minVal)
                        {
                            const float sd2 = r*r + c*c;
                            const float id  = p-q;
                            const float id2 = id*id;
                            const float sw  = __expf(-(sd2) / (2 * gs * gs));
                            const float iw  = __expf(-(id2) / (2 * gr * gr));
                            const float w   = sw*iw;

                            sumw += w;
                            sum  += w * q;
                        }
                    }
                }
            }

            if ( sumw )
                d_image_out[y*stridef1+x] = (float)sum/sumw;
            else
                d_image_out[y*stridef1+x] = 0.0f;
        }
        else
        {
            d_image_out[y*stridef1+x] = 0.0f;
        }


    }

}

void filterBilateral(float* d_image_in,
                     float* d_image_out,
                     const unsigned int stridef1,
                     float sigma_s,
                     float sigma_r,
                     int2 img_size,
                     int2 filter_radius,
                     float minVal)
{
    const int blockWidthx = 32;
    const int blockWidthy = 32;

    int width  = img_size.x;
    int height = img_size.y;

    const dim3 dimBlock(boost::math::gcd<unsigned>( width, blockWidthx),
                        boost::math::gcd<unsigned>(height, blockWidthy),
                        1);

    const dim3 dimGrid(width  / dimBlock.x,
                       height / dimBlock.y,
                       1);

    cufilterBilateral<<<dimGrid, dimBlock >>>(d_image_in,
                                              d_image_out,
                                              stridef1,
                                              sigma_s,
                                              sigma_r,
                                              width,
                                              height,
                                              filter_radius,
                                              minVal);

}

__global__ void cuDownsampleDepth(float* depth_in,
                                  const unsigned int stridefin,
                                  float* depth_out,
                                  const unsigned int stridefout,
                                  const unsigned int width_in,
                                  const unsigned int height_in,
                                  const unsigned int width_out,
                                  const unsigned int height_out,
                                  const float n_plane,
                                  const float f_plane)
{
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    depth_out[y*stridefout+x] = 0.0f;

    if ( x < width_out && y < height_out)
    {

        int x_in = 2 * x;
        int y_in = 2 * y;

        float depth_val = 0;
        int avg_count  = 0;

//        const int D = 5;

//        int center = depth_in[y_in*stridefin + x_in];

//        int x_mi = max(0, x_in - D/2) - x_in;
//        int y_mi = max(0, y_in - D/2) - y_in;

//        int x_ma = min(width_in, x_in -D/2+D) - x_in;
//        int y_ma = min(height_in, y_in -D/2+D) - y_in;

//        float sum = 0;
//        float wall = 0;

//        float sigma_color = 0.3;

//        float weights[] = {0.375f, 0.25f, 0.0625f} ;

//        for(int yi = y_mi; yi < y_ma; ++yi)
//        {
//            for(int xi = x_mi; xi < x_ma; ++xi)
//            {
//                float val = depth_in[(y_in + yi)*width_in+x_in + xi];

//                if (abs (val - center) < 3 * sigma_color)
//                {
//                    sum += val * weights[abs(xi)] * weights[abs(yi)];
//                    wall += weights[abs(xi)] * weights[abs(yi)];
//                }
//            }
//        }

//        //            dst.ptr (y)[x] = static_cast<int>(sum /wall);

//        depth_out[y*stridefout+x] = sum/wall;


        if ( x_in + 1 < width_in && y_in + 1 < height_in)
        {
            ///x = 0, y = 0
            float depth00  = depth_in[y_in*stridefin + x_in];
            if ( depth00 )
            {
                depth_val += depth00;
                avg_count +=1;
            }

            /// x = 0, y = 1
            float depth01  = depth_in[(y_in+1)*stridefin + x_in];
            if ( depth01 )
            {
                depth_val += depth01;
                avg_count +=1;
            }

            /// x = 1, y = 0
            float depth10  = depth_in[(y_in)*stridefin + x_in+1];
            if ( depth10 )
            {
                depth_val += depth10;
                avg_count +=1;
            }

            /// x = 1, y = 1
            float depth11  = depth_in[(y_in+1)*stridefin + x_in+1];
            if ( depth11 )
            {
                depth_val += depth11;
                avg_count +=1;
            }

            depth_val = depth_val / (float)avg_count;
            //if (! (depth_val > 0.4f && depth_val < 10.0f) )
            if (!( depth_val > n_plane && depth_val < f_plane))
                depth_val = 0.0f;

        }

        depth_out[y*stridefout+x] = depth_val;
    }

}


void downSampleDepth(float *depth_in,
                     const unsigned int stridefin,
                     float *depth_out,
                     const unsigned int stridefout,
                     const unsigned int width_in,
                     const unsigned int height_in,
                     const unsigned int width_out,
                     const unsigned int height_out,
                     const float n_plane,
                     const float f_plane)
{

    const int blockWidthx = 32;
    const int blockWidthy = 32;

    const dim3 dimBlock(boost::math::gcd<unsigned>( width_out,blockWidthx),
                        boost::math::gcd<unsigned>(height_out,blockWidthy),
                        1);

    const dim3 dimGrid(width_out  / dimBlock.x,
                       height_out / dimBlock.y,
                       1);

    cuDownsampleDepth<<<dimGrid, dimBlock >>>(depth_in,
                                              stridefin,
                                              depth_out,
                                              stridefout,
                                              width_in,
                                              height_in,
                                              width_out,
                                              height_out,
                                              n_plane,
                                              f_plane);


}

/// It seems to be working right now.
/// Look at this paper: Velodyne SLAM for setting the weights instead of
/// just averaging, do weighted averaging.
///
/// Seems to take about 0.6ms on my Toshiba
/// 770M laptop while the standard one cross product takes about 0.5ms
__global__ void cuComputeNormalsFromVertexRobust(float4* normal,
                                                 const float4* vertex,
                                                 const unsigned int stridef4,
                                                 const unsigned int width,
                                                 const unsigned int height)
{

    /// Always declare as int if you're going to do comparisons
    /// like x - 1 > 0
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    normal[y*stridef4+x] = make_float4(0.0f);

    float4 N_ru = make_float4(0.0f);
    float4 N_ul = make_float4(0.0f);
    float4 N_ld = make_float4(0.0f);
    float4 N_dr = make_float4(0.0f);

    float4 a ;
    float4 b ;
    float magaxb;

    if( x-1> 0 && x+1 < width && y-1>0 && y+1 < height )
    {

        const float4 Vc = vertex[(y+0)*stridef4+(x+0)];
        const float4 Vr = vertex[(y+0)*stridef4+(x+1)];
        const float4 Vu = vertex[(y+1)*stridef4+(x+0)];
        const float4 Vd = vertex[(y-1)*stridef4+(x+0)];
        const float4 Vl = vertex[(y+0)*stridef4+(x-1)];


        /// N_ru
        {

            if ( Vc.w == 1.0f && Vr.w == 1.0f && Vu.w == 1.0f )
            {
                a = Vr - Vc;
                b = Vu - Vc;

                const float3 axb_ru = make_float3(a.y*b.z - a.z*b.y,
                                                  a.z*b.x - a.x*b.z,
                                                  a.x*b.y - a.y*b.x);

                magaxb = length(axb_ru);

                if (magaxb)
                {
                    N_ru = make_float4(axb_ru.x/magaxb, axb_ru.y/magaxb, axb_ru.z/magaxb,1.0f);
                }
            }
        }

        /// N_ul
        {

            if ( Vc.w == 1.0f && Vu.w == 1.0f && Vl.w == 1.0f  )
            {
                a = Vu - Vc;
                b = Vl - Vc;

                const float3 axb_ul = make_float3(a.y*b.z - a.z*b.y,
                                                  a.z*b.x - a.x*b.z,
                                                  a.x*b.y - a.y*b.x);

                magaxb = length(axb_ul);

                if (magaxb)
                {
                    N_ul = make_float4(axb_ul.x/magaxb, axb_ul.y/magaxb, axb_ul.z/magaxb,1.0f);
                }

            }
        }

        /// N_ld
        {
            if ( Vc.w == 1.0f && Vd.w == 1.0f && Vl.w == 1.0f )
            {
                a = Vl - Vc;
                b = Vd - Vc;

                const float3 axb_ld = make_float3(a.y*b.z - a.z*b.y,
                                                  a.z*b.x - a.x*b.z,
                                                  a.x*b.y - a.y*b.x);

                magaxb = length(axb_ld);

                if (magaxb)
                {
                    N_ld = make_float4(axb_ld.x/magaxb, axb_ld.y/magaxb, axb_ld.z/magaxb,1.0f);
                }

            }
        }

        /// N_dr
        {
            if ( Vc.w == 1.0f && Vd.w == 1.0f && Vr.w == 1.0f )
            {
                a = Vd - Vc;
                b = Vr - Vc;

                const float3 axb_dr = make_float3(a.y*b.z - a.z*b.y,
                                                  a.z*b.x - a.x*b.z,
                                                  a.x*b.y - a.y*b.x);

                magaxb = length(axb_dr);

                if (magaxb)
                {
                    N_dr = make_float4(axb_dr.x/magaxb, axb_dr.y/magaxb, axb_dr.z/magaxb,1.0f);
                }

            }
        }


        float4 N = make_float4(0.0f);

        N = (N_ru + N_ul + N_dr + N_ld)/4;

        if ( N.w > 0 )
        {
            float3 n = make_float3(N.x,N.y,N.z);

            N = N / length(n);

            N.w = 1;

            normal[y*stridef4+x] = N;
        }
    }
}



void ComputeNormalsFromVertexRobust(float4* normal,
                              float4* vertex,
                              const unsigned int stridef4,
                              const unsigned int width,
                              const unsigned int height)
{
        const int blockWidthx = 32;
        const int blockWidthy = 32;

        const dim3 dimBlock(boost::math::gcd<unsigned>( width,blockWidthx),
                            boost::math::gcd<unsigned>(height,blockWidthy),
                            1);

        const dim3 dimGrid(width  / dimBlock.x,
                           height / dimBlock.y,
                           1);

        cuComputeNormalsFromVertexRobust<<<dimGrid, dimBlock >>> (normal,
                                            vertex,
                                            stridef4,
                                            width,
                                            height);
}


__global__ void cuComputeNormalsFromVertex(float4* normal,
                                           const float4* vertex,
                                           const unsigned int stridef4,
                                           const unsigned int width,
                                           const unsigned int height)
{
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    normal[y*stridef4+x] = make_float4(0.0f);

    if( x < width && y < height )
    {
        if( x+1 < width && y+1 < height )
        {
            const float4 Vc = vertex[(y+0)*stridef4+(x+0)];
            const float4 Vr = vertex[(y+0)*stridef4+(x+1)];
            const float4 Vu = vertex[(y+1)*stridef4+(x+0)];

            if ( Vc.w == 1.0f && Vr.w == 1.0f && Vu.w == 1.0f )
            {
                const float4 a = Vr - Vc;
                const float4 b = Vu - Vc;

                /// Pretty useless way of doing it..
                /// Improve by weighted avergaging of neighbours
                const float3 axb = make_float3(a.y*b.z - a.z*b.y,
                                               a.z*b.x - a.x*b.z,
                                               a.x*b.y - a.y*b.x);

                const float magaxb = length(axb);

                if (magaxb)
                {
                    const float4 N = make_float4(axb.x/magaxb, axb.y/magaxb, axb.z/magaxb,1.0f);
                    normal[y*stridef4+x] = N;
                }                
            }            
        }        
    }
}


void ComputeNormalsFromVertex(float4* normal,
                              float4* vertex,
                              const unsigned int stridef4,
                              const unsigned int width,
                              const unsigned int height)
{

    const int blockWidthx = 32;
    const int blockWidthy = 32;

    const dim3 dimBlock(boost::math::gcd<unsigned>( width,blockWidthx),
                        boost::math::gcd<unsigned>(height,blockWidthy),
                        1);

    const dim3 dimGrid(width  / dimBlock.x,
                       height / dimBlock.y,
                       1);

    cuComputeNormalsFromVertexRobust<<<dimGrid, dimBlock >>> (normal,
                                        vertex,
                                        stridef4,
                                        width,
                                        height);

//    cuComputeNormalsFromVertex<<<dimGrid, dimBlock >>> (normal,
//                                        vertex,
//                                        stridef4,
//                                        width,
//                                        height);

}




__global__ void cuCopyNormals(float4 *predNormals,
                              float4 *inNormals,
                              const unsigned int stridef4,
                              const unsigned int width,
                              const unsigned int height)
{
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if ( x < width && y < height )
        predNormals[y*stridef4+x] = inNormals[y*stridef4+x];

}

void CopyNormals(float4 *predNormals,
                 float4 *inNormals,
                 const unsigned int stridef4,
                 const unsigned int width,
                 const unsigned int height)
{

    const int blockWidthx = 32;
    const int blockWidthy = 32;

    const dim3 dimBlock(boost::math::gcd<unsigned>( width,blockWidthx),
                        boost::math::gcd<unsigned>(height,blockWidthy),
                        1);

    const dim3 dimGrid(width  / dimBlock.x,
                       height / dimBlock.y,
                       1);

    cuCopyNormals<<<dimGrid, dimBlock >>> (predNormals,
                                           inNormals,
                                           stridef4,
                                           width,
                                           height);

}

__global__ void cuInterpolateNormals(float4* inNormals,
                                     const unsigned stridef4In,
                                     float4* outNormals,
                                     const unsigned stridef4Out,
                                     const unsigned int widthIn,
                                     const unsigned int heightIn,
                                     const unsigned int widthOut,
                                     const unsigned int heightOut)
{
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if ( x < widthOut && y < heightOut)
    {

        int xIn = 2 * x;
        int yIn = 2 * y;

        float4 normal_avg = make_float4(0.0f);
        int avg_count = 0;

        if ( xIn + 1 < widthIn && yIn + 1 < heightIn)
        {
            ///x = 0, y = 0
            float4 normal00  = inNormals[yIn*stridef4In + xIn];
            if ( normal00.w > 0 )
            {
                normal_avg += normal00;
                avg_count +=1;
            }

            /// x = 0, y = 1
            float4 normal01  = inNormals[(yIn+1)*stridef4In + xIn];
            if ( normal01.w > 0 )
            {
                normal_avg += normal01;
                avg_count +=1;
            }

            /// x = 1, y = 0
            float4 normal10  = inNormals[(yIn)*stridef4In + xIn+1];
            if ( normal10.w > 0 )
            {
                normal_avg += normal10;
                avg_count +=1;
            }

            /// x = 1, y = 1
            float4 normal11  = inNormals[(yIn+1)*stridef4In + xIn+1];
            if ( normal11.w > 0 )
            {
                normal_avg += normal11;
                avg_count +=1;
            }

            normal_avg = normal_avg / (float)avg_count;

            float3 normal_avg3 = make_float3(normal_avg.x,normal_avg.y,normal_avg.z);

            normal_avg = make_float4(normalize(normal_avg3),1.0f);
        }

        outNormals[y*stridef4Out+x] = normal_avg;
    }

}


void InterpolateNormals(float4* inNormals,
                        const unsigned stridef4in,
                        float4* outNormals,
                        const unsigned stridef4out,
                        const unsigned int widthIn,
                        const unsigned int heightIn,
                        const unsigned int widthOut,
                        const unsigned int heightOut)
{

    const unsigned int blockWidthx = 32;
    const unsigned int blockWidthy = 32;

    const dim3 dimBlock(boost::math::gcd<unsigned>( widthOut,blockWidthx),
                        boost::math::gcd<unsigned>(heightOut,blockWidthy),
                        1);

    const dim3 dimGrid(widthOut  / dimBlock.x,
                       heightOut / dimBlock.y,
                       1);

    cuInterpolateNormals<<<dimGrid, dimBlock>>>(inNormals,
                                                stridef4in,
                                                outNormals,
                                                stridef4out,
                                                widthIn,
                                                heightIn,
                                                widthOut,
                                                heightOut);

}



__global__ void cuComputeVertexFromDepth(  float* depth,
                                           const unsigned int stridef1,
                                           float4* vertex,
                                           const unsigned int stridef4,
                                           const float2 fl,
                                           const float2 pp,
                                           const unsigned int width,
                                           const unsigned int height,
                                           const float near_plane,
                                           const float far_plane)
{

    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    vertex[y*stridef4+x] = make_float4(0.0f,0.0f,0.0f,0.0f);

    if ( x < width && y < height )
    {
        float depthval = depth[y*stridef1+x];

        if ( depthval > near_plane && depthval < far_plane )
        {
            vertex[y*stridef4+x] = make_float4(depthval*((float)x-pp.x)/fl.x,
                                               depthval*((float)y-pp.y)/fl.y,
                                               depthval,
                                               1.0f);
        }
        else
        {
            vertex[y*stridef4+x] = make_float4(0.0f,0.0f,0.0f,0.0f);
        }
    }

}


void ComputeVertexFromDepth(float* depth,
                               const unsigned int stridef1,
                               float4* vertex,
                               const unsigned int stridef4,
                               const unsigned int width,
                               const unsigned int height,
                               const float2 fl,
                               const float2 pp,
                               const float near_plane,
                               const float far_plane)
{

    const unsigned int blockWidthx = 32;
    const unsigned int blockWidthy = 32;

    const dim3 dimBlock(boost::math::gcd<unsigned>( width,blockWidthx),
                        boost::math::gcd<unsigned>(height,blockWidthy),
                        1);

    const dim3 dimGrid(width  / dimBlock.x,
                       height / dimBlock.y,
                       1);

    cuComputeVertexFromDepth<<<dimGrid, dimBlock>>>(depth,
                                                    stridef1,
                                                    vertex,
                                                    stridef4,
                                                    fl,
                                                    pp,
                                                    width,
                                                    height,
                                                    near_plane,
                                                    far_plane);

}

__global__ void cuComputeDepthFromVertex(  float* depth,
                                           const unsigned int stridef1,
                                           const float4* vertex,
                                           const unsigned int stridef4,
                                           const unsigned int width,
                                           const unsigned int height,
                                           const float2 fl,
                                           const float2 pp)
{

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int index4 = (x + y*stridef4);


    const float4 v = vertex[index4];

    if(v.w > 0 && v.z > 0)
    {
        float _x_d = (v.x*fl.x/v.z) + pp.x;
        float _y_d = (v.y*fl.y/v.z) + pp.y;

        int x_d = (int)(_x_d + 0.5f);
        int y_d = (int)(_y_d + 0.5f);

        if ( x_d >= 0 && x_d < width && y_d >= 0 && y_d < height )
        {
            int index = (x_d + y_d*stridef1);
            depth[index] = v.z;
        }
    }
}



void ComputeDepthFromVertex(float4* vertex,
                            const unsigned int stridef4,
                            float* depth,
                            const unsigned int stridef1,
                            const unsigned int width,
                            const unsigned int height,
                            const float2 fl,
                            const float2 pp)
{

    const unsigned int blockWidthx = 32;
    const unsigned int blockWidthy = 32;

    const dim3 dimBlock(boost::math::gcd<unsigned>( width,blockWidthx),
                        boost::math::gcd<unsigned>(height,blockWidthy),
                        1);

    const dim3 dimGrid(width  / dimBlock.x,
                       height / dimBlock.y,
                       1);

    cuComputeDepthFromVertex<<<dimGrid, dimBlock>>>(depth,
                                                    stridef1,
                                                    vertex,
                                                    stridef4,
                                                    width,
                                                    height,
                                                    fl,
                                                    pp);

}


}
