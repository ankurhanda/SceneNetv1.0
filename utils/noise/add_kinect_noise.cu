#include "add_kinect_noise.h"


#include<thrust/random.h>
#include<thrust/transform.h>
#include<thrust/device_vector.h>
#include<thrust/transform.h>
#include<thrust/iterator/counting_iterator.h>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/tuple.h>
#include <boost/math/common_factor_rt.hpp>
#include <iu/iucore.h>
#include <iu/iumath.h>

#include <thrust/random/normal_distribution.h>


typedef thrust::device_vector<float4>::iterator Float4Iterator;
typedef thrust::tuple<Float4Iterator, Float4Iterator> VertexNormalIteratorTuple;
typedef thrust::zip_iterator<VertexNormalIteratorTuple> ZipIterator;
typedef thrust::tuple<float4, float4> VertexNormalTuple;


/// Texture be declared in the .cu file - otherwise compilation errors


namespace noise{

texture<float, 2, cudaReadModeElementType> ref_depth_tex;
cudaChannelFormatDesc channelfloat1  = cudaCreateChannelDesc<float>();

__host__ __device__
unsigned int hash(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

struct ccd_camera_noise
{
    const float sigma_s_red;
    const float sigma_s_green;
    const float sigma_s_blue;

    const float sigma_c_red;
    const float sigma_c_green;
    const float sigma_c_blue;

    const float scale;

    ccd_camera_noise(float _sigma_s_red,
                    float _sigma_s_green,
                    float _sigma_s_blue,
                    float _sigma_c_red,
                    float _sigma_c_green,
                    float _sigma_c_blue,
                    float _scale) : sigma_s_red(_sigma_s_red),
       sigma_s_green(_sigma_s_green),
       sigma_s_blue(_sigma_s_blue),
       sigma_c_red(_sigma_c_red),
       sigma_c_green(_sigma_c_green),
       sigma_c_blue(_sigma_c_blue),
       scale(_scale)
       {}

  __host__ __device__  float4 operator()(const float4& val, const unsigned int& thread_id )
  {

      float4 noisy_pix;

      clock_t start_time = clock();

      unsigned int seed = hash(thread_id) + start_time;

      thrust::minstd_rand rng(seed);

      noisy_pix.x = val.x/scale;
      noisy_pix.y = val.y/scale;
      noisy_pix.z = val.z/scale;

      thrust::random::/*experimental::*/normal_distribution<float> red_pnoise  (0.0f,sqrt(val.x)*sigma_s_red  );
      thrust::random::/*experimental::*/normal_distribution<float> green_pnoise(0.0f,sqrt(val.y)*sigma_s_green);
      thrust::random::/*experimental::*/normal_distribution<float> blue_pnoise (0.0f,sqrt(val.z)*sigma_s_blue );

      thrust::random::/*experimental::*/normal_distribution<float> red_cnoise   (0.0f,sigma_c_red  );
      thrust::random::/*experimental::*/normal_distribution<float> green_cnoise (0.0f,sigma_c_green);
      thrust::random::/*experimental::*/normal_distribution<float> blue_cnoise  (0.0f,sigma_c_blue );

      noisy_pix.x = noisy_pix.x  + red_pnoise(rng)   + red_cnoise(rng);
      noisy_pix.y = noisy_pix.y  + green_pnoise(rng) + green_cnoise(rng);
      noisy_pix.z = noisy_pix.z  + blue_pnoise(rng)  + blue_cnoise(rng);

      noisy_pix.w = 1.0f;

      return noisy_pix;
  }
};


void launch_add_camera_noise(float4* img_array, float4* noisy_image, float4 sigma_s, float4 sigma_c,
                             const unsigned int stridef4, const unsigned int height, float scale)
{
    thrust::device_ptr<float4>img_src(img_array);

    thrust::device_ptr<float4>img_dest(noisy_image);

    thrust::transform(img_src,img_src + stridef4*height, thrust::make_counting_iterator(0), img_dest,
                                                                  ccd_camera_noise(sigma_s.x,
                                                                                   sigma_s.y,
                                                                                   sigma_s.z,
                                                                                   sigma_c.x,
                                                                                   sigma_c.y,
                                                                                   sigma_c.z,
                                                                                   scale)
                                                                                   );
}



struct add_kinect_noise
{
    float focal_length;
    float theta_1;
    float theta_2;

    float z1;
    float z2;
    float z3;

    add_kinect_noise(float _focal_length,
                 float _theta_1,
                 float _theta_2,
                 float _z1,
                 float _z2,
                 float _z3):
        focal_length(_focal_length),
        theta_1(_theta_1),
        theta_2(_theta_2),
        z1(_z1),
        z2(_z2),
        z3(_z3){}

  __host__ __device__  float4 operator()(const VertexNormalTuple& vertex_normal_tuple,
                                         const unsigned int& thread_id
                                        )
  {
      float4 noisy_3D;
      float4 noisy_lateral = make_float4(0);
      float4 noisy_axial  = make_float4(0);

      /// Get the seed up
      clock_t start_time = clock();
      unsigned int seed = hash(thread_id) + start_time;

      thrust::minstd_rand rng(seed);

      const float4 point3D  = thrust::get<0>(vertex_normal_tuple);
      const float4 normal3D = thrust::get<1>(vertex_normal_tuple);

      float depth = point3D.z;
      float my_pi = 22.0f/7.0f;


      /// Subtract the 1 from the dot product; points are represented in homogeneous form with point.w =1
      float dot_prod = dot(normal3D,point3D)-1 ;

      /// xyz of point
      float4 point3D_3 = point3D;
      point3D_3.w = 0;
      float norm_point = length(point3D_3);

      /// negative sign to indicate the position vector of the point starts from the point
      float theta = fabs(acosf(-dot_prod/norm_point));

      float sigma_theta = theta_1 + theta_2*(theta)/(my_pi/2-theta);

      sigma_theta = sigma_theta*(depth)/focal_length;

      thrust::random::normal_distribution<float> normal_noise(0,sigma_theta);

      noisy_lateral.x = point3D.x + normal_noise(rng)*normal3D.x;
      noisy_lateral.y = point3D.y + normal_noise(rng)*normal3D.y;
      noisy_lateral.z = point3D.z + normal_noise(rng)*normal3D.z;

      noisy_3D = noisy_lateral + noisy_axial;

      if ( fabs(my_pi/2 - theta ) <= 8.0/180.0f*my_pi)
      {
          noisy_3D.z = 0.0f;
      }
      noisy_3D.w = 1.0f;

      return noisy_3D;
  }

};


void launch_add_kinect_noise(float4* points3D,
                             float4* normals3D,
                             float4* noisy_points,
                             const unsigned int stridef4,
                             const unsigned int height,
                             float focal_length,
                             float theta_1,
                             float theta_2,
                             float z1,
                             float z2,
                             float z3)
{
    thrust::device_ptr<float4>points_src(points3D);
    thrust::device_ptr<float4>normals_src(normals3D);

    thrust::device_ptr<float4>points_dest(noisy_points);

    ZipIterator vertex_normal_tuple(thrust::make_tuple(points_src, normals_src));


    thrust::transform(vertex_normal_tuple,
                      vertex_normal_tuple+stridef4*height,
                      thrust::make_counting_iterator(0),
                      points_dest,add_kinect_noise(focal_length,
                                                   theta_1,
                                                   theta_2,
                                                   z1,
                                                   z2,
                                                   z3));
}



struct colour_from_normals{
    colour_from_normals(){};

    __host__ __device__ float4 operator()(const float4& normal)
    {
        float4 colour;

        colour.x = normal.x*0.5f+0.5f;
        colour.y = normal.y*0.5f+0.5f;
        colour.z = normal.z*0.5f+0.5f;
        colour.w = 1;

        return colour;
    }

};



void launch_colour_from_normals(float4* normals,
                                float4* colour,
                                const unsigned int stridef4,
                                const unsigned int height)
{

    thrust::device_ptr<float4>normal_src(normals);

    thrust::device_ptr<float4>colour_dest(colour);

    thrust::transform(normal_src,normal_src + stridef4*height, colour_dest,
                      colour_from_normals());

}



struct gaussian_rand{

    float sigma;

    gaussian_rand(float _sigma):sigma(_sigma){}

    __host__ __device__  float2 operator()( float2 point,
                                            const unsigned int& thread_id
                                         )
    {
        float2 noise;

        clock_t start_time = clock();

        unsigned int seed = hash(thread_id) + start_time;

        thrust::minstd_rand rng(seed);
        thrust::random::normal_distribution<float>randn(0,1);


        noise.x = randn(rng)/sigma;
        noise.y = randn(rng)/sigma;

        return noise;
    }
};


void gaussian_shifts(float2* tex_coods,
                     const unsigned int stridef2,
                     const unsigned int height,
                     const float _sigma)
{

    thrust::device_ptr<float2>coords_src(tex_coods);

    thrust::transform(coords_src,coords_src+stridef2*height,
                      thrust::make_counting_iterator(0),
                      coords_src,
                      gaussian_rand(_sigma));

}


struct gaussian_depth_noise{

    float sigma;

    gaussian_depth_noise(){}

    __host__ __device__  float operator()( float& depth,
                                            const unsigned int& thread_id
                                         )
    {
        float noisy_depth;

        clock_t start_time = clock();

        unsigned int seed = hash(thread_id) + start_time;

        thrust::minstd_rand rng(seed);
        thrust::random::normal_distribution<float>  randn(0,1);

        noisy_depth = (35130/round(35130/round(depth*100) + randn(rng)*(1.0/10.0f) + 0.5))/100;

        return noisy_depth;
    }
};


void add_depth_noise_barronCVPR2013(float* depth_copy,
                               const int stridef1,
                               const int height)
{

    thrust::device_ptr<float>depth_src(depth_copy);

    thrust::transform(depth_src,
                      depth_src+stridef1*height,
                      thrust::make_counting_iterator(0),
                      depth_src,
                      gaussian_depth_noise());




}

__global__ void get_z_coordinate_only(float4* vertex_with_noise,
                                      const unsigned int stridef4,
                                      float* noisy_depth,
                                      const unsigned int stridef1)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    noisy_depth[y*stridef1+x] = vertex_with_noise[y*stridef4+x].z;

}



void launch_get_z_coordinate_only(float4* vertex_with_noise,
                                  const unsigned int stridef4,
                                  const unsigned int width,
                                  const unsigned int height,
                                  float* noisy_depth,
                                  const unsigned int stridef1
                                  )
{
    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    get_z_coordinate_only<<<grid, block>>>(vertex_with_noise,
                                           stridef4,
                                           noisy_depth,
                                           stridef1);
}



__global__ void convert_depth2png (float* noisy_depth,
                                  const unsigned int stridef1,
                                   uint16_t* noisy_depth_png,
                                   const unsigned int strideu16)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    noisy_depth_png[y*strideu16+x] = (u_int16_t)(noisy_depth[y*stridef1+x]*5000);
//    noisy_depth_png[y*strideu16+x] = (u_int16_t)(noisy_depth[y*stridef1+x]*500);
}






void  launch_convert_depth2png(float* noisy_depth,
                               const unsigned int stridef1,
                               u_int16_t* noisy_depth_png,
                               const unsigned int strideu16,
                               const unsigned int width,
                               const unsigned int height)
{

    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    convert_depth2png<<<grid, block>>>(noisy_depth,
                                       stridef1,
                                       noisy_depth_png,
                                       strideu16);
}






__device__ float Interpolate(float x0, float x1, float alpha)
{
   return x0 * (1 - alpha) + alpha * x1;
}

__global__ void cu_generateSmoothNoise(float* smoothNoise,
                                  const unsigned int stridef1,
                                  float* baseNoise,
                                  const float samplePeriod,
                                  const float sampleFrequency,
                                  unsigned int width,
                                  unsigned int height)
{

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    //calculate the horizontal sampling indices
    int sample_i0 = (x / (int)samplePeriod) * (int)samplePeriod;
    int sample_i1 = (sample_i0 + (int)samplePeriod) % width; //wrap around
    float horizontal_blend = (x - sample_i0) * sampleFrequency;

    //calculate the vertical sampling indices
    int sample_j0 = (y / (int)samplePeriod) * (int)samplePeriod;
    int sample_j1 = (sample_j0 + (int)samplePeriod) % height; //wrap around
    float vertical_blend = (y - sample_j0) * sampleFrequency;

    //blend the top two corners
    float top = Interpolate(baseNoise[sample_i0+stridef1*sample_j0],
                            baseNoise[sample_i1+stridef1*sample_j0],
                            horizontal_blend);

    //blend the bottom two corners
    float bottom = Interpolate(baseNoise[sample_i0+stridef1*sample_j1],
                               baseNoise[sample_i1+stridef1*sample_j1],
                               horizontal_blend);


    smoothNoise[x+y*stridef1] = Interpolate(top, bottom, vertical_blend);

}

void generate_smooth_noise(iu::ImageGpu_32f_C1* smoothNoise,
                           iu::ImageGpu_32f_C1* baseNoise,
                           const float samplePeriod,
                           const float sampleFrequency,
                           const unsigned int width,
                           const unsigned int height)
{

    dim3 blockdim(boost::math::gcd<unsigned>(width, 32), boost::math::gcd<unsigned>(height, 32), 1);
    dim3 griddim( width / blockdim.x, height / blockdim.y);

    cu_generateSmoothNoise<<<griddim,blockdim>>>(smoothNoise->data(),
                                           smoothNoise->stride(),
                                           baseNoise->data(),
                                           samplePeriod,
                                           sampleFrequency,
                                           smoothNoise->width(),
                                           smoothNoise->height());


}


__global__ void cu_addNoise2Vertex(float4* vertex,
                                   float4* normals,
                                   float4* vertex_with_noise,
                                   const unsigned int stridef4,
                                   float* noise,
                                   const unsigned int stridef1,
                                   const unsigned int width,
                                   const unsigned int height)
{

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if ( x > 0 && x < width && y > 0 && y < height )
    {
        int ind4 = x+y*stridef4;
        int ind1 = x+y*stridef1;

        vertex_with_noise[ind4] = vertex[ind4] + noise[ind1]*make_float4(normals[ind4].x,
                                                                         normals[ind4].y,
                                                                         normals[ind4].z,0);
    }

}


void add_noise2vertex(iu::ImageGpu_32f_C4* vertex,
                      iu::ImageGpu_32f_C4* normals,
                      iu::ImageGpu_32f_C4* vertex_with_noise,
                      iu::ImageGpu_32f_C1* perlinNoise)
{

    const int2 imageSize = make_int2(vertex->width(), vertex->height());
    const int w = imageSize.x;
    const int h = imageSize.y;

    dim3 blockdim(boost::math::gcd<unsigned>(w, 32), boost::math::gcd<unsigned>(h, 32), 1);
    dim3 griddim( w / blockdim.x, h / blockdim.y);


    cu_addNoise2Vertex<<<griddim,blockdim>>>(vertex->data(),
                       normals->data(),
                       vertex_with_noise->data(),
                       vertex->stride(),
                       perlinNoise->data(),
                       perlinNoise->stride(),
                       perlinNoise->width(),
                       perlinNoise->height());


}


__global__ void cu_verts2depth(
        float* d_depth,
        float4* d_vert,
        const float2 pp, const float2 fl,
        size_t stridef1, size_t stridef4)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int index4 = (x + y*stridef4);

    const float4 v = d_vert[index4];

    if(v.w > 0 && v.z > 0)// && v.z < 1000)
    {
        float _x_d = (v.x*fl.x/v.z) + pp.x;
        float _y_d = (v.y*fl.y/v.z) + pp.y;
        int x_d = (int)(_x_d + 0.5f);
        int y_d = (int)(_y_d + 0.5f);
        int index = (x_d + y_d*stridef1);
        d_depth[index] = v.z;
    }
}

void convertVerts2Depth(iu::ImageGpu_32f_C1* depth, iu::ImageGpu_32f_C4* vertex, float2 pp, float2 fl)
{
    const int2 imageSize = make_int2(depth->width(), depth->height());
    const size_t stridef1 = depth->stride();
    const size_t stridef4 = vertex->stride();
    const int w = imageSize.x;
    const int h = imageSize.y;

    dim3 blockdim(boost::math::gcd<unsigned>(w, 32), boost::math::gcd<unsigned>(h, 32), 1);
    dim3 griddim( w / blockdim.x, h / blockdim.y);

    cu_verts2depth<<<griddim, blockdim>>>(depth->data(),
                                          vertex->data(),
                                          pp, fl,
                                          stridef1, stridef4);
}

void uploadTexture2CUDA(float *depth,
                        unsigned int pitchf1,
                        unsigned int width,
                        unsigned int height)
{
    /// set texture parameters
    ref_depth_tex.normalized      = false;                  // access with normalized texture coordinates
    ref_depth_tex.filterMode      = cudaFilterModeLinear;   // linear interpolation
    ref_depth_tex.addressMode[0]  = cudaAddressModeClamp;   // clamp texture coordinates
    ref_depth_tex.addressMode[1]  = cudaAddressModeClamp;

    /// Bind the texture
    cudaBindTexture2D(NULL,
                      &ref_depth_tex,
                      depth,
                      &channelfloat1,
                      width,
                      height,
                      pitchf1);
}

__global__ void cu_warpImage(float* d_depth,
                             unsigned int stridef1,
                             float2* tex_coords,
                             unsigned int stridef2,
                             unsigned int width,
                             unsigned int height)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    const int indexf1 = (x + y*stridef1);
    const int indexf2 = (x + y*stridef2);

    const float2 v = tex_coords[indexf2];

    const float x_ = (float)x;// + v.x;
    const float y_ = (float)y;// + v.y;

    if( x_ >=0 && x_ < (float)width && y_ >= 0 && y_ < (float)height )
    {
        d_depth[indexf1] = tex2D(ref_depth_tex,x+0.5f,y+0.5f);
    }
}



void  warpImage(float* depth_new,
                unsigned int stridef1,
                float2* tex_coords,
                unsigned int stridef2,
                unsigned int width,
                unsigned int height)
{
    dim3 blockdim(boost::math::gcd<unsigned>(width, 32), boost::math::gcd<unsigned>(height, 32), 1);
    dim3 griddim( width / blockdim.x, height / blockdim.y);

    cu_warpImage<<<griddim, blockdim>>>(depth_new,
                                        stridef1,
                                        tex_coords,
                                        stridef2,
                                        width,
                                        height);

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

    cuComputeNormalsFromVertex<<<dimGrid, dimBlock >>> (normal,
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
