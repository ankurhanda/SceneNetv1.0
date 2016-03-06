#include <iostream>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <iu/iuio.h>
#include <iu/iucore.h>
#include <cvd/image.h>
#include <cvd/image_io.h>
#include <cuda.h>
#include <boost/thread.hpp>
#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>
#include <pangolin/display.h>
#include <pangolin/plotter.h>

#include <boost/thread.hpp>
#include <pangolin/simple_math.h>

//#include <gvars3/default.h>
//#include <gvars3/gvars3.h>

#include "VaFRIC/VaFRIC.h"

//#include <icarus/icarus.h>

#include <iu/iuio.h>
#include <iu/iumath.h>
#include <iu/iufilter.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utils/noise/add_kinect_noise.h"

#include "math/aux_math.h"
//#include "rendering/openglrendering.h"

#include "utils/noise/noise.h"
#include "utils/noise/perlin.h"
#include "utils/noise/gaussian.h"

using namespace pangolin;
using namespace std;
using namespace CVD;
using namespace TooN;
//using namespace GVars3;

void GPUMemory()
{
    long unsigned int uCurAvailMemoryInBytes;
    long unsigned int uTotalMemoryInBytes;
    int nNoOfGPUs;

    CUresult result;
    CUdevice device;
    CUcontext context;

    cuDeviceGetCount( &nNoOfGPUs ); // Get number of devices supporting CUDA
    for( int nID = 0; nID < nNoOfGPUs; nID++ )
    {
        cuDeviceGet( &device, nID ); // Get handle for device
        cuCtxCreate( &context, 0, device ); // Create context
        result = cuMemGetInfo( &uCurAvailMemoryInBytes, &uTotalMemoryInBytes );
        if( result == CUDA_SUCCESS )
        {
            printf( "Device: %d\nTotal Memory: %ld MB, Free Memory: %ld MB\n",
                    nID,
                    uTotalMemoryInBytes / ( 1024 * 1024 ),
                    uCurAvailMemoryInBytes / ( 1024 * 1024 ));
        }
        cuCtxDetach( context ); // Destroy context
    }
}




const float invalid_disp_ = 99999999.9;

////// filter disparity with a 9x9 correlation window
void filterDisp(const cv::Mat& disp, cv::Mat& out_disp, cv::Mat& dot_pattern_)
{
    const int size_filt_ = 9;

    // initialize filter matrices for simulated disparity
    cv::Mat weights_ = cv::Mat(size_filt_, size_filt_, CV_32FC1);

    for (int x = 0; x < size_filt_; ++x)
    {
        float *weights_i = weights_.ptr<float>(x);

        for (int y = 0; y < size_filt_; ++y)
        {
            int tmp_x = x - size_filt_ / 2;
            int tmp_y = y - size_filt_ / 2;

            if (tmp_x != 0 && tmp_y != 0)
                weights_i[y] = 1.0 / ((1.2*(float)tmp_x)*(1.2*(float)tmp_x) + (1.2*(float)tmp_y)*(1.2*(float)tmp_x));
            else
                weights_i[y] = 1.0;
        }
    }

    cv::Mat fill_weights_ = cv::Mat(size_filt_, size_filt_, CV_32FC1);
    for (int x = 0; x < size_filt_; ++x){
        float *weights_i = fill_weights_.ptr<float>(x);
        for (int y = 0; y < size_filt_; ++y){
            int tmp_x = x - size_filt_ / 2;
            int tmp_y = y - size_filt_ / 2;
            if (std::sqrt(tmp_x*tmp_x + tmp_y*tmp_y) < 3.1)
                weights_i[y] = 1.0 / (1.0 + tmp_x*tmp_x + tmp_y*tmp_y);
            else
                weights_i[y] = -1.0;
        }
    }

    const float window_inlier_distance_ = 0.1;

//    cv::Mat dot_pattern_ = cv::imread("../data/kinect-pattern_3x3.png", 0);

    cv::Mat interpolation_map = cv::Mat::zeros(disp.rows, disp.cols, CV_32FC1);

    cv::Mat noise_field;
    noise_field = cv::Mat::zeros(disp.rows, disp.cols, CV_32FC1);

//    float perlin_scale = 0.2;
//    render_kinect::Noise* noise_gen_ = new render_kinect::PerlinNoise( 640, 480, perlin_scale);
//    noise_gen_->generateNoiseField(noise_field);

//    float mean = 0.0;
//    float std  = 0.15;
//    render_kinect::Noise* noise_gen_  = new render_kinect::GaussianNoise( 640, 480, mean, std);
//    noise_gen_->generateNoiseField(noise_field);

    // mysterious parameter
    float noise_smooth = 1.5;

    // initialise output arrays
    out_disp = cv::Mat(disp.rows, disp.cols, disp.type());
    out_disp.setTo(invalid_disp_);


    // determine filter boundaries
    unsigned lim_rows = std::min(disp.rows - size_filt_, dot_pattern_.rows - size_filt_);
    unsigned lim_cols = std::min(disp.cols - size_filt_, dot_pattern_.cols - size_filt_);
    int center = size_filt_ / 2.0;
    for (unsigned r = 0; r < lim_rows; ++r)
    {
        const float* disp_i = disp.ptr<float>(r + center);

        const float* dots_i = dot_pattern_.ptr<float>(r + center);

        float* out_disp_i = out_disp.ptr<float>(r + center);


        float* noise_i = noise_field.ptr<float>((int)((r + center) / noise_smooth));

        // window shifting over disparity image
        for (unsigned c = 0; c < lim_cols; ++c)
        {
            if (dots_i[c + center] > 0 && disp_i[c + center] < invalid_disp_)
            {
                cv::Rect roi = cv::Rect(c, r, size_filt_, size_filt_);
                cv::Mat window = disp(roi);
                cv::Mat dot_win = dot_pattern_(roi);

                // check if we are at a occlusion boundary without valid disparity values
                // return value not binary but between 0 or 255
                cv::Mat valid_vals = (window < invalid_disp_);
                cv::Mat valid_dots;

                cv::bitwise_and(valid_vals, dot_win, valid_dots);

                cv::Scalar n_valids = cv::sum(valid_dots) / 255.0;
                cv::Scalar n_thresh = cv::sum(dot_win) / 255.0;

                // only add depth value at center of window if there are more
                // valid disparity values than 2/3 of the number of dots
                if (n_valids(0) > n_thresh(0) / 1.2)
                {
                    // compute mean only over the valid values of disparities in that window
                    cv::Scalar mean = cv::mean(window, valid_vals);

                    // weighted deviation from mean
                    cv::Mat diffs = cv::abs(window - mean);
                    cv::multiply(diffs, weights_, diffs);

                    // get valid values that fall on dot pattern
                    cv::Mat valids = (diffs < window_inlier_distance_);
                    cv::bitwise_and(valids, valid_dots, valid_dots);

                    n_valids = cv::sum(valid_dots) / 255.0;

                    // only add depth value at center of window if there are more
                    // valid disparity values than 2/3 of the number of dots
                    if (n_valids(0) > n_thresh(0) / 1.2)
                    {
                        float accu = window.at<float>(center, center);

                        assert(accu < invalid_disp_);

                        out_disp_i[c + center] = round((accu + noise_i[(int)((c + center) / noise_smooth)])*8.0) / 8.0;

                        cv::Mat interpolation_window = interpolation_map(roi);
                        cv::Mat disp_data_window = out_disp(roi);
//                        cv::Mat label_data_window;

                        cv::Mat substitutes = interpolation_window < fill_weights_;
                        fill_weights_.copyTo(interpolation_window, substitutes);
                        disp_data_window.setTo(out_disp_i[c + center], substitutes);
                    }
                }
            }
        }
    }
}

int main(void)
{
    int scale = 1;

    int win_height=768;

    pangolin::CreateWindowAndBind("Main",1024,win_height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glewInit();

    int UI_WIDTH=150;

    pangolin::OpenGlRenderState s_cam;
    s_cam.Set(ProjectionMatrix(640,480,420,420,320,240,0.1,2000));
    s_cam.Set(IdentityMatrix(GlModelViewStack));

    View& d_panel = pangolin::CreatePanel("ui")
      .SetBounds(1.0, 0.0, 0, Attach::Pix(UI_WIDTH));

    int width = 640/scale;
    int height= 480/scale;

    View& d_cam = pangolin::Display("cam")
            .SetBounds(0.0, 1.0, Attach::Pix(UI_WIDTH), 1.0, 640.0f/480.0f)
            .SetHandler(new Handler3D(s_cam));

    GlBufferCudaPtr pbo_debug(GlPixelUnpackBuffer,
                              width*height*sizeof(float),
                              cudaGraphicsMapFlagsNone,
                              GL_DYNAMIC_DRAW);

    GlTexture tex_show(width, height, GL_LUMINANCE);

    View& displayView1 = Display("displayView1")
                        .SetBounds(Attach::Pix(win_height - height/2),
                                   1.0,
                                   Attach::Pix(UI_WIDTH),
                                   Attach::Pix(UI_WIDTH + width/2),
                                   1.0)
                        .SetAspect(640.0f/480.0f);

    View& displayView2 = Display("displayView2")
                            .SetBounds(Attach::Pix(win_height - height/2),
                                       1.0, Attach::Pix(UI_WIDTH + width/2),
                                       Attach::Pix(UI_WIDTH + width/2 + width/2),
                                       1.0)
                            .SetAspect(640.0f/480.0f);

    View& displayView3 = Display("displayView3")
                            .SetBounds(Attach::Pix(win_height - height/2),
                                       1.0,
                                       Attach::Pix(UI_WIDTH + width/2 + width/2),
                                       Attach::Pix(UI_WIDTH + width/2 + width/2 + width/2),
                                       1.0)
                            .SetAspect(640.0f/480.0f);



    // Create vertex and colour buffer objects and register them with CUDA
    GlBufferCudaPtr vertex_array_0(
        GlArrayBuffer, width * height * sizeof(float4),
        cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW
    );

    GlBufferCudaPtr colour_array_0(
        GlArrayBuffer, width * height * sizeof(uchar4),
        cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW
    );


    float* cu_tex_buffer;
    cudaMalloc(&cu_tex_buffer,sizeof(float)*width*height);

    float K[3][3] = { 481.2,      0,     319.50,
                          0, -480.0,     239.50,
                          0,      0,       1.00};


    iu::ImageCpu_32f_C1* h_depth = new iu::ImageCpu_32f_C1(IuSize(width,height));

    /// Data related to vertex
    iu::ImageGpu_32f_C1* depth       = new iu::ImageGpu_32f_C1(IuSize(width,height));
    iu::ImageGpu_16u_C1* noisy_depth_png = new iu::ImageGpu_16u_C1(IuSize(width,height));

    iu::ImageGpu_32f_C1* all_one     = new iu::ImageGpu_32f_C1(IuSize(width,height));
    iu::setValue(1,all_one,all_one->roi());

    iu::ImageGpu_32f_C4* vertex  = new iu::ImageGpu_32f_C4(IuSize(width,height));
    iu::ImageGpu_32f_C4* normals = new iu::ImageGpu_32f_C4(IuSize(width,height));
    iu::ImageGpu_32f_C4* vertex_with_noise = new iu::ImageGpu_32f_C4(IuSize(width,height));

    iu::setValue(make_float4(0),vertex_with_noise,vertex_with_noise->roi());

    iu::ImageGpu_32f_C4* colour = new iu::ImageGpu_32f_C4(IuSize(width,height));

    iu::ImageGpu_32f_C1* noisy_depth = new iu::ImageGpu_32f_C1(IuSize(width,height));
    iu::ImageCpu_32f_C1* h_noisy_depth = new iu::ImageCpu_32f_C1(IuSize(width,height));

    iu::ImageGpu_32f_C2* tex_coords = new iu::ImageGpu_32f_C2(IuSize(width,height));

    iu::ImageGpu_32f_C1* noisy_depth_texture = new iu::ImageGpu_32f_C1(IuSize(width,height));

    srand (time(NULL));

    iu::setValue(0,noisy_depth,noisy_depth->roi());
    iu::setValue(0,noisy_depth_texture,noisy_depth_texture->roi());
    iu::setValue(make_float2(0.5),tex_coords,tex_coords->roi());


    iu::ImageGpu_32f_C1* noisy_depth_copy = new iu::ImageGpu_32f_C1(IuSize(width,height));

//    int count=0;

    float2 fl = make_float2(420.0f,-420.0f)/scale;
    float2 pp = make_float2(319.5f, 239.5f)/scale;


    std::cout<<"Entering the Pangolin Display Loop" << std::endl;

    iu::ImageGpu_8u_C4* d_colour_l0 = new iu::ImageGpu_8u_C4(width,height);
//    iu::ImageCpu_8u_C4* h_colour_l0 = new iu::ImageCpu_8u_C4(width,height);

//    uchar4* colour_data = h_colour_l0->data();

    uchar4 colour_val = make_uchar4(255,255,255,1);
    iu::setValue(colour_val,d_colour_l0,d_colour_l0->roi());


    std::cout<<"Going to the while loop" << std::endl;


    // baseline between IR projector and IR camera
    float baseline = 0.075; // in metres

    cv::Mat dot_pattern_ = cv::imread("../data/kinect-pattern_3x3.png", 0);


    while(!pangolin::ShouldQuit())
    {
        static Var<int> ref_image_no("ui.ref_img_no",0,0,1000);
        static Var<float>scale_disp("ui.scale_disp",1,1,100);

        static Var<int>disp_threshold("ui.disp_thresh",10,0,20);

        static Var<float> focal_length("ui.focal_length",480,10,1000);

        static Var<float> theta1("ui.theta1",0.138,0,1);
        static Var<float> theta2("ui.theta2",0.035,0,1);

        static Var<bool>write_images("ui.write_images",true);

        static Var<float>sigma_shift("ui.sigma shift",1/2.0f,0,1);
        static Var<float>sigma("ui.sigma",0.5,0,1);
        static Var<int>kernel_size("ui.kernel_size",3,1,10);

        /// The depth is between

        float* h_depth_data = h_depth->data();

        char imgFileName[300];

        sprintf(imgFileName,"../data/room_89_simple_data/scenedepth_00_%07d.png",
                (int)ref_image_no);

        std::cout<<imgFileName << std::endl;

        CVD::Image<u_int16_t> depthImage(CVD::ImageRef(width,height));
        CVD::img_load(depthImage,imgFileName);

        std::cout << "File has been read ! " << std::endl;
        std::cout<<" width = " << width << ", height = " << height << std::endl;

        for(int yy = 0; yy < height; yy++)
        {
            for(int xx = 0; xx < width; xx++)
            {
                double val = (float)(depthImage[CVD::ImageRef(xx,yy)])/5000.0f;

                if ( val > 0 && val < 10 )
                    h_depth_data[xx+yy*width] = val;
                else
                    h_depth_data[xx+yy*width] = 0;
            }
        }


        iu::copy(h_depth,depth);

        /// Convert the depth into vertices
        aux_math::ComputeVertexFromDepth(depth->data(),
                                         depth->stride(),
                                         vertex->data(),
                                         vertex->stride(),
                                         width,
                                         height,
                                         fl,
                                         pp,
                                         0,
                                         10);

        /// Compute Normals from these vertices
        aux_math::ComputeNormalsFromVertex(normals->data(),
                                           vertex->data(),
                                           vertex->stride(),
                                           width,
                                           height);

        noise::launch_add_kinect_noise(vertex->data(),
                                       normals->data(),
                                       vertex_with_noise->data(),
                                       vertex->stride(),
                                       vertex->height(),
                                       fl.x,
                                       theta1,
                                       theta2,
                                       0,
                                       0,
                                       0);


        /// Add noise to the vertices
//        iu::copy(vertex,vertex_with_noise);

        iu::setValue(0,noisy_depth,noisy_depth->roi());

        /// Convert these noisy vertices to depth
        aux_math::ComputeDepthFromVertex(vertex_with_noise->data(),
                                         vertex_with_noise->stride(),
                                         noisy_depth->data(),
                                         noisy_depth->stride(),
                                         width,
                                         height,
                                         fl,
                                         pp);

        iu::copy(noisy_depth,h_noisy_depth);


        /// Get gaussian shifts
        noise::gaussian_shifts(tex_coords->data(),
                               tex_coords->stride(),
                               tex_coords->height(),
                               sigma_shift);


        /// http://gpuocelot.googlecode.com/svn/trunk/ocelot/ocelot/cuda/test/textures/texture2D.cu
        noise::uploadTexture2CUDA(noisy_depth->data(),
                                  noisy_depth->pitch(),
                                  noisy_depth->width(),
                                  noisy_depth->height());

        noise::warpImage(noisy_depth_copy->data(),
                         noisy_depth_copy->stride(),
                         tex_coords->data(),
                         tex_coords->stride(),
                         tex_coords->width(),
                         tex_coords->height());

        iu::copy(noisy_depth_copy,h_noisy_depth);


        float max_val = -1E10;
        float min_val =  1E10;

        cv::Mat disp     = cv::Mat(height,width,CV_32FC1);
        cv::Mat out_disp = cv::Mat(height,width,CV_32FC1);

//#pragma omp parallel for collapse(2)
        /// Convert to baseline
        for(int yy =0 ; yy < height; yy++)
        {
            for(int xx = 0; xx < width; xx++)
            {
                if (h_depth_data[yy*width+xx] > 0 ) //&& (float)rand()/RAND_MAX > 0.1)
                {
                    float val = fl.x * baseline / h_depth_data[yy*width+xx];
                    disp.at<float>(yy,xx) = round(val*8.0)/8.0;
                }
                else
                {
                    disp.at<float>(yy,xx) = 0;//invalid_disp_+1;
                }

                if ( disp.at<float>(yy,xx) < min_val )
                {
                    min_val = disp.at<float>(yy,xx);
                }

                if ( disp.at<float>(yy,xx) > max_val )
                {
                    max_val = disp.at<float>(yy,xx);
                }
            }
        }


        std::cout<<"disp: min_val = " << min_val <<", max_val " << max_val << std::endl;

        filterDisp(disp,out_disp,dot_pattern_);

        max_val = -1E10;
        min_val =  1E10;

//#pragma omp parallel for collapse(2)
        for(int yy =0 ; yy < height; yy++)
        {
            for(int xx = 0; xx < width; xx++)
            {
                float val = fl.x * baseline / out_disp.at<float>(yy,xx);

                if ( val > 0 && val < 10.0f )
                {
                    h_depth_data[yy*width+xx] = val;
                }
                else
                {
                    h_depth_data[yy*width+xx] = 0;
                }

                if ( out_disp.at<float>(yy,xx) < min_val )
                {
                    min_val = out_disp.at<float>(yy,xx);
                }

                if ( out_disp.at<float>(yy,xx) > max_val )
                {
                    max_val = out_disp.at<float>(yy,xx);
                }
            }
        }

        std::cout<<"out_disp: min_val = " << min_val <<", max_val " << max_val << std::endl;


        max_val = -1E10;
        min_val =  1E10;

        iu::copy(h_depth,noisy_depth_copy);


        /// Add the final noise
        noise::add_depth_noise_barronCVPR2013(noisy_depth_copy->data(),
                                              noisy_depth_copy->stride(),
                                              noisy_depth_copy->height());

        iu::copy(noisy_depth_copy,noisy_depth);

        noise::launch_convert_depth2png(noisy_depth->data(),
                                 noisy_depth->stride(),
                                 noisy_depth_png->data(),
                                 noisy_depth_png->stride(),
                                 noisy_depth_png->width(),
                                 noisy_depth_png->height());



        aux_math::ComputeVertexFromDepth(noisy_depth->data(),
                                         noisy_depth->stride(),
                                         vertex_with_noise->data(),
                                         vertex_with_noise->stride(),
                                         width,
                                         height,
                                         fl,
                                         pp,
                                         0,
                                         10);



        iu::minMax(depth,depth->roi(),min_val,max_val);

        iu::addWeighted(depth,1.0f/(max_val-min_val),all_one,
                        -min_val/(max_val-min_val),depth,depth->roi());

        std::cout << "max_vald = " << max_val <<", min_vald = " << min_val << std::endl;


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glDisable(GL_DEPTH_TEST);

//        renderutils::DisplayFloatDeviceMemNorm(&displayView1,
//                                               depth->data(),
//                                               depth->pitch(),
//                                               pbo_debug,
//                                               tex_show,
//                                               true,false);

        max_val = -1E10;
        min_val =  1E10;

        iu::minMax(noisy_depth,noisy_depth->roi(),min_val,max_val);

        std::cout << "noisy_depth: max_val = " << max_val <<", min_val = " << min_val << std::endl;

        iu::addWeighted(noisy_depth,1/(max_val-min_val),
                        all_one,-min_val/(max_val-min_val),
                        noisy_depth,noisy_depth->roi());

//        renderutils::DisplayFloatDeviceMemNorm(&displayView2,
//                                               noisy_depth->data(),
//                                               noisy_depth->pitch(),
//                                               pbo_debug,
//                                               tex_show,
//                                               true,false);

        if ( write_images )
        {
            if ( ref_image_no < 1000 )
            {
                CVD::Image< u_int16_t >depthImage= CVD::Image<u_int16_t>(CVD::ImageRef(width,height));

                max_val = -1E10;
                min_val =  1E10;

//                iu::minMax(noisy_depth_png,noisy_depth_png->roi(),min_val,max_val);

//                std::cout<<"16bit min, max = " << max_val <<", "<< min_val << std::endl;


                cudaMemcpy2D(depthImage.data(),
                             width*sizeof(u_int16_t),
                             noisy_depth_png->data(),
                             noisy_depth_png->pitch(),
                             width*sizeof(u_int16_t),
                             height,
                             cudaMemcpyDeviceToHost);

                char depthFileName[300];
//                char imgFileName[300];
//                char txtFileName[300];

                sprintf(depthFileName,"scene_00_%04d_noisy_depth.png",(int)ref_image_no);

//                sprintf(imgFileName,"traj3_noise/scene_00_%04d.png",(int)ref_image_no);
//                sprintf(txtFileName,"traj3_noise/scene_00_%04d.txt",(int)ref_image_no);

                img_save(depthImage,depthFileName);
//                img_save(dataset.getPNGImage< CVD::Rgb<u_int16_t> >(ref_image_no,0),imgFileName);

//                ofstream ofile;
//                ofile.open(txtFileName);
//                ofile << dataset.computeTpov_cam(ref_image_no,0)<<endl;
//                ofile.close();

//                ref_image_no = ref_image_no+1;
            }

            else
            {
                write_images = false;
            }
        }

        d_cam.ActivateAndScissor(s_cam);

        {

            glEnable(GL_DEPTH_TEST);

            ref_image_no = ref_image_no + 1;

        }

        d_panel.Render();
        pangolin::FinishFrame();

        GPUMemory();

    }
}



//#include <iostream>
//#include <opencv2/opencv.hpp>

//using namespace std;
//using namespace cv;
//float invalid_disp_ = 999999;
/*
void filterDisp(const cv::Mat& disp, cv::Mat& out_disp)
{
    const int size_filt_ = 9;

    // initialize filter matrices for simulated disparity
    cv::Mat weights_ = cv::Mat(size_filt_, size_filt_, CV_32FC1);

    for (int x = 0; x < size_filt_; ++x)
    {
        float *weights_i = weights_.ptr<float>(x);

        for (int y = 0; y < size_filt_; ++y)
        {
            int tmp_x = x - size_filt_ / 2;
            int tmp_y = y - size_filt_ / 2;

            if (tmp_x != 0 && tmp_y != 0)
                weights_i[y] = 1.0 / ((1.2*(float)tmp_x)*(1.2*(float)tmp_x) + (1.2*(float)tmp_y)*(1.2*(float)tmp_x));
            else
                weights_i[y] = 1.0;
        }
    }

    const float window_inlier_distance_ = 0.1;

    cv::Mat dot_pattern_;
    dot_pattern_ = cv::imread("..//kinect-pattern_3x3.png", 0);

    cv::Mat noise_field;
    noise_field = cv::Mat::zeros(disp.rows, disp.cols, CV_32FC1);

    // mysterious parameter
    float noise_smooth = 1.5;

    // initialise output arrays
    out_disp = cv::Mat(disp.rows, disp.cols, disp.type());
    out_disp.setTo(invalid_disp_);


    // determine filter boundaries
    unsigned int lim_rows = std::min(disp.rows - size_filt_, dot_pattern_.rows - size_filt_);
    unsigned int lim_cols = std::min(disp.cols - size_filt_, dot_pattern_.cols - size_filt_);

    int center = size_filt_ / 2.0;

    for (unsigned int r = 0; r < lim_rows; r++)
    {
        const float* disp_i = disp.ptr<float>(r + center);
        const float* dots_i = dot_pattern_.ptr<float>(r + center);

        float* out_disp_i = out_disp.ptr<float>(r + center);

        float* noise_i = noise_field.ptr<float>((int)((r + center) / noise_smooth));

        /// window shifting over disparity image
        for (unsigned int c = 0; c < lim_cols; c++)
        {
            if (dots_i[c + center] > 0 && disp_i[c + center] < invalid_disp_)
            {
                cv::Rect roi = cv::Rect(c, r, size_filt_, size_filt_);

                cv::Mat window = disp(roi);
                cv::Mat dot_win = dot_pattern_(roi);

                // check if we are at a occlusion boundary without valid disparity values
                // return value not binary but between 0 or 255
                cv::Mat valid_vals = (window < invalid_disp_);
                cv::Mat valid_dots;

                cv::bitwise_and(valid_vals, dot_win, valid_dots);

                cv::Scalar n_valids = cv::sum(valid_dots) / 255.0;
                cv::Scalar n_thresh = cv::sum(dot_win) / 255.0;

                // only add depth value at center of window if there are more
                // valid disparity values than 2/3 of the number of dots
                if (n_valids(0) > n_thresh(0) / 1.5)
                {
                    // compute mean only over the valid values of disparities in that window
                    cv::Scalar mean = cv::mean(window, valid_vals);

                    // weighted deviation from mean
                    cv::Mat diffs = cv::abs(window - mean);
                    cv::multiply(diffs, weights_, diffs);

                    // get valid values that fall on dot pattern
                    cv::Mat valids = (diffs < window_inlier_distance_);
                    cv::bitwise_and(valids, valid_dots, valid_dots);

                    n_valids = cv::sum(valid_dots) / 255.0;

                    // only add depth value at center of window if there are more
                    // valid disparity values than 2/3 of the number of dots
                    if (n_valids(0) > n_thresh(0) / 1.5)
                    {
                        float accu = window.at<float>(center, center);

                        assert(accu < invalid_disp_);

                        out_disp_i[c + center] = round((accu)*8.0) / 8.0;


                    }
                }
            }
        }
    }
}*/

//void filterDisp(const cv::Mat& disp, cv::Mat& out_disp)
//{
//    const int size_filt_ = 9;

//    // initialize filter matrices for simulated disparity
//    cv::Mat weights_ = cv::Mat(size_filt_, size_filt_, CV_32FC1);

//    for (int x = 0; x < size_filt_; ++x)
//    {
//        float *weights_i = weights_.ptr<float>(x);

//        for (int y = 0; y < size_filt_; ++y)
//        {
//            int tmp_x = x - size_filt_ / 2;
//            int tmp_y = y - size_filt_ / 2;

//            if (tmp_x != 0 && tmp_y != 0)
//                weights_i[y] = 1.0 / ((1.2*(float)tmp_x)*(1.2*(float)tmp_x) + (1.2*(float)tmp_y)*(1.2*(float)tmp_x));
//            else
//                weights_i[y] = 1.0;
//        }
//    }

//    Mat fill_weights_ = cv::Mat(size_filt_, size_filt_, CV_32FC1);
//    for (int x = 0; x < size_filt_; ++x){
//        float *weights_i = fill_weights_.ptr<float>(x);
//        for (int y = 0; y < size_filt_; ++y){
//            int tmp_x = x - size_filt_ / 2;
//            int tmp_y = y - size_filt_ / 2;
//            if (std::sqrt(tmp_x*tmp_x + tmp_y*tmp_y) < 3.1)
//                weights_i[y] = 1.0 / (1.0 + tmp_x*tmp_x + tmp_y*tmp_y);
//            else
//                weights_i[y] = -1.0;
//        }
//    }

//    const float window_inlier_distance_ = 0.1;

//    cv::Mat dot_pattern_;
//    dot_pattern_ = cv::imread("../data/kinect-pattern_3x3.png", 0);

//    cv::Mat interpolation_map = cv::Mat::zeros(disp.rows, disp.cols, CV_32FC1);

//    cv::Mat noise_field;
//    noise_field = cv::Mat::zeros(disp.rows, disp.cols, CV_32FC1);

//    // mysterious parameter
//    float noise_smooth = 1.5;

//    // initialise output arrays
//    out_disp = cv::Mat(disp.rows, disp.cols, disp.type());
//    out_disp.setTo(invalid_disp_);


//    // determine filter boundaries
//    unsigned lim_rows = std::min(disp.rows - size_filt_, dot_pattern_.rows - size_filt_);
//    unsigned lim_cols = std::min(disp.cols - size_filt_, dot_pattern_.cols - size_filt_);
//    int center = size_filt_ / 2.0;
//    for (unsigned r = 0; r < lim_rows; ++r)
//    {
//        const float* disp_i = disp.ptr<float>(r + center);
//        const unsigned char* labels_i;

//        const float* dots_i = dot_pattern_.ptr<float>(r + center);

//        float* out_disp_i = out_disp.ptr<float>(r + center);
//        unsigned char* out_labels_i;


//        float* noise_i = noise_field.ptr<float>((int)((r + center) / noise_smooth));

//        // window shifting over disparity image
//        for (unsigned c = 0; c < lim_cols; ++c)
//        {
//            if (dots_i[c + center] > 0 && disp_i[c + center] < invalid_disp_)
//            {
//                cv::Rect roi = cv::Rect(c, r, size_filt_, size_filt_);
//                cv::Mat window = disp(roi);
//                cv::Mat dot_win = dot_pattern_(roi);
//                // check if we are at a occlusion boundary without valid disparity values
//                // return value not binary but between 0 or 255
//                cv::Mat valid_vals = (window < invalid_disp_);
//                cv::Mat valid_dots;
//                cv::bitwise_and(valid_vals, dot_win, valid_dots);
//                cv::Scalar n_valids = cv::sum(valid_dots) / 255.0;
//                cv::Scalar n_thresh = cv::sum(dot_win) / 255.0;

//                // only add depth value at center of window if there are more
//                // valid disparity values than 2/3 of the number of dots
//                if (n_valids(0) > n_thresh(0) / 1.5)
//                {
//                    // compute mean only over the valid values of disparities in that window
//                    cv::Scalar mean = cv::mean(window, valid_vals);
//                    // weighted deviation from mean
//                    cv::Mat diffs = cv::abs(window - mean);
//                    cv::multiply(diffs, weights_, diffs);
//                    // get valid values that fall on dot pattern
//                    cv::Mat valids = (diffs < window_inlier_distance_);
//                    cv::bitwise_and(valids, valid_dots, valid_dots);
//                    n_valids = cv::sum(valid_dots) / 255.0;

//                    // only add depth value at center of window if there are more
//                    // valid disparity values than 2/3 of the number of dots
//                    if (n_valids(0) > n_thresh(0) / 1.5)
//                    {
//                        float accu = window.at<float>(center, center);

//                        assert(accu < invalid_disp_);

//                        out_disp_i[c + center] = round((accu + noise_i[(int)((c + center) / noise_smooth)])*8.0) / 8.0;

//                        cv::Mat interpolation_window = interpolation_map(roi);
//                        cv::Mat disp_data_window = out_disp(roi);
//                        cv::Mat label_data_window;

//                        cv::Mat substitutes = interpolation_window < fill_weights_;
//                        fill_weights_.copyTo(interpolation_window, substitutes);
//                        disp_data_window.setTo(out_disp_i[c + center], substitutes);
//                    }
//                }
//            }
//        }
//    }
//}

//int main(int argc, const char *argv[]) {
//    //reference: http://docs.opencv.org/master/modules/contrib/doc/facerec/colormaps.html


//    Mat depth = imread("/home/ankur/workspace/code/OffScreenDepthRender/data/bedroom1_data/scenedepth_00_0000000.png", IMREAD_UNCHANGED);
//    float B = 0.075f;
//    float f = 420.f;
//    Mat disp;
//    divide(5000.f * B * f, depth, disp, CV_32FC1);
//    cout << depth.rowRange(10, 11).colRange(300,340) << endl;
//    cout << disp.rowRange(10, 11).colRange(300, 340) << endl;

//    Mat out_disp = disp.clone();
//    filterDisp(disp, out_disp);
//    //cout << disp.rowRange(50,70) << endl;
//    // Show the result:
//    Mat out_depth;
//    divide(5000.f * B * f, out_disp, out_depth, CV_32FC1);
//    cout << out_disp.rowRange(10, 11).colRange(300, 340) << endl;
//    cout << out_depth.rowRange(10, 11).colRange(300, 340) << endl;
//    Mat disp_depth;
//    out_depth.convertTo(disp_depth, CV_16UC1);

//    imwrite("noise.png", disp_depth);



//    waitKey(0);

//    return 0;
//}



