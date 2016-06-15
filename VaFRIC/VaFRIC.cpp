/* Copyright (c) 2013 Ankur Handa
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "VaFRIC.h"

#include <time.h>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

using namespace TooN;
//using namespace boost;

namespace dataset{


TooN::SE3<> vaFRIC::computeTpov_cam(int ref_img_no, int which_blur_sample)
{
    char text_file_name[360];

    sprintf(text_file_name,"%s/scene_%02d_%04d.txt",filebasename.c_str(),
            which_blur_sample,ref_img_no);

    ifstream cam_pars_file(text_file_name);

    char readlinedata[300];

    float4 direction;
    float4 upvector;
    TooN::Vector<3>posvector;


    while(1)
    {
        cam_pars_file.getline(readlinedata,300);

        if ( cam_pars_file.eof())
            break;

        istringstream iss;

        if ( strstr(readlinedata,"cam_dir")!= NULL)
        {
            std::string cam_dir_str(readlinedata);

            cam_dir_str = cam_dir_str.substr(cam_dir_str.find("= [")+3);
            cam_dir_str = cam_dir_str.substr(0,cam_dir_str.find("]"));

            iss.str(cam_dir_str);
            iss >> direction.x ;
            iss.ignore(1,',');
            iss >> direction.y ;
            iss.ignore(1,',') ;
            iss >> direction.z;
            iss.ignore(1,',');
            //cout << direction.x<< ", "<< direction.y << ", "<< direction.z << endl;
            direction.w = 0.0f;

        }

        if ( strstr(readlinedata,"cam_up")!= NULL)
        {

            string cam_up_str(readlinedata);

            cam_up_str = cam_up_str.substr(cam_up_str.find("= [")+3);
            cam_up_str = cam_up_str.substr(0,cam_up_str.find("]"));


            iss.str(cam_up_str);
            iss >> upvector.x ;
            iss.ignore(1,',');
            iss >> upvector.y ;
            iss.ignore(1,',');
            iss >> upvector.z ;
            iss.ignore(1,',');


            upvector.w = 0.0f;

        }

        if ( strstr(readlinedata,"cam_pos")!= NULL)
        {
            string cam_pos_str(readlinedata);

            cam_pos_str = cam_pos_str.substr(cam_pos_str.find("= [")+3);
            cam_pos_str = cam_pos_str.substr(0,cam_pos_str.find("]"));

            iss.str(cam_pos_str);
            iss >> posvector[0] ;
            iss.ignore(1,',');
            iss >> posvector[1] ;
            iss.ignore(1,',');
            iss >> posvector[2] ;
            iss.ignore(1,',');

        }

    }

    /// z = dir / norm(dir)
    Vector<3> z;
    z[0] = direction.x;
    z[1] = direction.y;
    z[2] = direction.z;
    normalize(z);

    /// x = cross(cam_up, z)
    Vector<3> x = Zeros(3);
    x[0] =  upvector.y * z[2] - upvector.z * z[1];
    x[1] =  upvector.z * z[0] - upvector.x * z[2];
    x[2] =  upvector.x * z[1] - upvector.y * z[0];

    normalize(x);

    /// y = cross(z,x)
    Vector<3> y = Zeros(3);
    y[0] =  z[1] * x[2] - z[2] * x[1];
    y[1] =  z[2] * x[0] - z[0] * x[2];
    y[2] =  z[0] * x[1] - z[1] * x[0];

    Matrix<3,3> R = Zeros(3,3);
    R[0][0] = x[0];
    R[1][0] = x[1];
    R[2][0] = x[2];

    R[0][1] = y[0];
    R[1][1] = y[1];
    R[2][1] = y[2];

    R[0][2] = z[0];
    R[1][2] = z[1];
    R[2][2] = z[2];


    return TooN::SE3<>(R, posvector);
}


void vaFRIC::readDepthFile(int ref_img_no, int which_blur_sample, std::vector<float> &depth_array)
{

    if(!depth_array.size())
        depth_array = std::vector<float>(img_width*img_height,0);

    char depthFileName[300];

    sprintf(depthFileName,"%s/scene_%02d_%04d.depth",filebasename.c_str(),which_blur_sample,ref_img_no);

    ifstream depthfile;
    depthfile.open(depthFileName);

    for(int i = 0 ; i < img_height ; i++)
    {
        for (int j = 0 ; j < img_width ; j++)
        {
            double val = 0;
            depthfile >> val;
            depth_array[i*img_width+j] = val;
        }
    }

    depthfile.close();
}

void vaFRIC::getEuclidean2PlanarDepth(int ref_img_no, int which_blur_sample, float* depth_array)
{
    char depthFileName[300];

    sprintf(depthFileName,"%s/scene_%02d_%04d.depth",filebasename.c_str(),which_blur_sample,ref_img_no);

    ifstream depthfile;
    depthfile.open(depthFileName);

    for(int i = 0 ; i < img_height ; i++)
    {
        for (int j = 0 ; j < img_width ; j++)
        {
            double val = 0;
            depthfile >> val;
            depth_array[i*img_width+j] = val;
        }
    }

    depthfile.close();

    for(int v = 0 ; v < img_height ; v++)
    {
        for(int u = 0 ; u < img_width ; u++)
        {
            float u_u0_by_fx = (u-u0)/focal_x;
            float v_v0_by_fy = (v-v0)/focal_y;

            depth_array[u+v*img_width] =  depth_array[u+v*img_width] / sqrt(u_u0_by_fx*u_u0_by_fx +
                                                                    v_v0_by_fy*v_v0_by_fy + 1 ) ;

        }
    }
}


void vaFRIC::getEuclidean2PlanarDepth(int ref_img_no, int which_blur_sample, std::vector<float>& depth_array)
{

    if(!depth_array.size())
        depth_array = std::vector<float>(img_width*img_height,0);

    assert(focal_y<0);

    readDepthFile(ref_img_no, which_blur_sample,depth_array);

    for(int v = 0 ; v < img_height ; v++)
    {
        for(int u = 0 ; u < img_width ; u++)
        {
            float u_u0_by_fx = (u-u0)/focal_x;
            float v_v0_by_fy = (v-v0)/focal_y;

            depth_array[u+v*img_width] =  depth_array[u+v*img_width] / sqrt(u_u0_by_fx*u_u0_by_fx +
                                                                    v_v0_by_fy*v_v0_by_fy + 1 ) ;

        }
    }

}

void vaFRIC::get3Dpositions(int ref_img_no, int which_blur_sample, float4* points3D)
{
    if ( points3D == NULL )
        points3D = new float4[img_width*img_height];

    std::vector<float> depth_array(img_width*img_height,0);

    readDepthFile(ref_img_no, which_blur_sample,depth_array);

    /// Convert into 3D points
    for(int v = 0 ; v < img_height ; v++)
    {
        for(int u = 0 ; u < img_width ; u++)
        {

            float u_u0_by_fx = (u-u0)/focal_x;
            float v_v0_by_fy = (v-v0)/focal_y;

            float z =  depth_array[u+v*img_width] / sqrt(u_u0_by_fx*u_u0_by_fx + v_v0_by_fy*v_v0_by_fy + 1 ) ;

//            cout <<" z =" << z << endl;
            points3D[u+v*img_width].z = z;
            points3D[u+v*img_width].y = (v_v0_by_fy)*(z);
            points3D[u+v*img_width].x = (u_u0_by_fx)*(z);
            points3D[u+v*img_width].w = 1.0f;
        }
    }

}

void vaFRIC::convertPOV2TUMformat(float *pov_format, u_int16_t *tum_format, int scale_factor)
{
    for(int i = 0 ; i < img_height ; i++)
    {
        for (int j = 0 ; j < img_width ; j++)
        {
            tum_format[i*img_width+j] = (u_int16_t)(pov_format[i*img_width+j]*scale_factor);
        }
    }
}



void vaFRIC::convertPOV2TUMformat(float *pov_format, float *tum_format, int scale_factor)
{
    for(int i = 0 ; i < img_height ; i++)
    {
        for (int j = 0 ; j < img_width ; j++)
        {
            tum_format[i*img_width+j] = (u_int16_t)(pov_format[i*img_width+j]*scale_factor);
        }
    }
}


void vaFRIC::convertDepth2NormalisedFloat(float *depth_arrayIn,
                                          float *depth_arrayOut,
                                          float max_depth, float min_depth)
{
    for(int i = 0 ; i < img_height ; i++)
    {
        for (int j = 0 ; j < img_width ; j++)
        {
            depth_arrayOut[i*img_width+j] = (depth_arrayIn[i*img_width+j]-min_depth)/(max_depth-min_depth);
        }
    }
}

void vaFRIC::convertDepth2NormalisedFloat(float *depth_arrayIn,
                                          float *depth_arrayOut, int scale_factor)
{
    for(int i = 0 ; i < img_height ; i++)
    {
        for (int j = 0 ; j < img_width ; j++)
        {
            depth_arrayOut[i*img_width+j] = (depth_arrayIn[i*img_width+j]*scale_factor)/65536;
        }
    }
}


void vaFRIC::addDepthNoise(std::vector<float>& depth_arrayIn, std::vector<float>& depth_arrayOut,
                           float z1, float z2,
                           float z3, int ref_img_no,
                           int which_blur_sample )
{
    /// http://www.bnikolic.co.uk/blog/cpp-boost-rand-normal.html

    /// https://github.com/mattdesl/lwjgl-basics/wiki/ShaderLesson6#wiki-GeneratingNormals
    float4* h_points3D = new float4[img_width*img_height];

    get3Dpositions(ref_img_no,which_blur_sample,h_points3D);

    depth_arrayOut.clear();

    if(!depth_arrayOut.size())
        depth_arrayOut = std::vector<float>(img_width*img_height,0);


    for(int i = 0 ; i < img_width; i++ )
    {
        for(int j = 0 ; j < img_height; j++)
        {
            if (i == 0 || j == 0 || i == img_width-1 || j == img_height-1)
                depth_arrayOut[i+j*img_width] = depth_arrayIn[i+j*img_width];
            else
            {
                Vector<3> vertex_left;
                Vector<3> vertex_right;
                Vector<3> vertex_up;
                Vector<3> vertex_down;

                vertex_left[0]  = h_points3D[i-1+j*img_width].x;
                vertex_left[1]  = h_points3D[i-1+j*img_width].y;
                vertex_left[2]  = h_points3D[i-1+j*img_width].z;

                vertex_right[0] = h_points3D[i+1+j*img_width].x;
                vertex_right[1] = h_points3D[i+1+j*img_width].y;
                vertex_right[2] = h_points3D[i+1+j*img_width].z;

                vertex_up[0]    = h_points3D[i+(j-1)*img_width].x;
                vertex_up[1]    = h_points3D[i+(j-1)*img_width].y;
                vertex_up[2]    = h_points3D[i+(j-1)*img_width].z;

                vertex_down[0]  = h_points3D[i+(j+1)*img_width].x;
                vertex_down[1]  = h_points3D[i+(j+1)*img_width].y;
                vertex_down[2]  = h_points3D[i+(j+1)*img_width].z;

                Vector<3>dxv = vertex_right - vertex_left;
                Vector<3>dyv = vertex_down  - vertex_up;

                Vector<3> normal_vector = dyv ^ dxv ; //dataset::cross(dyv,dxv);

                normalize(normal_vector);

                double c = normal_vector[2];

                double theta = acos(fabs(c));

                double z = depth_arrayIn[i+j*img_width]/100.0;

                double theta_const = (theta/(M_PI/2-theta))*(theta/(M_PI/2-theta))+1E-6;

                double sigma_z = z1 + z2*(z-z3)*(z-z3);// + (z3/sqrt(z)+1E-6)*(theta/(M_PI/2-theta+1E-6))*(theta/(M_PI/2-theta+1E-6));



                sigma_z = sigma_z + (0.0001/sqrt(z))*(theta_const);

                /*boost::variate_generator<boost::mt19937, boost::normal_distribution<> >
                    generator(boost::mt19937(time(0)),
                              boost::normal_distribution<>(0,sigma_z));*/

//                static dataset::RNGType rng;

                static boost::mt19937 rand_number(std::time(0));

//                rng.seed();
                boost::normal_distribution<> rdist(0.0,sigma_z); /**< normal distribution
                                           with mean of 1.0 and standard deviation of 0.5 */

                double noisy_depth = rdist(rand_number)*100;

//                cout << "noisy_depth = " << noisy_depth << endl;

                depth_arrayOut[i+j*img_width] = depth_arrayIn[i+j*img_width] + noisy_depth;

                if ( depth_arrayOut[i+j*img_width] <= 0 || fabs(theta-M_PI/2) <= 2*M_PI/180.0f )
                    depth_arrayOut[i+j*img_width] = 0;//depth_arrayIn[i+j*img_width];
                if ( depth_arrayOut[i+j*img_width] >= 5E2 )
                        depth_arrayOut[i+j*img_width] = depth_arrayIn[i+j*img_width];

//                cout << "depth value = " << depth_arrayIn[i+j*img_width] << endl;
//                cout << "rand value generated = " << depth_arrayIn[i+j*img_width] + noisy_depth << endl;


            }
        }
    }

    delete h_points3D;

}


void vaFRIC::convertDepth2NormalImage(int ref_img_no, int which_blur_sample, string imgName)
{
    /// https://github.com/mattdesl/lwjgl-basics/wiki/ShaderLesson6#wiki-GeneratingNormals
    float4* h_points3D = new float4[img_width*img_height];

    get3Dpositions(ref_img_no,which_blur_sample,h_points3D);

    CVD::Image< CVD::Rgb<CVD::byte> > normalImage(CVD::ImageRef(img_width,img_height));

    for(int i = 0 ; i < img_width; i++ )
    {
        for(int j = 0 ; j < img_height; j++)
        {
            if (i == 0 || j == 0 || i == img_width-1 || j == img_height-1)
                normalImage[CVD::ImageRef(i,j)] = CVD::Rgb<CVD::byte>(255,255,255);
            else
            {
                Vector<3> vertex_left;
                Vector<3> vertex_right;
                Vector<3> vertex_up;
                Vector<3> vertex_down;

                vertex_left[0]  = h_points3D[i-1+j*img_width].x;
                vertex_left[1]  = h_points3D[i-1+j*img_width].y;
                vertex_left[2]  = h_points3D[i-1+j*img_width].z;

                vertex_right[0] = h_points3D[i+1+j*img_width].x;
                vertex_right[1] = h_points3D[i+1+j*img_width].y;
                vertex_right[2] = h_points3D[i+1+j*img_width].z;

                vertex_up[0]    = h_points3D[i+(j-1)*img_width].x;
                vertex_up[1]    = h_points3D[i+(j-1)*img_width].y;
                vertex_up[2]    = h_points3D[i+(j-1)*img_width].z;

                vertex_down[0]  = h_points3D[i+(j+1)*img_width].x;
                vertex_down[1]  = h_points3D[i+(j+1)*img_width].y;
                vertex_down[2]  = h_points3D[i+(j+1)*img_width].z;

                Vector<3>dxv = vertex_right - vertex_left;
                Vector<3>dyv = vertex_down  - vertex_up;

                Vector<3> normal_vector = dyv ^ dxv ; //dataset::cross(dyv,dxv);

                normalize(normal_vector);

                normalImage[CVD::ImageRef(i,j)] = CVD::Rgb<CVD::byte>(
                            (unsigned char)(normal_vector[0]*128.f+128.f),
                            (unsigned char)(normal_vector[1]*128.f+128.f),
                            (unsigned char)(normal_vector[2]*128.f+128.f)
                            );
            }
        }

    }

    CVD::img_save(normalImage,imgName.c_str());
    delete h_points3D;
}

}
