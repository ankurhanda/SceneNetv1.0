// To read log file
#include <lcm/lcm-cpp.hpp>

#include <stdio.h>
#include <iostream>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdarg.h>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <png.h>
#include <cvd/image_io.h>
#include "lcmtypes/openni_frame_msg_t.h"
#include "lcmtypes/openni_depth_msg_t.h"
#include "lcmtypes/openni_image_msg_t.h"

static void usage(const char* progname)
{
    fprintf (stderr, "Usage: %s [options]\n"
            "\n"
            "Options:\n"
            "  -o (filename)   Use LCM log out file. (REQUIRED)\n"
            "  -d (directory)  Use directory.\n"
            "  -h              This help message.\n"
            , progname);
    exit(1);
}

int main(int argc, char ** argv)
{
    int c;

    char* data_file, *output_filename;
    while ((c = getopt (argc, argv, "o:d:h")) >= 0) {
        switch (c) {
            case 'o':
                output_filename = optarg;
                std::cout << "Using lcm outfile "<< output_filename<<std::endl;
                break;
            case 'd':
                data_file = optarg;
                std::cout<<"Using directory "<< data_file <<std::endl;
                break;
            case 'h':
            default:
                usage(argv[0]);
                break;
        }
    }

    std::string log_file(output_filename);
    std::string directory(data_file);

    //LogFile lcm_log_file(log_file);
    lcm::LogFile lr(log_file, "w");
    if (!lr.good())
    {
        std::cout<< "LogFile is not good!"<<std::endl;
        return -1;
    }

    CVD::Image<CVD::Rgb<CVD::byte> > img(CVD::ImageRef(640,480));
    CVD::Image<uint16_t> depth_img(CVD::ImageRef(640, 480));

    int depth_compress_buf_size = 640*480*sizeof(int16_t)*2;
    int rgb_buf_size = 640*480*sizeof(int8_t)*3;

    lcm_t * lcm;
    lcm = lcm_create(NULL);
    if (!lcm)
    {
        std::cout<<"unable to initialize LCM"<<std::endl;
        return -1;
    }

    for (int i = 0; i < 4967 ; i++)
    {
        std::stringstream rgb_filename, depth_filename;
        rgb_filename << directory;
        rgb_filename << "scene_00_"<<std::setfill('0') << std::setw(7) << i <<".png";
        depth_filename << directory;
        depth_filename << "scenedepth_00_"<<std::setfill('0') << std::setw(7) << i <<".png";
        
        std::cout << "\rread file "<<rgb_filename.str().c_str()<<"          ";
        fflush(stdout);
        img = CVD::img_load(rgb_filename.str().c_str());
        depth_img = CVD::img_load(depth_filename.str().c_str());

        
        openni_frame_msg_t nextFrame;

        nextFrame.timestamp = time(NULL);

        nextFrame.depth.compression = OPENNI_DEPTH_MSG_T_COMPRESSION_NONE;

        nextFrame.depth.width = 640;
        nextFrame.depth.height = 480;
        nextFrame.depth.depth_data_nbytes = nextFrame.depth.width * nextFrame.depth.height * sizeof(short);
        nextFrame.depth.timestamp = nextFrame.timestamp;

        nextFrame.disparity.width = 640;
        nextFrame.disparity.height = 480;
        nextFrame.disparity.disparity_data_nbytes = 0;

        nextFrame.image.width = 640;
        nextFrame.image.height = 480;
        nextFrame.image.image_data_nbytes = nextFrame.image.width * nextFrame.image.height * sizeof(unsigned char) * 3;
        nextFrame.image.image_data_format = OPENNI_IMAGE_MSG_T_VIDEO_RGB;
        nextFrame.image.timestamp = nextFrame.timestamp;

        //nextFrame.depth.depth_data = (uint16_t*)malloc(depth_compress_buf_size);
        uint16_t * depth_compress_buf = (uint16_t*)malloc(depth_compress_buf_size);
        uint8_t * rgb_buf = (uint8_t*)malloc(rgb_buf_size);

        unsigned long compressed_size = depth_compress_buf_size;

        // Copy data over
        for (int y = 0; y < 480; y++)
        {
            for (int x = 0; x < 640; x++)
            {
                int depth_ind = y*640+x;
                // is 32 right? Going from 2^16 (png files) to 11 bit depth from 
                // kinect
                depth_compress_buf[depth_ind] = depth_img[CVD::ImageRef(x,y)]/32;
                int rgb_ind = depth_ind*3;
                rgb_buf[rgb_ind+0] = img[CVD::ImageRef(x,y)].red;
                rgb_buf[rgb_ind+1] = img[CVD::ImageRef(x,y)].green;
                rgb_buf[rgb_ind+2] = img[CVD::ImageRef(x,y)].blue;
            }
        }

        nextFrame.depth.depth_data_nbytes = compressed_size;
        nextFrame.depth.depth_data = (uint8_t*)depth_compress_buf;

        nextFrame.image.image_data_nbytes = rgb_buf_size;
        nextFrame.image.image_data = rgb_buf;

        openni_frame_msg_t_publish(lcm, "KINECT_DATA", &nextFrame);

        free(depth_compress_buf);
        free(rgb_buf);

       // openni_frame_msg_t_publish(lcm, "KINECT_DATA", &nextFrame);
        //sleep(1);
        usleep(10000);
    }
    std::cout << "Done"<<std::endl;
    return 1;
}


















