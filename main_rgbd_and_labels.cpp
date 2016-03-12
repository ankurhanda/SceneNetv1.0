#include <iostream>
#include <sstream>
#include <chrono>
#include <thread>
#include <Eigen/Eigen>

#include <stdio.h>
#include <stdlib.h>

#include <pangolin/pangolin.h>
#include <pangolin/display.h>
#include <SceneGraph/SceneGraph.h>

#include <TooN/TooN.h>
#include <TooN/se3.h>
#include <vector>

#include <cvd/image_io.h>
#include <cvd/gl_helpers.h>

#include "utils/map_object_label2training_label.h"


using namespace std;
using namespace pangolin;

void Usage() {
    cout << "Usage: ModelViewer filename" << endl;
}

int main( int argc, char* argv[] )
{
    if(argc != 2) {
        Usage();
        exit(-1);
    }

    std::string obj_basename(argv[1]);
    std::size_t find_dot   = obj_basename.find(".obj");
    std::size_t find_slash = obj_basename.find_last_of('/');

    std::cout<<"  find_dot = " << find_dot << std::endl;
    std::cout<<"find_slash = " << find_slash << std::endl;

    obj_basename = obj_basename.substr(find_slash+1,find_dot-find_slash-1);

    std::string data_dir = "../data/" + obj_basename + "_data";

    const std::string model_filename(argv[1]);

    int UI_WIDTH = 150;

    // Create OpenGL window in single line thanks to GLUT
    pangolin::CreateWindowAndBind("Main",640+UI_WIDTH,480);
    SceneGraph::GLSceneGraph::ApplyPreferredGlSettings();
    glClearColor( 0,0,0,0);
    glewInit();

    std::string system_command = std::string("grep -R \"o \" ") + std::string(argv[1]) +
            std::string(" > object_names.txt");

    std::cout<<system_command << std::endl;

    system(system_command.c_str());

    std::vector<std::string>objectNames;

    ifstream objectNamesFile("object_names.txt");

    if ( objectNamesFile.is_open())
    {
        char readlinedata[200];

        while(1)
        {
            objectNamesFile.getline(readlinedata,200);

            if ( objectNamesFile.eof())
                break;

            istringstream iss(readlinedata);
            std::string objectName(iss.str());

            objectName = objectName.substr(2,objectName.size()-2);
            std::cout<<objectName<< std::endl;

            objectNames.push_back(objectName);
        }
    }

    objectNamesFile.close();


    // Count the number of shapes
    int max_label = 0;
    for (int i = 0; i < objectNames.size(); i++)
    {
        int training_label = obj_label2training_label(objectNames.at(i));
        max_label = std::max(max_label, training_label);
    }


//    TooN::Matrix<Dynamic, Dynamic> colours(max_label, 3);

    Eigen::MatrixXd colours(max_label,3);
    std::map<int, int>colour2indexMap;

    // <= since the label was seen, so need to include
    for ( int i = 0; i <= max_label; i++)
    {
        // With the same seeded label, the colors should be the same across runs
        // +1 since srand(0) is srand(1) since rand can't be initialized by 0
        srand(i+1);
        colours(i, 0) = static_cast<float>(rand()) /
                static_cast<float>(RAND_MAX);
        colours(i, 1) = static_cast<float>(rand()) /
                static_cast<float>(RAND_MAX);
        colours(i, 2) = static_cast<float>(rand()) /
                static_cast<float>(RAND_MAX);

        colour2indexMap[ (int)round((colours(i,0))*255) +
                ((int)round(colours(i,1)*255))*256 +
                ((int)round(colours(i,2)*255))*256*256] = i;
    }

    Eigen::MatrixXd renderingColours(objectNames.size(),3);

    for(int i = 0; i < objectNames.size();i++)
    {
        int training_label = obj_label2training_label(objectNames.at(i));

        renderingColours(i,0) = colours(training_label,0);
        renderingColours(i,1) = colours(training_label,1);
        renderingColours(i,2) = colours(training_label,2);
    }



    // Scenegraph to hold GLObjects and relative transformations
    SceneGraph::GLSceneGraph glGraph;

    SceneGraph::GLLight light(10,10,-100);
    glGraph.AddChild(&light);

    SceneGraph::AxisAlignedBoundingBox bbox;
    
#ifdef HAVE_ASSIMP
    // Define a mesh object and try to load model
    SceneGraph::GLMesh glMesh;
    try {
        glMesh.Init(model_filename);
        glGraph.AddChild(&glMesh);
        bbox = glMesh.ObjectAndChildrenBounds();
    }catch(exception e) {
        cerr << "Cannot load mesh." << endl;
        cerr << e.what() << std::endl;
        exit(-1);
    }
#endif // HAVE_ASSIMP

    
    const Eigen::Vector3d center = bbox.Center();
    double size = bbox.Size().norm();
   
    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState stacks3d(
        pangolin::ProjectionMatrixRDF_BottomLeft(640,480,420,420,320,240, 0.1, 1000),
        pangolin::ModelViewLookAt(center(0), center(1) + size, center(2) + size/4,
                                  center(0), center(1), center(2), pangolin::AxisNegZ)
    );


    /// Create a Panel
    pangolin::View& d_panel = pangolin::CreatePanel("ui")
            .SetBounds(1.0, 0.0, 0, pangolin::Attach::Pix(UI_WIDTH));

    /// Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::Display("cam")
      .SetBounds(0.0, 1, Attach::Pix(UI_WIDTH), 1, -640.0f/480.0f)
      .SetHandler(new Handler3D(stacks3d));


    int width  = 640;
    int height = 480;

    std::vector<TooN::SE3<> >poses2render;
    int render_pose_count = 0;

    float depth_arrayf[width*height];

    CVD::Image<u_int16_t>depth_image(CVD::ImageRef(width,height));

    float near =0.1;
    float far = 1000;

    srand (time(NULL));

    CVD::Image<CVD::Rgb<CVD::byte> > img_flipped(CVD::ImageRef(640,480));

    char trajectory_fileName[300];

//    bedroom1_poses_0_interpolated.txt

    sprintf(trajectory_fileName,"%s/%s_poses_0.txt",
            data_dir.c_str(),
            obj_basename.c_str());

    ifstream SE3PoseFile(trajectory_fileName);


    TooN::SE3<>T_wc;

    if (SE3PoseFile.is_open())
    {
        while(1)
        {
            SE3PoseFile >> T_wc;

            if ( SE3PoseFile.eof() )
                break;

            poses2render.push_back(T_wc);

        }

    }
    SE3PoseFile.close();

    OpenGlMatrix openglSE3Matrix;

    int skip_frame = 1;

    ofstream model_file("3dmodel.obj");

    float u0 = 320.0;
    float v0 = 240.0;
    float fx = 420.0;
    float fy =-420.0;

    // Default hooks for exiting (Esc) and fullscreen (tab).
    while( !pangolin::ShouldQuit() )
    {
        static Var<int>numposes2plot("ui.numposes2plot",0,0,100);

        {
            numposes2plot  = render_pose_count;

            if ( numposes2plot >= (int)poses2render.size())
                return 1;

            TooN::SE3<>T_wc = poses2render.at(render_pose_count);

            TooN::SE3<>T_cw = T_wc.inverse();

            TooN::SO3<>Rot = T_cw.get_rotation();

            TooN::Matrix<4>SE3Mat = TooN::Identity(4);

            /// copy rotation
            TooN::Matrix<3>SO3Mat = Rot.get_matrix();
            SE3Mat.slice(0,0,3,3) = SO3Mat;

            /// copy translation
            TooN::Vector<3>trans = T_cw.get_translation();
            SE3Mat(0,3) = trans[0];
            SE3Mat(1,3) = trans[1];
            SE3Mat(2,3) = trans[2];

            /// Ref: http://www.felixgers.de/teaching/jogl/generalTransfo.html
            /// It should be a transpose - stored in column major
            for(int col = 0; col < 4; col++ )
            {
                for(int row = 0; row < 4; row++)
                {
                    openglSE3Matrix.m[col*4+row] = SE3Mat(row,col);
                }
            }

            /// set the model view matrix to this pose
            stacks3d.SetModelViewMatrix(openglSE3Matrix);

            stacks3d.Apply();

            numposes2plot = render_pose_count;

            // Clear whole screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            //        d_cam.ActivateAndScissor();
            d_cam.ActivateScissorAndClear(stacks3d);

            glEnable(GL_DEPTH_TEST);

            glClear(GL_COLOR_BUFFER_BIT);

#ifdef GENERATE_RGBD_VIDEO
            glMesh.DrawCanonicalObject();
#else
            glMesh.DrawCanonicalObjectSegmentation(renderingColours);
#endif

            CVD::Image<CVD::Rgb<CVD::byte> > img = CVD::glReadPixels<CVD::Rgb<CVD::byte> >(CVD::ImageRef(640,480),
                                                                                           CVD::ImageRef(150,0));

            /// save the image
//#pragma omp parallel for
            for(int yy = 0; yy < height; yy++ )
            {
                for(int xx = 0; xx < width; xx++)
                {
                    img_flipped[CVD::ImageRef(xx,height-1-yy)] = img[CVD::ImageRef(xx,yy)];
                }
            }

            char fileName[300];

            sprintf(fileName,"%s/scene_00_%07d.png",data_dir.c_str(),render_pose_count/skip_frame);

            CVD::img_save(img_flipped,fileName);

//            CVD::Image<CVD::byte>labelImage = CVD::Image<CVD::byte>(CVD::ImageRef(640,480));

//            /// save the annotations
//#pragma omp parallel for
//            for(int yy = 0; yy < height; yy++ )
//            {
//                for(int xx = 0; xx < width; xx++)
//                {
//                    CVD::Rgb<CVD::byte> pix = img_flipped[CVD::ImageRef(xx,yy)];

//                    int ind = pix.red + 256*pix.green + 256*256*pix.blue;

//                    labelImage[CVD::ImageRef(xx,yy)] = colour2indexMap[ind];
//                }
//            }

//            sprintf(fileName,"%s/label_00_%07d.png",data_dir.c_str(),render_pose_count/skip_frame);
//            CVD::img_save(labelImage,fileName);

            glReadPixels(150, 0, 640, 480, GL_DEPTH_COMPONENT, GL_FLOAT, depth_arrayf);

            int scale = 5000;

            /// convert to real-depth
//#pragma omp parallel for
            for(int i = 0; i < width*height; ++i)
            {
                float z_b = depth_arrayf[i];
                float z_n = 2.0f * z_b - 1.0f;
                depth_arrayf[i] = 2.0 * near * far / (far + near - z_n * (far - near));
            }

            /// save the depth image
//#pragma omp parallel for
            for(int y = 0; y < height; y++)
            {
                for(int x = 0; x < width; x++)
                {
                    int ind = (height-1-y)*width+x;
                    float depth_val = depth_arrayf[ind];

                    u_int16_t val = (u_int16_t)(depth_val * scale);

                    if( val < 65535 && depth_val < 100)
                        depth_image[CVD::ImageRef(x,y)] = val;
                    else
                        depth_image[CVD::ImageRef(x,y)] = 0 ;
                }
            }

            char depthImageFileName[300];

            {
                sprintf(depthImageFileName,"%s/scenedepth_00_%07d.png",data_dir.c_str(),render_pose_count/skip_frame);
            }

            float rand_r = (float)rand()/RAND_MAX;
            float rand_g = (float)rand()/RAND_MAX;
            float rand_b = (float)rand()/RAND_MAX;

            if( render_pose_count%1000 == 0  )
            {

                float max_depth = *std::max_element(depth_arrayf,depth_arrayf+width*height);
                float min_depth = *std::min_element(depth_arrayf,depth_arrayf+width*height);

                std::cout<<"max_depth = " << max_depth << std::endl;
                std::cout<<"min_depth = " << min_depth << std::endl;

                for(int y = 0; y < height; y++)
                {
                    for(int x = 0; x < width; x++)
                    {
                        float depth = (float)depth_image[CVD::ImageRef(x,y)]/scale;

                        if ( depth < 10 && depth > 0 )
                        {
                            TooN::Vector<4>p_w = T_wc * TooN::makeVector(depth*(x-u0)/fx,
                                                                         depth*(y-v0)/fy,
                                                                         depth,
                                                                         1.0);

                            model_file << "v " << p_w[0]<<" "<<p_w[1]<<" "<<p_w[2]<<" "<<rand_r<<" "<<rand_g<<" "<<rand_b<<std::endl;

                        }
                    }
                }
            }


            std::string depthImageFileName_string(depthImageFileName);

            CVD::img_save(depth_image,depthImageFileName_string);

            render_pose_count++;


            d_panel.Render();

            // Swap frames and Process Events
            pangolin::FinishFrame();
        }

        // Pause for 1/60th of a second.
//        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / 60));
    }

    return 0;
}
