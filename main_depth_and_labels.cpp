#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <TooN/TooN.h>

#include <pangolin/pangolin.h>
#include <pangolin/display.h>
#include <cvd/image_io.h>
#include "utils/povray_utils.h"

#include "tinyobjloader/tiny_obj_loader.h"
#include "utils/map_object_label2training_label.h"


using namespace pangolin;

static void PrintInfo(const std::vector<tinyobj::shape_t>& shapes, const std::vector<tinyobj::material_t>& materials)
{
  std::cout << "# of shapes    : " << shapes.size() << std::endl;
  std::cout << "# of materials : " << materials.size() << std::endl;

  for (size_t i = 0; i < shapes.size(); i++) {
    printf("shape[%ld].name = %s\n", i, shapes[i].name.c_str());
    assert((shapes[i].mesh.indices.size() % 3) == 0);

  }

  for (size_t i = 0; i < materials.size(); i++) {
    printf("material[%ld].name = %s\n", i, materials[i].name.c_str());
    printf("  material.Ka = (%f, %f ,%f)\n", materials[i].ambient[0], materials[i].ambient[1], materials[i].ambient[2]);
    printf("  material.Kd = (%f, %f ,%f)\n", materials[i].diffuse[0], materials[i].diffuse[1], materials[i].diffuse[2]);
    printf("  material.Ks = (%f, %f ,%f)\n", materials[i].specular[0], materials[i].specular[1], materials[i].specular[2]);
    printf("  material.Tr = (%f, %f ,%f)\n", materials[i].transmittance[0], materials[i].transmittance[1], materials[i].transmittance[2]);
    printf("  material.Ke = (%f, %f ,%f)\n", materials[i].emission[0], materials[i].emission[1], materials[i].emission[2]);
    printf("  material.Ns = %f\n", materials[i].shininess);
    printf("  material.Ni = %f\n", materials[i].ior);
    printf("  material.dissolve = %f\n", materials[i].dissolve);
    printf("  material.illum = %d\n", materials[i].illum);
    printf("  material.map_Ka = %s\n", materials[i].ambient_texname.c_str());
    printf("  material.map_Kd = %s\n", materials[i].diffuse_texname.c_str());
    printf("  material.map_Ks = %s\n", materials[i].specular_texname.c_str());
    printf("  material.map_Ns = %s\n", materials[i].normal_texname.c_str());
    std::map<std::string, std::string>::const_iterator it(materials[i].unknown_parameter.begin());
    std::map<std::string, std::string>::const_iterator itEnd(materials[i].unknown_parameter.end());
    for (; it != itEnd; it++) {
      printf("  material.%s = %s\n", it->first.c_str(), it->second.c_str());
    }
    printf("\n");
  }
}

bool TestLoadObj(
  const char* filename,
//  const char* basepath = NULL,
  std::vector<tinyobj::shape_t>& shapes,
  std::vector<tinyobj::material_t>& materials)
{
  std::cout << "Loading " << filename << std::endl;

//  std::vector<tinyobj::shape_t> shapes;
//  std::vector<tinyobj::material_t> materials;
  std::string err = tinyobj::LoadObj(shapes, materials, filename, NULL);

  if (!err.empty()) {
    std::cerr << err << std::endl;
    return false;
  }

  PrintInfo(shapes, materials);

  return true;
}


#define RADPERDEG 0.0174533

void Arrow(GLdouble x1,GLdouble y1,GLdouble z1,GLdouble x2,GLdouble y2,GLdouble z2,GLdouble D)
{
  double x=x2-x1;
  double y=y2-y1;
  double z=z2-z1;
  double L=sqrt(x*x+y*y+z*z);

    GLUquadricObj *quadObj;

    glPushMatrix ();

      glTranslated(x1,y1,z1);

      if((x!=0.)||(y!=0.)) {
        glRotated(atan2(y,x)/RADPERDEG,0.,0.,1.);
        glRotated(atan2(sqrt(x*x+y*y),z)/RADPERDEG,0.,1.,0.);
      } else if (z<0){
        glRotated(180,1.,0.,0.);
      }

      glTranslatef(0,0,L-4*D);

      quadObj = gluNewQuadric ();
      gluQuadricDrawStyle (quadObj, GLU_FILL);
      gluQuadricNormals (quadObj, GLU_SMOOTH);
      gluCylinder(quadObj, 2*D, 0.0, 4*D, 32, 1);
      gluDeleteQuadric(quadObj);

      quadObj = gluNewQuadric ();
      gluQuadricDrawStyle (quadObj, GLU_FILL);
      gluQuadricNormals (quadObj, GLU_SMOOTH);
      gluDisk(quadObj, 0.0, 2*D, 32, 1);
      gluDeleteQuadric(quadObj);

      glTranslatef(0,0,-L+4*D);

      quadObj = gluNewQuadric ();
      gluQuadricDrawStyle (quadObj, GLU_FILL);
      gluQuadricNormals (quadObj, GLU_SMOOTH);
      gluCylinder(quadObj, D, D, L-4*D, 32, 1);
      gluDeleteQuadric(quadObj);

      quadObj = gluNewQuadric ();
      gluQuadricDrawStyle (quadObj, GLU_FILL);
      gluQuadricNormals (quadObj, GLU_SMOOTH);
      gluDisk(quadObj, 0.0, D, 32, 1);
      gluDeleteQuadric(quadObj);

    glPopMatrix ();

}
void drawAxes(GLdouble length)
{
    glPushMatrix();
    glColor3f(1.0,0,0);
    glTranslatef(-length,0,0);
    Arrow(0,0,0, 2*length,0,0, 0.1);
    glPopMatrix();

    glPushMatrix();
    glColor3f(0.0,1.0,0);
    glTranslatef(0,-length,0);
    Arrow(0,0,0, 0,2*length,0, 0.1);
    glPopMatrix();

    glPushMatrix();
    glColor3f(0.0,0.0,1.0);
    glTranslatef(0,0,-length);
    Arrow(0,0,0, 0,0,2*length, 0.1);
    glPopMatrix();
}


void change_basis(TooN::SE3<>& T_wc_ref,
                  TooN::Matrix<4>&T)
{
    TooN::Matrix<4>T4x4 = T.T() * T_wc_ref * T  ;

    TooN::Matrix<3>R_slice = TooN::Data(T4x4(0,0),T4x4(0,1),T4x4(0,2),
                                        T4x4(1,0),T4x4(1,1),T4x4(1,2),
                                        T4x4(2,0),T4x4(2,1),T4x4(2,2));


    TooN::Vector<3>t_slice = TooN::makeVector(T4x4(0,3),T4x4(1,3),T4x4(2,3));

    T_wc_ref = TooN::SE3<>(TooN::SO3<>(R_slice),t_slice);

}

int main(int argc, char *argv[])
{

    if ( argc < 2)
    {
        std::cerr<<"Usage: ./binary_name obj_file_path " << std::endl;
        std::cerr<<"example: ./opengl_depth_rendering ../data/room_89_simple.obj" << std::endl;
        exit(1);
    }

    std::string obj_basename(argv[1]);
    std::size_t find_dot   = obj_basename.find(".obj");
    std::size_t find_slash = obj_basename.find_last_of('/');

    std::cout<<"  find_dot = " << find_dot << std::endl;
    std::cout<<"find_slash = " << find_slash << std::endl;

    obj_basename = obj_basename.substr(find_slash+1,find_dot-find_slash-1);

    std::string data_dir = "../data/" + obj_basename + "_data";

    /// Simple OBJ reader - main code begins after this..

    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    TestLoadObj(argv[1],shapes,materials);

    std::vector<float*>shape_vertices(shapes.size(),NULL);

    /// Reading the obj mesh
    for(int i = 0; i < shape_vertices.size(); i++)
    {
        int num_vertices = shapes[i].mesh.positions.size()/3;
        int num_faces    = shapes[i].mesh.indices.size() / 3;

        shape_vertices[i] = new float[num_faces*3*3];

        int count=0;

        for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++)
        {
            int v1_idx = shapes[i].mesh.indices[3*f+0];
            int v2_idx = shapes[i].mesh.indices[3*f+1];
            int v3_idx = shapes[i].mesh.indices[3*f+2];

            int max_index = max(max(v1_idx,v2_idx),v3_idx);

            if ( max_index > num_vertices )
            {
                std::cerr<<"max_vertex_index exceeds the number of vertices, something fishy!" << std::endl;
                return 1;
            }

            shape_vertices[i][count+0] = shapes[i].mesh.positions[3*v1_idx+0];
            shape_vertices[i][count+1] = shapes[i].mesh.positions[3*v1_idx+1];
            shape_vertices[i][count+2] = shapes[i].mesh.positions[3*v1_idx+2];

            count+=3;

            shape_vertices[i][count+0] = shapes[i].mesh.positions[3*v2_idx+0];
            shape_vertices[i][count+1] = shapes[i].mesh.positions[3*v2_idx+1];
            shape_vertices[i][count+2] = shapes[i].mesh.positions[3*v2_idx+2];

            count+=3;

            shape_vertices[i][count+0] = shapes[i].mesh.positions[3*v3_idx+0];
            shape_vertices[i][count+1] = shapes[i].mesh.positions[3*v3_idx+1];
            shape_vertices[i][count+2] = shapes[i].mesh.positions[3*v3_idx+2];

            count+=3;

        }
    }

    /// Scale 1 means 640x480 images
    /// Scale 2 means 320x240 images

    int scale              = 1;

    int width              = 640/scale;
    int height             = 480/scale;



    int w_width  = 640;
    int w_height = 480;
    const int UI_WIDTH = 150;

    pangolin::CreateWindowAndBind("GUISandbox",w_width+150,w_height);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glewInit();

    /// Create a Panel
    pangolin::View& d_panel = pangolin::CreatePanel("ui")
            .SetBounds(1.0, 0.0, 0, pangolin::Attach::Pix(150));

    /// This is the one I used for rendering....

    float near_plane = 0.1;
    float far_plane  = 1000;


    pangolin::OpenGlRenderState s_cam(
      ProjectionMatrixRDF_BottomLeft(640,480,420.0,420.0,320,240,near_plane,far_plane),
      ModelViewLookAt(3,3,3, 0,0,0, AxisNegZ)
    );

    /// Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::Display("cam")
      .SetBounds(0.0, 1, Attach::Pix(UI_WIDTH), 1, -640.0f/480.0f)
      .SetHandler(new Handler3D(s_cam));


    OpenGlMatrix openglSE3Matrix;

    TooN::Matrix<17,3>colours = TooN::Data(0,0,1.0,
                                           0.9137,0.3490,0.1882,
                                           0,0.8549,0,
                                           0.5843,0,0.9412,
                                           0.8706,0.9451,0.0941,
                                           1.0000,0.8078,0.8078,
                                           0,0.8784,0.8980,
                                           0.4157,0.5333,0.8000,
                                           0.4588,0.1137,0.1608,
                                           0.9412,0.1333,0.9216,
                                           0,0.6549,0.6118,
                                           0.9765,0.5451,0,
                                           0.8824,0.8980,0.7608,
                                           1.0000,0,0,
                                           0.8118,0.7176,0.2706,
                                           0.7922,0.5804,0.5804,
                                           0.4902,0.4824,0.4784);


    std::map<int,int>colour2indexMap;
    /// NYU
//    colour2indexMap[  0 +   0*256 + 255*256*256] = 1;
//    colour2indexMap[233 +  89*256 +  48*256*256] = 2;
//    colour2indexMap[  0 + 218*256 +   0*256*256] = 3;
//    colour2indexMap[  0 +   0*256 +   0*256*256] = 4;
//    //Books: colour2indexMap[  0 +   0*256 +   0*256*256] = 5;
//    colour2indexMap[255 + 206*256 + 206*256*256] = 6;
//    colour2indexMap[  0 + 224*256 + 229*256*256] = 7;
//    colour2indexMap[106 + 136*256 + 204*256*256] = 8;
//    colour2indexMap[117 +  29*256 +  41*256*256] = 9;
////    colour2indexMap[  0 +   0*256 +   0*256*256] = 10;
//    colour2indexMap[  0 + 167*256 + 156*256*256] = 11;
//    colour2indexMap[249 + 139*256 +   0*256*256] = 12;
//    colour2indexMap[225 + 229*256 + 194*256*256] = 13;
//    colour2indexMap[255 +   0*256 +   0*256*256] = 14;

    ///SynthCam3D
    colour2indexMap[  0 +   0*256 + 255*256*256] = 1;
    colour2indexMap[233 +  89*256 +  48*256*256] = 2;
    colour2indexMap[  0 + 218*256 +   0*256*256] = 3;
    colour2indexMap[149 +   0*256 + 240*256*256] = 4;
    colour2indexMap[222 + 241*256 +  24*256*256] = 5;
    colour2indexMap[255 + 206*256 + 206*256*256] = 6; // FURNITURE
    colour2indexMap[  0 + 224*256 + 229*256*256] = 7;
    colour2indexMap[106 + 136*256 + 204*256*256] = 8; // SHELF
    colour2indexMap[117 +  29*256 +  41*256*256] = 9;
    colour2indexMap[240 +  34*256 + 235*256*256] = 10;
    colour2indexMap[  0 + 167*256 + 156*256*256] = 11;
//    colour2indexMap[249 + 139*256 +   0*256*256] = 12; //DOOR
    colour2indexMap[225 + 229*256 + 194*256*256] = 13-1;
    colour2indexMap[255 +   0*256 +   0*256*256] = 14-1;
    colour2indexMap[207 + 183*256 +  69*256*256] = 15-1;
    colour2indexMap[202 + 148*256 + 148*256*256] = 16-1;
    colour2indexMap[125 + 123*256 + 122*256*256] = 17-1;
    colour2indexMap[  0 + 0*256   +   0*256*256] = 17;

    std::vector<TooN::SE3<> >poses2render;

    int render_pose_count = 0;

    srand (time(NULL));


    CVD::Image<CVD::Rgb<CVD::byte> > img_flipped(CVD::ImageRef(640,480));

    char trajectory_fileName[300];

    sprintf(trajectory_fileName,"%s/%s_trajectory_random_poses_SE3_3x4.txt",
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

            //std::cout<<"T_wc = " <<T_wc << std::endl;

            poses2render.push_back(T_wc);

        }

    }
    SE3PoseFile.close();

    std::cout<<"Trajectory file = " << trajectory_fileName << std::endl;
    std::cout<<"poses2render.size() = " << poses2render.size() << std::endl;


    float depth_arrayf[width*height];
    CVD::Image<u_int16_t>depth_image(CVD::ImageRef(width,height));


    /// Render every 5th frame. Change this to 1 if you want to render every frame.
    int skip_frame = 5;

    std::set<int> unique_objects;

    bool inserted_objects=false;

    while(!pangolin::ShouldQuit())
    {
        static Var<int>numposes2plot("ui.numposes2plot",0,0,100);

        {

            numposes2plot  = render_pose_count;

            if ( numposes2plot >= poses2render.size())
                break;

            TooN::SE3<>T_wc = poses2render.at(render_pose_count);

            TooN::SE3<>T_cw = T_wc.inverse();

            TooN::SO3<>Rot = T_cw.get_rotation();
            TooN::Matrix<3>SO3Mat = Rot.get_matrix();
            TooN::Vector<3>trans = T_cw.get_translation();

            TooN::Matrix<4>SE3Mat = Identity(4);

            SE3Mat.slice(0,0,3,3) = SO3Mat;

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

            s_cam.SetModelViewMatrix(openglSE3Matrix);

            s_cam.Apply();

            numposes2plot = render_pose_count;

            d_cam.ActivateScissorAndClear(s_cam);


            /// Do the plotting of the 3D model

            glEnable(GL_DEPTH_TEST);

            glClear(GL_COLOR_BUFFER_BIT);

            glColor3f(1.0f,1.0f,1.0f);

            for(int i = 0 ; i < shapes.size();i++)
            {
                int training_label = obj_label2training_label(shapes[i].name);

                if ( !inserted_objects )
                {
                    unique_objects.insert(training_label);
                }

                glColor3f(colours(training_label,0),colours(training_label,1),colours(training_label,2));

                glEnableClientState(GL_VERTEX_ARRAY);

                glVertexPointer(3,GL_FLOAT,0,shape_vertices[i]);
                glDrawArrays(GL_TRIANGLES,0,shapes[i].mesh.indices.size());
                glDisableClientState(GL_VERTEX_ARRAY);

            }

            if ( !inserted_objects )
            {
                inserted_objects = true;
            }

            CVD::Image<CVD::Rgb<CVD::byte> > img = CVD::glReadPixels<CVD::Rgb<CVD::byte> >(CVD::ImageRef(640,480),
                                                                                           CVD::ImageRef(150,0));


#pragma omp parallel for
            for(int yy = 0; yy < height; yy++ )
            {
                for(int xx = 0; xx < width; xx++)
                {
                    img_flipped[CVD::ImageRef(xx,height-1-yy)] = img[CVD::ImageRef(xx,yy)];
                }
            }

            char fileName[500];

            sprintf(fileName,"%s/scene_00_%07d.png",data_dir.c_str(),render_pose_count/skip_frame);

            CVD::img_save(img_flipped,fileName);

            CVD::Image<CVD::byte>labelImage = CVD::Image<CVD::byte>(CVD::ImageRef(640,480));

            std::vector<int>objects_this_frame(width*height,0);

#pragma omp parallel for
            for(int yy = 0; yy < height; yy++ )
            {
                for(int xx = 0; xx < width; xx++)
                {
                    CVD::Rgb<CVD::byte> pix = img_flipped[CVD::ImageRef(xx,yy)];

                    int ind = pix.red + 256*pix.green + 256*256*pix.blue;

                    labelImage[CVD::ImageRef(xx,yy)] = colour2indexMap[ind];

                    objects_this_frame[yy*width+xx] = labelImage[CVD::ImageRef(xx,yy)];
                }
            }

            std::set<int> unique_objects_this_frame(objects_this_frame.begin(),objects_this_frame.end());

            std::cout <<"unique objects this frame = ";
            std::set<int>::iterator it;
            for (it=unique_objects_this_frame.begin(); it!=unique_objects_this_frame.end(); ++it)
            {
                std::cout << ' ' << *it;
            }
            std::cout<<std::endl;

            sprintf(fileName,"%s/label_00_%07d.png",data_dir.c_str(),render_pose_count/skip_frame);
            CVD::img_save(labelImage,fileName);


            /// Render depth

            glReadPixels(150, 0, 640, 480, GL_DEPTH_COMPONENT, GL_FLOAT, depth_arrayf);
            int scale = 5000;

            /// convert to real-depth
#pragma omp parallel for
            for(int i = 0; i < width*height; i++)
            {
                float z_b = depth_arrayf[i];
                float z_n = 2.0f * z_b - 1.0f;
                depth_arrayf[i] = 2.0 * near_plane * far_plane / (far_plane + near_plane - z_n * (far_plane - near_plane));
            }

            /// save the depth image
#pragma omp parallel for
            for(int y = 0; y < height; y++)
            {
                for(int x = 0; x < width; x++)
                {
                    int ind = (height-1-y)*width+x;
                    float depth_val = depth_arrayf[ind];

                    if( depth_val * scale < 65535 )
                        depth_image[CVD::ImageRef(x,y)] = (u_int16_t)(depth_val*scale);
                    else
                        depth_image[CVD::ImageRef(x,y)] = 65535 ;
                }
            }

            char depthImageFileName[500];

            {
                sprintf(depthImageFileName,"%s/scenedepth_00_%07d.png",data_dir.c_str(),render_pose_count/skip_frame);
            }

            std::string depthImageFileName_string(depthImageFileName);

            CVD::img_save(depth_image,depthImageFileName_string);


            /// Save the poses

            char poseFileName[300];

            sprintf(poseFileName,"%s/pose_%07d.txt",data_dir.c_str(),render_pose_count/skip_frame);

            ofstream ofile(poseFileName);

            ofile << T_wc << std::endl;

            ofile.close();

        }


        d_panel.Render();
        pangolin::FinishFrame();

        render_pose_count+=skip_frame;


    }

}


//            float projectionMatrix[16];
//            glGetFloatv(GL_PROJECTION_MATRIX, projectionMatrix);

//            for(int r = 0; r < 4; r++)
//            {
//                for(int c = 0; c < 4; c++)
//                {
//                    std::cout<<projectionMatrix[r*4+c]<<" ";
//                }
//                std::cout<<std::endl;
//            }


//            float modelviewMatrix[16];
//            glGetFloatv(GL_MODELVIEW_MATRIX, modelviewMatrix);
//            for(int c = 0; c < 4; c++)
//            {
//                for(int r = 0; r < 4; r++ )
//                {
//                    std::cout<<modelviewMatrix[r*4+c]<<" ";
//                }
//                std::cout<<std::endl;
//            }
