#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include <TooN/TooN.h>

#include <pangolin/pangolin.h>
#include <pangolin/display.h>

#include <cvd/image_io.h>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/random_number_generator.hpp>

#include "utils/map_object_label2training_label.h"
#include "utils/povray_utils.h"
#include "tinyobjloader/tiny_obj_loader.h"

#include "utils/mesh_utils.h"
//#include "utils/distinct_colours.h"
//#include "utils/simulated_annealing.h"
//#include "sceneGraph/sceneGraph.h"
//#include "utils/convex_hull.h"

using namespace pangolin;

static void PrintInfo(const std::vector<tinyobj::shape_t>& shapes,
                      const std::vector<tinyobj::material_t>& materials)
{
  std::cout << "# of shapes    : " << shapes.size() << std::endl;
  std::cout << "# of materials : " << materials.size() << std::endl;

  for (size_t i = 0; i < shapes.size(); i++) {
    printf("shape[%ld].name = %s\n", i, shapes[i].name.c_str());
//    printf("Size of shape[%ld].indices: %ld\n", i, shapes[i].mesh.indices.size());
//    printf("Size of shape[%ld].material_ids: %ld\n", i, shapes[i].mesh.material_ids.size());
    assert((shapes[i].mesh.indices.size() % 3) == 0);
//    for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
//      printf("  idx[%ld] = %d, %d, %d. mat_id = %d\n", f, shapes[i].mesh.indices[3*f+0], shapes[i].mesh.indices[3*f+1], shapes[i].mesh.indices[3*f+2], shapes[i].mesh.material_ids[f]);
//    }

//    printf("shape[%ld].vertices: %ld\n", i, shapes[i].mesh.positions.size());
//    assert((shapes[i].mesh.positions.size() % 3) == 0);
//    for (size_t v = 0; v < shapes[i].mesh.positions.size() / 3; v++) {
//      printf("  v[%ld] = (%f, %f, %f)\n", v,
//        shapes[i].mesh.positions[3*v+0],
//        shapes[i].mesh.positions[3*v+1],
//        shapes[i].mesh.positions[3*v+2]);
//    }
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


void setOpenglMatrix(OpenGlMatrix& openglSE3Matrix, TooN::SE3<>&T_wcam)
{
    TooN::SE3<>T_cw = T_wcam.inverse();

    TooN::SO3<>Rot = TooN::SO3<>(T_cw.get_rotation()) /** TooN::SO3<>(TooN::makeVector((float)rx,(float)ry,(float)rz))*/;
    TooN::Matrix<3>SO3Mat = Rot.get_matrix();
    TooN::Vector<3>trans = T_cw.get_translation();

    TooN::Matrix<4>SE3Mat = Identity(4);

    SE3Mat.slice(0,0,3,3) = SO3Mat;

    SE3Mat(0,3) = trans[0]/*+(float)tx*/;
    SE3Mat(1,3) = trans[1]/*+(float)ty*/;
    SE3Mat(2,3) = trans[2]/*+(float)tz*/;

    /// Ref: http://www.felixgers.de/teaching/jogl/generalTransfo.html
    /// It should be a transpose - stored in column major
    for(int col = 0; col < 4; col++ )
    {
        for(int row = 0; row < 4; row++)
        {
            openglSE3Matrix.m[col*4+row] = SE3Mat(row,col);
        }
    }
}

double checkDisplacement(TooN::SE3<>&T1, TooN::SE3<>&T2)
{
    TooN::Vector<6>t1_ln = T1.ln();
    TooN::Vector<6>t2_ln = T2.ln();

    return TooN::norm(t1_ln-t2_ln);
}




int main(int argc, char *argv[])
{
    std::cout<<"Able to compile and print" << std::endl;

    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string fileName = "../data/living-room.obj";

    TestLoadObj(fileName.c_str(),
                shapes,
                materials);

    std::cout<<"shapes = " << shapes.size() << std::endl;
    std::cout<<"materials = " << shapes.size() << std::endl;


    std::vector<float*>shape_vertices(shapes.size(),NULL);
    std::vector<float*>shape_normals(shapes.size(),NULL);


    /// Reading the obj mesh
    for(int i = 0; i < shape_vertices.size(); i++)
    {
        int num_vertices = shapes[i].mesh.positions.size()/3;
        int num_faces    = shapes[i].mesh.indices.size() / 3;

        shape_vertices[i] = new float[num_faces*3*3];
        shape_normals[i]  = new float[num_faces*3*3];


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

            shape_normals[i][count+0]  = shapes[i].mesh.normals[3*v1_idx+0];
            shape_normals[i][count+1]  = shapes[i].mesh.normals[3*v1_idx+1];
            shape_normals[i][count+2]  = shapes[i].mesh.normals[3*v1_idx+2];

            count+=3;

            shape_vertices[i][count+0] = shapes[i].mesh.positions[3*v2_idx+0];
            shape_vertices[i][count+1] = shapes[i].mesh.positions[3*v2_idx+1];
            shape_vertices[i][count+2] = shapes[i].mesh.positions[3*v2_idx+2];

            shape_normals[i][count+0]  = shapes[i].mesh.normals[3*v2_idx+0];
            shape_normals[i][count+1]  = shapes[i].mesh.normals[3*v2_idx+1];
            shape_normals[i][count+2]  = shapes[i].mesh.normals[3*v2_idx+2];

            count+=3;

            shape_vertices[i][count+0] = shapes[i].mesh.positions[3*v3_idx+0];
            shape_vertices[i][count+1] = shapes[i].mesh.positions[3*v3_idx+1];
            shape_vertices[i][count+2] = shapes[i].mesh.positions[3*v3_idx+2];

            shape_normals[i][count+0]  = shapes[i].mesh.normals[3*v3_idx+0];
            shape_normals[i][count+1]  = shapes[i].mesh.normals[3*v3_idx+1];
            shape_normals[i][count+2]  = shapes[i].mesh.normals[3*v3_idx+2];

            count+=3;

        }
    }


    ifstream ifile("../data/ross_trajectories_2015/bedroom5_poses_0.txt");

    TooN::SE3<>T_wc;
    std::vector< TooN::SE3<> > gtPoses;

    while(1)
    {
        if (ifile.eof())
            break;

        ifile >> T_wc;

//        std::cout<<"T_wc = " << T_wc << std::endl;

        gtPoses.push_back(T_wc);
    }

    ifile.close();



    std::vector<float>red_colours;
    std::vector<float>green_colours;
    std::vector<float>blue_colours;

    for(int i = 0; i < shapes.size(); i++)
    {
        float red   = (float)rand()/RAND_MAX;
        float green = (float)rand()/RAND_MAX;
        float blue  = (float)rand()/RAND_MAX;

        red_colours.push_back( red );
        green_colours.push_back( green );
        blue_colours.push_back( blue );
    }




//    /// Scale 1 means 640x480 images
//    /// Scale 2 means 320x240 images

    int w_width  = 640;
    int w_height = 480;
    const int UI_WIDTH = 150;

    glutInit(& argc, argv);

    pangolin::CreateWindowAndBind("GUISandbox",w_width+150,w_height);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glewInit();

    /// Create a Panel
    pangolin::View& d_panel = pangolin::CreatePanel("ui")
            .SetBounds(1.0, 0.0, 0, pangolin::Attach::Pix(150));


    pangolin::OpenGlRenderState browsing_cam;
    browsing_cam.SetProjectionMatrix(ProjectionMatrixRDF_BottomLeft(640, 480, 420, 420, 320, 320, 0.1, 1000.0));
    browsing_cam.SetModelViewMatrix(ModelViewLookAt(-0.63628,0.457925,10.21563,
                                                    0,0,0,AxisNegZ));



    pangolin::OpenGlRenderState s_cam(
      ProjectionMatrixRDF_BottomLeft(640,480,420.0,420.0,320,240,0.1, 1000.0),
      ModelViewLookAt(3,3,3, 0,0,0, AxisNegZ)
    );


    pangolin::View& display_browsing_cam = pangolin::Display("cam")
      .SetBounds(0.0, 1, Attach::Pix(UI_WIDTH), 1/*0.5*/, -640.0f/480.0f)
      .SetHandler(new Handler3D(browsing_cam));

//    std::cout<<"entering the while loop" << std::endl;


    std::vector<TooN::Vector<3> > sizes_bb(shapes.size());
    std::vector<TooN::Vector<3> > sizes_ctr(shapes.size());

    TooN::Vector<3>size_;
    TooN::Vector<3>center_;

//    std::vector< CVD::Rgb<float> > _object_colours;

//    for(int i = 0; i < grouped_object_vertices.size();i++)
//    {
//        int rand_val;
//        rand_val = rand()%63;
//        _object_colours.push_back(colorConverter(distinct_colours[rand_val]));
//    }

/////    http://www.sitmo.com/article/generating-random-numbers-in-c-with-boost/
//    typedef boost::random::mt19937                     ENG;    // Mersenne Twister
//    typedef boost::random::normal_distribution<double> DIST;   // Normal Distribution
//    typedef boost::random::variate_generator<ENG,DIST> GEN;    // Variate generator

    TooN::Matrix<>objectMetaData(shape_vertices.size(),6);

    for(int i = 0 ; i < shape_vertices.size();i++)
    {
        int num_faces    = shapes[i].mesh.indices.size() / 3;

        int mesh_size    = num_faces*3*3;

        get_bbox_vecf(shape_vertices.at(i),
                     mesh_size,
                     size_,
                     center_);

         sizes_bb.at(i)  = size_;
        sizes_ctr.at(i) = center_;

        objectMetaData(i,0) = size_[0];
        objectMetaData(i,1) = size_[1];
        objectMetaData(i,2) = size_[2];

        objectMetaData(i,3) = center_[0];
        objectMetaData(i,4) = center_[1];
        objectMetaData(i,5) = center_[2];

        std::cout<<"size = " << size_ << std::endl;
        std::cout<<"center = " << center_ << std::endl;
    }

    float new_val_x = 0, new_val_y = 0, new_val_z= 0;


    std::cout<<"number of poses = " << gtPoses.size() << std::endl;

    int render_pose_count=0;
    int numposes2plot = 0;

    CVD::Image<CVD::Rgb<CVD::byte> > img_flipped(CVD::ImageRef(640,480));
    OpenGlMatrix openglSE3Matrix;


    while(!pangolin::ShouldQuit())
    {

        static Var<float>x_min("ui.x_min",-4.935,0,5);
        static Var<float>x_max("ui.x_max",3.486,0,5);
        static Var<float>z_min("ui.z_min",-4.449,0,5);
        static Var<float>z_max("ui.z_max",4.645,0,5);
//        static Var<float>height("ui.height",0.3,0,5);
//        static Var<float>c_angle("ui.angle",0,-180,180);
//        static Var<int> max_iters("ui.max_iters",1000,1,25000);

        static Var<float>rand_x("ui.rand_x",0,0,5);
        static Var<float>rand_y("ui.rand_y",0,0,5);
        static Var<float>rand_z("ui.rand_z",0,0,5);
        static Var<float>vel("ui.vel",1,1,100);

        static Var<int>models2show ("ui.models2show",(int)shapes.size(),
                                    0,(int)shapes.size());

        static Var<float> line_width("ui.line_width",2,0,100);
        static Var<float> end_pt("ui.end_pt",0.1,0,10);
        static Var<bool>render_images("ui.render_images",false);

//        static Var<float>LightX("ui.x_Light",10,-10,10);
//        static Var<float>LightY("ui.y_Light",5.652,-10,10);
//        static Var<float>LightZ("ui.z_Light",7.981,-10,10);

//        static Var<float>transparency("ui.transparency",1,0,1);

//        static Var<bool> random_center("ui.random_center",false);
        static Var<bool> plot_3d_model("ui.plot_model",true);
//        static Var<bool> optimise("ui.optimise",false);
//        static Var<bool> rotate_random("ui.rotate_random",false);
        static Var<int> poses_size("ui.poses_size",0);
//        static Var<bool> write_file("ui.write_file",false);
//        static Var<bool> pairwise_flag("ui.pairwise",true);
//        static Var<bool> visibility_flag("ui.visibility",true);
//        static Var<bool> boundingbox_flag("ui.boundingbox",true);
//        static Var<bool> angle_flag("ui.angle_flag",true);


//         {

        if ( render_images )
        {

            numposes2plot  = render_pose_count;

            if ( numposes2plot >= gtPoses.size())
                break;

            TooN::SE3<>T_wc = TooN::SE3<>(TooN::SO3<>(), TooN::makeVector(-0.3203,
                                                                          -0.1365,
                                                                           0.1449)) * gtPoses.at(render_pose_count);

//            gtPoses.at(render_pose_count) = T_wc;

            TooN::SE3<>T_cw = T_wc.inverse();

            TooN::SO3<>Rot  = T_cw.get_rotation();
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

            display_browsing_cam.ActivateScissorAndClear(s_cam);

        }

        else
        {

            display_browsing_cam.ActivateScissorAndClear(browsing_cam);
        }

            glEnable(GL_DEPTH_TEST);
            glClear(GL_COLOR_BUFFER_BIT);
            glColor3f(1.0f,1.0f,1.0f);

//            /// Code to plot each of the objects in the mesh
            if( plot_3d_model )
            {
                for(int i = 0; i < std::min((int)models2show,(int)shapes.size()-1); i++)
                {

                    if (!render_images && (i == 28 || i == 29)  /*room enclosure*/)
                        continue;

                    int num_faces    = shapes[i].mesh.indices.size() / 3;
                    int mesh_size    = num_faces*3*3;

                    std::string shape_name = shapes[i].name;
//                    std::cout<<"shape_name = " << shape_name << std::endl;

                    TooN::Vector<3>dims = sizes_bb.at(i);
                    {
                        dims = sizes_bb.at(i);

                        int num_faces    = shapes[i].mesh.indices.size() / 3;

                        float* vertices  = new float[num_faces*3*3];

                        for(int p =0; p < num_faces*3; p++)
                        {
                            vertices[3*p+0] = shape_vertices.at(i)[3*p+0] - sizes_ctr.at(i)[0];// + new_val_x;
                            vertices[3*p+1] = shape_vertices.at(i)[3*p+1] - sizes_ctr.at(i)[1];// + new_val_y;
                            vertices[3*p+2] = shape_vertices.at(i)[3*p+2] - sizes_ctr.at(i)[2];// + new_val_z;
                        }

                        glPushMatrix();


//                        TooN::Vector<3>center_random = TooN::makeVector(((float)rand()/RAND_MAX)*(x_max-x_min)+x_min,
//                                                            ((float)rand()/RAND_MAX)*(z_max-z_min)+z_min,/*sizes_bb.at(i)[1],*/
//                                                            dims[2]);

//                        center_random[0] = center_random[0];
//                        center_random[1] = center_random[1];


//                        if ( shape_name.find("chair1") != std::string::npos )
//                        {

//                            sizes_ctr.at(i)[0] = (float)x_min;
//                            sizes_ctr.at(i)[2] = (float)z_min;

//                            glTranslatef((float)x_min,
//                                          sizes_ctr.at(i)[1],
//                                         (float)z_min);

//                            std::cout<<"size_ctr.at(i)[0] = " << sizes_ctr.at(i)[0] << std::endl;
//                            std::cout<<"size_ctr.at(i)[2] = " << sizes_ctr.at(i)[2] << std::endl;

//                        }
//                        else
                        {
                            glTranslatef(sizes_ctr.at(i)[0],
                                         sizes_ctr.at(i)[1],
                                         sizes_ctr.at(i)[2]);
                        }

//                        TooN::Vector<3>ctr = TooN::makeVector(0,0,
//                                                              dims[2]/2);
//                        draw_bbox_only(dims,
//                                       ctr);

                        glColor3f(red_colours.at(i),
                                  green_colours.at(i),
                                  blue_colours.at(i));

                        glEnableClientState(GL_VERTEX_ARRAY);

                        glVertexPointer(3,GL_FLOAT,0,vertices);
//                        glVertexPointer(3,GL_FLOAT,0,shape_vertices[i]);
//                        glDrawArrays(GL_TRIANGLES,0,grouped_object_vertices.at(i).size()/3);
                        glDrawArrays(GL_TRIANGLES,0, mesh_size/3);
//                        glDrawArrays(GL_TRIANGLES,0, num_vertices*3);
                        glDisableClientState(GL_VERTEX_ARRAY);

                        glPopMatrix();

                        delete vertices;

                    }

                }

            }

            if ( render_images )
            {
//                CVD::Image<CVD::Rgb<CVD::byte> > img = CVD::glReadPixels<CVD::Rgb<CVD::byte> >(CVD::ImageRef(640,480),
//                                                                                               CVD::ImageRef(150,0));


//                int height = 480;
//                int width  = 640;

//#pragma omp parallel for
//                for(int yy = 0; yy < height; yy++ )
//                {
//                    for(int xx = 0; xx < width; xx++)
//                    {
//                        img_flipped[CVD::ImageRef(xx,height-1-yy)] = img[CVD::ImageRef(xx,yy)];
//                    }
//                }

//                char fileName[500];

//                sprintf(fileName,"scene_00_%07d.png",render_pose_count);

//                CVD::img_save(img_flipped,fileName);

                render_pose_count++;
            }

            if ( !render_images)
            {
            for(int i =0; i < gtPoses.size();i++)
            {
                TooN::SE3<>T_wc = TooN::SE3<>(TooN::SO3<>(), TooN::makeVector((float)rand_x,
                                                                              (float)rand_y,
                                                                              (float)rand_z));

                povray_utils::DrawCamera(T_wc * gtPoses.at(i),(float)end_pt, (float)line_width, false);
            }
            }

            poses_size = (int)gtPoses.size();

//            if ( optimise )
//            {
//                while(1)
//                {
//                    static int count = 0;

//                    srand(time(NULL));

//                    x_dims = TooN::makeVector((float)x_min,
//                                              (float)x_max);


//                    z_dims = TooN::makeVector((float)z_min,
//                                              (float)z_max);

//                    simulated_annealing_objects(vertex_pos,
//                                                sizes_bb,
//                                                vertex_rot,
//                                                bb_constraints,
//                                                pw_constraints,
//                                                angle_constraints,
//                                                access_constraints,
//                                                x_dims,
//                                                z_dims,
//                                                scene_graph.modelNames,
//                                                pairwise_flag,
//                                                visibility_flag,
//                                                boundingbox_flag,
//                                                angle_flag,
//                                                (int)max_iters);

//                    std::cout<<"Entering here.. " << count++ << std::endl;


//                    if (valid_configuration(vertex_pos,
//                                            sizes_bb,
//                                            access_constraints,
//                                            vertex_rot,
//                                            bb_constraints)  )
//                    {
////                        optimise = false;

//                            break;
//                    }
//                }

//                optimised = true;
//                optimise  = false;


//            }

//            if ( optimised )
//            {
//                plot_3d_model = false;

//                /// Enabling alpha channel

//                x_dims = TooN::makeVector((float)x_min,
//                                          (float)x_max);


//                z_dims = TooN::makeVector((float)z_min,
//                                          (float)z_max);

//                glEnable( GL_BLEND );
//                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

////                glColor4f(1.0,1.0,0.0,0.3);
////                glColor3f(1.0,1.0,0.0);
////                glRectf(x_dims[0],z_dims[0],x_dims[1],z_dims[1]);
////                glDisable(GL_BLEND);

////                draw_bbox_only(dims,
////                               ctr);

//                glColor3f(0.0,1.0,1.0);

//                glBegin(GL_LINE_LOOP);

//                glVertex2f(x_dims[0],z_dims[0]);
//                glVertex2f(x_dims[0],z_dims[1]);
//                glVertex2f(x_dims[1],z_dims[1]);
//                glVertex2f(x_dims[1],z_dims[0]);

//                glEnd();

//                GLfloat lightKa[] = {1.f, 0.2f, 1.f, 1.0f};  // ambient light
//                GLfloat lightKd[] = {1.f, 1.f, 1.f, 1.0f};  // diffuse light
//                GLfloat lightKs[] = {1, 1, 1, 1};           // specular light

//                /// Enabling the light

//                GLfloat _aLight[4];
//                // position the light
//                _aLight[0] = (float)LightX;
//                _aLight[1] = (float)LightY;
//                _aLight[2] = (float)LightZ;
//                _aLight[3] = 1.0f;

//                glEnable(GL_COLOR_MATERIAL);
//                glEnable(GL_LIGHTING);
//                glEnable(GL_LIGHT0);                        // MUST enable each light source after configuration
//                glLightfv(GL_LIGHT0, GL_POSITION,_aLight);
//                glLightfv(GL_LIGHT0, GL_AMBIENT, lightKa);
//                glLightfv(GL_LIGHT0, GL_DIFFUSE, lightKd);
//                glLightfv(GL_LIGHT0, GL_SPECULAR, lightKs);

//                float colorMaterials[] = { 1.0f, 1.0f, 1.0f, 1.0f };
//                glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, colorMaterials);

//                std::vector<Point>input_points;

//                int table_ind = 0;
//                for(int i = 0; i < grouped_object_vertices.size(); i++)
//                {
//                    if ( scene_graph.modelNames.at(i).find("table") != std::string::npos)
//                    {
//                        table_ind = i;
//                        break;
//                    }
//                }

//                for(int i = 0; i < min((int)models2show,(int)grouped_object_vertices.size());i++)
//                {

//                    int training_label = obj_label2training_label(scene_graph.modelNames.at(i));

//                    TooN::Vector<3>dims = sizes_bb.at(i);

//                    {
//                        float* vertices = new float[grouped_object_vertices.at(i).size()];
//                        float* normals  = new float[grouped_object_normals.at(i).size()];


//                        for(int p =0; p < grouped_object_vertices.at(i).size()/3; p++)
//                        {
//                            vertices[3*p+0] = (grouped_object_vertices.at(i).at(3*p+0));
//                            vertices[3*p+1] = (grouped_object_vertices.at(i).at(3*p+1));
//                            vertices[3*p+2] = (grouped_object_vertices.at(i).at(3*p+2));
//                        }

//                        if (grouped_object_normals.at(i).size() ==  grouped_object_vertices.at(i).size())
//                        {
//                            for(int vn = 0; vn < grouped_object_normals.at(i).size()/3; vn++)
//                            {
//                                normals[3*vn+0] = grouped_object_normals.at(i).at(3*vn+0);
//                                normals[3*vn+1] = grouped_object_normals.at(i).at(3*vn+1);
//                                normals[3*vn+2] = grouped_object_normals.at(i).at(3*vn+2);
//                            }

//                        }

//                        dims = sizes_bb.at(i);
//                        glPushMatrix();

//                        glTranslatef(vertex_pos.at(i)[0],
//                                     vertex_pos.at(i)[1],
//                                     0);

//                        Point p;
//                        p.x = vertex_pos.at(i)[0]-dims[0]/2;
//                        p.y = vertex_pos.at(i)[1]-dims[1]/2;
//                        input_points.push_back(p);

//                        p.x = vertex_pos.at(i)[0]+dims[0]/2;
//                        p.y = vertex_pos.at(i)[1]-dims[1]/2;
//                        input_points.push_back(p);

//                        p.x = vertex_pos.at(i)[0]-dims[0]/2;
//                        p.y = vertex_pos.at(i)[1]+dims[1]/2;
//                        input_points.push_back(p);

//                        p.x = vertex_pos.at(i)[0]+dims[0]/2;
//                        p.y = vertex_pos.at(i)[1]+dims[1]/2;
//                        input_points.push_back(p);


//                        float current_angle = vertex_rot.at(i);

//                        if ( scene_graph.modelNames.at(i).find("chair") != std::string::npos
//                             /*&& rotate_random*/ )

//                        {
//                             float diff_y = vertex_pos.at(table_ind)[1]-
//                                     vertex_pos.at(i)[1];
//                             float diff_x = vertex_pos.at(table_ind)[0]-
//                                     vertex_pos.at(i)[0];

//                             /// Measuring angle with negative y-axis
//                             float angle = atan2(diff_x,-diff_y)*180/M_PI;

//                            current_angle = (float)c_angle;

//                            if ( rotate_random )
//                            {
//                                current_angle = (float)angle;//(-angle+180.f);
//                            }

//                        }

//                        if (scene_graph.modelNames.at(i).find("sofa") != std::string::npos )
//                        {
//                                if( vertex_pos.at(table_ind)[0] > vertex_pos.at(i)[0] )
//                                {
//                                    current_angle = 180.0f;
//                                }
//                        }

//                        if (scene_graph.modelNames.at(i).find("night_stand") != std::string::npos )
//                        {
//                            current_angle = 90.0f;
//                        }

//                        if (scene_graph.modelNames.at(i).find("tv_set") != std::string::npos )
//                        {
////                            current_angle = 90.0f;

//                            /// Find the nearest wall
//                            float dist_x_max = fabs(vertex_pos.at(i)[0]-x_dims[1]);
//                            float dist_x_min = fabs(vertex_pos.at(i)[0]-x_dims[0]);

//                            float dist_y_max = fabs(vertex_pos.at(i)[1]-z_dims[1]);
//                            float dist_y_min = fabs(vertex_pos.at(i)[1]-z_dims[0]);

//                            float min_d = min(min(dist_x_min,dist_x_max),min(dist_y_max,dist_y_min));


//                            current_angle = 90.0f; (float)c_angle;

//                        }


//                        glRotatef(current_angle,0,0,1);

//                        vertex_rot.at(i) = current_angle;

//                        TooN::Vector<3>ctr = TooN::makeVector(0,0,
//                                                              dims[2]/2);

//                        draw_bbox_only(dims,
//                                       ctr);

//                        glColor3f(colours(training_label,0),colours(training_label,1),colours(training_label,2));


//                        if ( grouped_object_normals.at(i).size() )
//                        {
//                            glEnableClientState(GL_NORMAL_ARRAY);
//                            glNormalPointer(GL_FLOAT,3*sizeof(float),normals);
//                        }

//                        glEnableClientState(GL_VERTEX_ARRAY);

//                        glVertexPointer(3,GL_FLOAT,0,vertices);
//                        glDrawArrays(GL_TRIANGLES,0,grouped_object_vertices.at(i).size()/3);
//                        glDisableClientState(GL_VERTEX_ARRAY);

//                        glPopMatrix();

//                        if ( grouped_object_normals.at(i).size() )
//                        {
//                            glDisableClientState(GL_NORMAL_ARRAY);
//                        }


//                        delete vertices;
//                        delete normals;
//                    }

//                }

//                std::vector<Point>_convexHull = convex_hull(input_points);

//                glBegin(GL_POLYGON);
//                glColor4f(0.0,0.0,1.0,(float)transparency);
//                for(int p=0; p<_convexHull.size()-1; p++)
//                {
//                    float _x = _convexHull.at(p).x;
//                    float _y = _convexHull.at(p).y;
//                    glVertex3f((GLfloat)_x, (GLfloat)_y,0.0f);
//                }
//                glEnd();

//                /// Wall
//                std::string label = std::string("wall");
//                int training_label = obj_label2training_label(label);
//                glBegin(GL_POLYGON);
//                glColor3f(colours(training_label,0),colours(training_label,1),colours(training_label,2));
//                glVertex3f((float)x_min,(float)z_min,0);
//                glVertex3f((float)x_max,(float)z_min,0);
//                glVertex3f((float)x_max,(float)z_min,(float)height);
//                glVertex3f((float)x_min,(float)z_min,(float)height);
//                glEnd();

//                glBegin(GL_POLYGON);
////                glColor4f((float)wall_red,(float)wall_green,(float)wall_blue,(float)transparency);
//                glVertex3f((float)x_min,(float)z_max,0);
//                glVertex3f((float)x_max,(float)z_max,0);
//                glVertex3f((float)x_max,(float)z_max,(float)height);
//                glVertex3f((float)x_min,(float)z_max,(float)height);
//                glEnd();

//                glBegin(GL_POLYGON);
////                glColor4f((float)wall_red,(float)wall_green,(float)wall_blue,(float)transparency);
//                glVertex3f((float)x_min,(float)z_min,0);
//                glVertex3f((float)x_min,(float)z_max,0);
//                glVertex3f((float)x_min,(float)z_max,(float)height);
//                glVertex3f((float)x_min,(float)z_min,(float)height);
//                glEnd();


//                glBegin(GL_POLYGON);
////                glColor4f((float)wall_red,(float)wall_green,(float)wall_blue,(float)transparency);
//                glVertex3f((float)x_max,(float)z_min,0);
//                glVertex3f((float)x_max,(float)z_max,0);
//                glVertex3f((float)x_max,(float)z_max,(float)height);
//                glVertex3f((float)x_max,(float)z_min,(float)height);
//                glEnd();

//                /// Floor
//                glBegin(GL_POLYGON);
//                label = std::string("floor");
//                training_label = obj_label2training_label(label);
//                glColor3f(colours(training_label,0),colours(training_label,1),colours(training_label,2));
//                glVertex3f((float)x_min,(float)z_min,0);
//                glVertex3f((float)x_min,(float)z_max,0);
//                glVertex3f((float)x_max,(float)z_max,0);
//                glVertex3f((float)x_max,(float)z_min,0);
//                glEnd();

//                glDisable(GL_LIGHTING);
//            }

//            if ( write_file )
//            {

//                ofstream objects_graph_file;

//                obj_basename = std::string("");

//                for(int i = 0; i < scene_graph.modelNames.size();i++)
//                {
//                    std::size_t find_dot = scene_graph.modelNames.at(i).find(".obj");
//                    std::size_t find_slash = scene_graph.modelNames.at(i).find_last_of('/');
//                    obj_basename = obj_basename + std::string(scene_graph.modelNames.at(i).substr(find_slash+1,find_dot-find_slash-1));
//                }

//                if ( !have_written_obj_graph )
//                {
//                    obj_basename = std::string("scenes/obj_graph_file_") + obj_basename + std::string(".txt");
//                    objects_graph_file.open(obj_basename.c_str());
//                }

//                std::vector<Point>input_points;

//                for(int i = 0; i < (int)grouped_object_vertices.size();i++)
//                {
//                    objects_graph_file << scene_graph.modelNames.at(i)<< std::endl;
//                    objects_graph_file<< objectMetaData(i,0)<<" "<<objectMetaData(i,1)<<" "<<objectMetaData(i,2);
//                    objects_graph_file<<" "<<objectMetaData(i,3)<<" "<<objectMetaData(i,4)<<" "<<objectMetaData(i,5)<<" ";
//                    objects_graph_file << vertex_pos.at(i)[0]<<" "<<vertex_pos.at(i)[1]<<" "<<random_scales.at(i)<<std::endl;

//                    TooN::Vector<3>dims = sizes_bb.at(i);

//                    Point p;
//                    p.x = vertex_pos.at(i)[0]-dims[0]/2;
//                    p.y = vertex_pos.at(i)[1]-dims[1]/2;
//                    input_points.push_back(p);

//                    p.x = vertex_pos.at(i)[0]+dims[0]/2;
//                    p.y = vertex_pos.at(i)[1]-dims[1]/2;
//                    input_points.push_back(p);

//                    p.x = vertex_pos.at(i)[0]-dims[0]/2;
//                    p.y = vertex_pos.at(i)[1]+dims[1]/2;
//                    input_points.push_back(p);

//                    p.x = vertex_pos.at(i)[0]+dims[0]/2;
//                    p.y = vertex_pos.at(i)[1]+dims[1]/2;
//                    input_points.push_back(p);
//                }

//                std::vector<Point>_convexHull = convex_hull(input_points);

//                for(int p=0; p<_convexHull.size()-1; p++)
//                {
//                    float _x = _convexHull.at(p).x;
//                    float _y = _convexHull.at(p).y;
//                    objects_graph_file<<_x <<" "<<_y << std::endl;
//                }

//                objects_graph_file.close();

//                std::cout<<"total shapes = " << scene_graph._gshapes.size() << std::endl;

//                /// UGLY UGLY UGLY but it works...

//                std::string objmodel("optimised_model.obj");

//                for(int _shapes = 0; _shapes < scene_graph._gshapes.size(); _shapes++)
//                {
//                    std::vector<float>new_mesh;//(scene_graph._gshapes[_shapes].mesh.indices.size()*3);

//                    scene_graph._gshapes[_shapes].name = scene_graph.objectInScene.at(_shapes);

//                    float ctr_x = objectMetaData(_shapes,3);
//                    float ctr_y = objectMetaData(_shapes,4);
//                    float ctr_z = objectMetaData(_shapes,5);

//                    float dim2  = objectMetaData(_shapes,2)/2;

//                    for (size_t k = 0; k < scene_graph._gshapes[_shapes].mesh.indices.size() / 3; k++)
//                    {
//                        for (int j = 0; j < 3; j++)
//                        {
//                            int idx = scene_graph._gshapes[_shapes].mesh.indices[3*k+j];

//                            float x_c = (scene_graph._gshapes[_shapes].mesh.positions[3*idx+0]-ctr_x)*random_scales.at(_shapes);
//                            float y_c = (scene_graph._gshapes[_shapes].mesh.positions[3*idx+1]-ctr_y)*random_scales.at(_shapes);
//                            float z_c = (scene_graph._gshapes[_shapes].mesh.positions[3*idx+2]-(ctr_z - dim2))*random_scales.at(_shapes)/*+ vertex_pos.at(_shapes)*/;

//                            TooN::Vector<3>dir = TooN::makeVector(0,0,vertex_rot.at(_shapes)*M_PI/180.0f);

//                            TooN::SO3<>_SO3Mat(dir);

//                            TooN::Vector<3>xyz = TooN::makeVector(x_c,y_c,z_c);
//                            TooN::Vector<3>rotXYZ = _SO3Mat * xyz;

//                            float x_ = rotXYZ[0];
//                            float y_ = rotXYZ[1];
//                            float z_ = rotXYZ[2];

//                            new_mesh.push_back(x_ + vertex_pos.at(_shapes)[0]);
//                            new_mesh.push_back(y_ + vertex_pos.at(_shapes)[1]);
//                            new_mesh.push_back(z_);

//                        }
//                    }

//                    for (size_t k = 0; k < scene_graph._gshapes[_shapes].mesh.indices.size() / 3; k++)
//                    {
//                        for (int j = 0; j < 3; j++)
//                        {
//                            int idx = scene_graph._gshapes[_shapes].mesh.indices[3*k+j];

//                            scene_graph._gshapes[_shapes].mesh.positions[3*idx+0] = new_mesh.at(9*k+3*j+0);
//                            scene_graph._gshapes[_shapes].mesh.positions[3*idx+1] = new_mesh.at(9*k+3*j+1);
//                            scene_graph._gshapes[_shapes].mesh.positions[3*idx+2] = new_mesh.at(9*k+3*j+2);
//                        }
//                    }

//                    new_mesh.clear();
//                }

////                for(int p =0; p < grouped_object_vertices.at(i).size()/3; p++)
////                {
////                    grouped_object_vertices.at(i).at(3*p+0) = (grouped_object_vertices.at(i).at(3*p+0)-(sizes_ctr.at(i)[0]/* - dims[0]/2*/))*object_scale;
////                    grouped_object_vertices.at(i).at(3*p+1) = (grouped_object_vertices.at(i).at(3*p+1)-(sizes_ctr.at(i)[1]/* - dims[1]/2*/))*object_scale;
////                    grouped_object_vertices.at(i).at(3*p+2) = (grouped_object_vertices.at(i).at(3*p+2)-(sizes_ctr.at(i)[2] - dims[2]/2))*object_scale;
////                }

//                scene_graph.writeOBJmodel(objmodel,
//                                          scene_graph._gshapes,
//                                          scene_graph._gmaterials);
////                std::ofstream model_file("optimised_model.obj");

////                for(int i = 0; i < min((int)models2show,(int)grouped_object_vertices.size());i++)
////                {
////                    for(int p =0; p < grouped_object_vertices.at(i).size()/3; p++)
////                    {
////                        float x = (grouped_object_vertices.at(i).at(3*p+0))+vertex_pos.at(i)[0];
////                        float y = (grouped_object_vertices.at(i).at(3*p+1))+vertex_pos.at(i)[1];
////                        float z = (grouped_object_vertices.at(i).at(3*p+2));//+0;

////                        model_file <<"v " << x<<" "<<y<<" "<<z<<" "<<(float)_object_colours.at(i).red<<" "<<(float)_object_colours.at(i).green<<" "<<(float)_object_colours.at(i).blue<< std::endl;


////                    }
////                }

////                model_file.close();

//                write_file = false;
//            }

//            if ( show_size )
//            {
//                std::cout<<"size of " << models2show-1 << " is "<< sizes_bb.at(models2show-1)<<std::endl;
//                float diag = 0.5*sqrt(sizes_bb.at(models2show-1)[0]*sizes_bb.at(models2show-1)[0] + sizes_bb.at(models2show-1)[1]*sizes_bb.at(models2show-1)[1]);
//                std::cout<<" diagonal = " << diag << std::endl;
//                std::cout<<" vertex position = " << vertex_pos.at(models2show-1) << std::endl;
//                show_size = false;
//            }


//            static Var<float> end_pt("ui.end_pt",16.74,0,20);
//            static Var<float> line_width("ui.line_width",2,0,100);

//            povray_utils::DrawCamera(TooN::SE3<>(),end_pt, line_width,false);

            static Var<int> max_grid_unit("ui.grid_max",10,0,100);
            float grid_max = (int)max_grid_unit;

            const float sizeL = 3.f;
            const float grid = 2.f;

            glPointSize(sizeL);
            glBegin(GL_LINES);
            glColor3f(.25,.25,.25);
            for(float i=-grid_max; i<=grid_max; i+=grid)
            {
                glVertex3f(-grid_max, i, 0.f);
                glVertex3f(grid_max, i, 0.f);

                glVertex3f(i, -grid_max, 0.f);
                glVertex3f(i, grid_max, 0.f);
            }
            glEnd();


//            glColor3f(0.0,1.0,1.0);

//            glBegin(GL_LINE_LOOP);

//            glVertex2f(x_dims[0],z_dims[0]);
//            glVertex2f(x_dims[0],z_dims[1]);
//            glVertex2f(x_dims[1],z_dims[1]);
//            glVertex2f(x_dims[1],z_dims[0]);

//            glEnd();
//        }

        d_panel.Render();
        pangolin::FinishFrame();

    }

    for(int i =0 ; i < gtPoses.size();i++)
    {
        TooN::SE3<>T_wc = TooN::SE3<>(TooN::SO3<>(), TooN::makeVector(-0.3203,
                                                                      -0.1365,
                                                                      0.1449)) * gtPoses.at(i);

        gtPoses.at(i) = T_wc;

    }

    std::cout<<"Files have been written !" << std::endl;
    povray_utils::generate_POVRay_commands(gtPoses);
}

