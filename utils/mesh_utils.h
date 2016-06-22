#include <glm/glm.hpp>
#include <GL/gl.h>
#include <iostream>
#include "../tinyobjloader/tiny_obj_loader.h"
#include <TooN/TooN.h>

#include <unistd.h>
#include <iostream>
#include <fstream>

#include <vector_types.h>
#include <vector_functions.h>

#include <boost/chrono.hpp>
#include <boost/timer.hpp>

#include <TooN/se3.h>

#include <cvd/gl_helpers.h>

void getScalesForModels(std::map<std::string, float>& objects_scales,
                        std::map<std::string, float>& objects_scales_var)
{
    /// - minus sign for x and z, + for y axis

//    beds
//    chairs
//    tables
//    lamps
//    rack
//    shoes
//    bin
//    cupboard

#ifdef USE_ARCHIVE_3D_NET_DATA
    objects_scales["beds"] = 1.9;
    objects_scales["chairs"] = 1.0;
    objects_scales["tables"] = 0.9;
    objects_scales["lamps"] = 1.2;
    objects_scales["rack"] = 1.4;
    objects_scales["bin"] = 0.4;
    objects_scales["shoes"] = 0.3;
    objects_scales["cupboard"] = 2.3;
#else
//    objects_scales["bed"] = 0.7;
//    objects_scales["chair"] = 1.0;
//    objects_scales["table"] = 0.8;
//    objects_scales["night_stand"] = 0.3;
//    objects_scales["sofa"] = 0.5;
//    objects_scales["desk"] = 1.0;
//    objects_scales["dining"] = 1.0;

    objects_scales["cupboard"] = 2.0;

    /// Generated from NYU bedrooms with MATLAB

//    objects_scales["wall"] = 1.776425e+00;
//    objects_scales["floor"] = 3.779333e-01;

    objects_scales["cabinet"] = 9.515046e-01;
    objects_scales["bed"] = 7.914470e-01;
    objects_scales["chair"] = 7.549843e-01;
    objects_scales["sofa"] = 6.884906e-01;
    objects_scales["table"] = 5.116806e-01;
    objects_scales["book_shelf_stanford"] = 1.5;


//    objects_scales["door"] = 1.606927e+00;
//    objects_scales["window"] = 8.923701e-01;
//    objects_scales["bookshelf"] = 1.367899e+00;
//    objects_scales["picture"] = 6.509395e-01;
//    objects_scales["counter"] = 2.016806e-01;
//    objects_scales["blinds"] = 1.017973e+00;

    objects_scales["desk"] = 8.589204e-01;

//    objects_scales["shelves"] = 9.284758e-01;
    objects_scales["curtain"] = 2.0;//1.317684e+00;
//    objects_scales["dresser"] = 8.968198e-01;
//    objects_scales["pillow"] = 3.921272e-01;
//    objects_scales["mirror"] = 1.338894e+00;
//    objects_scales["floor mat"] = 2.113735e-01;
//    objects_scales["clothes"] = 5.034630e-01;
//    objects_scales["ceiling"] = 4.107350e-01;
//    objects_scales["books"] = 5.595141e-01;
//    objects_scales["refridgerator"] = 0;
//    objects_scales["tv"] = 2*5.918130e-01;
    objects_scales["tv_set"] = 1.450542;

//    objects_scales["paper"] = 2.084798e-01;
//    objects_scales["towel"] = 4.948436e-01;
//    objects_scales["shower curtain"] = 0;
//    objects_scales["box"] = 3.342348e-01;
//    objects_scales["whiteboard"] = 0;
//    objects_scales["person"] = 1.363450e+00;

    objects_scales["night_stand"] = 5.017400e-01;

//    objects_scales["toilet"] = 0;
//    objects_scales["sink"] = 0;

    objects_scales["lamp"] = 8.006681e-01;

    ///
    objects_scales["plant"] = 5.0e-01;
    objects_scales["clutter"] = 0.28;
    objects_scales["plate"] = 0.01;
    objects_scales["office_desk"] = 1.0;
    objects_scales["gym"] = 0.7;

    ///

//    objects_scales["bathtub"] = 0;
//    objects_scales["bag"] = 3.302822e-01;
//    objects_scales["otherstructure"] = 4.357814e-01;
//    objects_scales["otherfurniture"] = 7.144713e-01;
//    objects_scales["otherprop"] = 8.167548e-01;


//    objects_scales["bin"] = 0.4;
//    objects_scales["shoes"] = 0.3;
//    objects_scales["cupboard"] = 2.3;


    /// Standard deviation, Generated from NYU bedrooms again with MATLAB
//    objects_scales_var["wall"] = 1.594817e-01;
//    objects_scales_var["floor"] = 3.625356e-02;
//    objects_scales_var["cabinet"] = 2.949913e-01;
    objects_scales_var["bed"] = 9.412461e-02;
    objects_scales_var["chair"] = 3.234992e-02;
    objects_scales_var["sofa"] = 2.587976e-02;
    objects_scales_var["table"] = 4.789561e-02;
//    objects_scales_var["door"] = 1.397652e-01;
//    objects_scales_var["window"] = 1.974009e-01;
//    objects_scales_var["bookshelf"] = 1.670101e-01;
//    objects_scales_var["picture"] = 1.295109e-01;
//    objects_scales_var["counter"] = 1.436607e-02;
//    objects_scales_var["blinds"] = 1.822358e-01;
    objects_scales_var["desk"] = 1.677821e-01;
//    objects_scales_var["shelves"] = 2.999963e-01;
    objects_scales_var["curtain"] = 2.628007e-01;
//    objects_scales_var["dresser"] = 8.674244e-02;
//    objects_scales_var["pillow"] = 5.904925e-02;
//    objects_scales_var["mirror"] = 2.940230e-01;
//    objects_scales_var["floor mat"] = 2.095035e-02;
//    objects_scales_var["clothes"] = 1.600811e-01;
//    objects_scales_var["ceiling"] = 1.213000e-01;
//    objects_scales_var["books"] = 2.281456e-01;
//    objects_scales_var["refridgerator"] = 0;
    objects_scales_var["tv"] = 2.250542e-02;
    objects_scales_var["tv_set"] = 2.250542e-03;
//    objects_scales_var["paper"] = 4.537702e-02;
//    objects_scales_var["towel"] = 1.002915e-01;
//    objects_scales_var["shower curtain"] = 0;
//    objects_scales_var["box"] = 8.752702e-02;
//    objects_scales_var["whiteboard"] = 0;
//    objects_scales_var["person"] = 1.217047e-01;
    objects_scales_var["night_stand"] = 4.285187e-02;
//    objects_scales_var["toilet"] = 0;
//    objects_scales_var["sink"] = 0;
    objects_scales_var["lamp"] = 0.96080e-01;

    objects_scales_var["plant"] = 3.0e-02;
    objects_scales_var["clutter"] = 0.05;
    objects_scales_var["office_desk"] = 3e-02;
    objects_scales_var["plate"] = 0.01;
    objects_scales_var["gym"] = 0.01;
    objects_scales_var["book_shelf_stanford"] = 1e-3;




//    objects_scales_var["bathtub"] = 0;
//    objects_scales_var["bag"] = 8.586405e-02;
//    objects_scales_var["otherstructure"] = 2.525073e-01;
//    objects_scales_var["otherfurniture"] = 1.782654e-01;
//    objects_scales_var["otherprop"] = 2.676780e-01;

    objects_scales_var["cupboard"] = 9e-02;

#endif
    return;
}

void draw_bb_xyz_minmax(float x_min, float x_max,
                        float y_min, float y_max,
                        float z_min, float z_max)
{
    glBegin(GL_QUADS);        // Draw The Cube Using quads
        glColor4f(0.0f,1.0f,0.0f,0.6f);    // Color Blue

//                      glBegin(GL_LINE_LOOP);
        glVertex3f( x_max*1.0f, y_max*1.0f,z_min*1.0f);    // Top Right Of The Quad (Top)
        glVertex3f( x_min*1.0f, y_max*1.0f,z_min*1.0f);    // Top Left Of The Quad (Top)
        glVertex3f(x_min*1.0f, y_max*1.0f, z_max*1.0f);    // Bottom Left Of The Quad (Top)
        glVertex3f( x_max*1.0f, y_max*1.0f, z_max*1.0f);    // Bottom Right Of The Quad (Top)
//                      glEnd();

        glColor4f(1.0f,0.5f,0.0f,0.6f);    // Color Orange

//                      glBegin(GL_LINE_LOOP);
        glVertex3f( x_max*1.0f,y_min*1.0f, z_max*1.0f);    // Top Right Of The Quad (Bottom)
        glVertex3f(x_min*1.0f,y_min*1.0f, z_max*1.0f);    // Top Left Of The Quad (Bottom)
        glVertex3f(x_min*1.0f,y_min*1.0f,z_min*1.0f);    // Bottom Left Of The Quad (Bottom)
        glVertex3f( x_max*1.0f,y_min*1.0f,z_min*1.0f);    // Bottom Right Of The Quad (Bottom)
//                      glEnd();

        glColor4f(1.0f,0.0f,0.0f,0.6f);    // Color Red

//                      glBegin(GL_LINE_LOOP);
        glVertex3f( x_max*1.0f, y_max*1.0f, z_max*1.0f);    // Top Right Of The Quad (Front)
        glVertex3f(x_min*1.0f, y_max*1.0f, z_max*1.0f);    // Top Left Of The Quad (Front)
        glVertex3f(x_min*1.0f,y_min*1.0f, z_max*1.0f);    // Bottom Left Of The Quad (Front)
        glVertex3f( x_max*1.0f,y_min*1.0f, z_max*1.0f);    // Bottom Right Of The Quad (Front)
//                      glEnd();

        glColor4f(1.0f,1.0f,0.0f,0.6f);    // Color Yellow

//                      glBegin(GL_LINE_LOOP);
        glVertex3f( x_max*1.0f,y_min*1.0f,z_min*1.0f);    // Top Right Of The Quad (Back)
        glVertex3f(x_min*1.0f,y_min*1.0f,z_min*1.0f);    // Top Left Of The Quad (Back)
        glVertex3f(x_min*1.0f, y_max*1.0f,z_min*1.0f);    // Bottom Left Of The Quad (Back)
        glVertex3f( x_max*1.0f, y_max*1.0f,z_min*1.0f);    // Bottom Right Of The Quad (Back)
//                      glEnd();

        glColor4f(0.0f,0.0f,1.0f,0.6f);    // Color Blue

//                      glBegin(GL_LINE_LOOP);
        glVertex3f(x_min*1.0f, y_max*1.0f, z_max*1.0f);    // Top Right Of The Quad (Left)
        glVertex3f(x_min*1.0f, y_max*1.0f,z_min*1.0f);    // Top Left Of The Quad (Left)
        glVertex3f(x_min*1.0f,y_min*1.0f,z_min*1.0f);    // Bottom Left Of The Quad (Left)
        glVertex3f(x_min*1.0f,y_min*1.0f, z_max*1.0f);    // Bottom Right Of The Quad (Left)
//                      glEnd();

        glColor4f(1.0f,0.0f,1.0f,0.6f);    // Color Violet

//                      glBegin(GL_LINE_LOOP);
        glVertex3f( x_max*1.0f, y_max*1.0f,z_min*1.0f);    // Top Right Of The Quad (Right)
        glVertex3f( x_max*1.0f, y_max*1.0f, z_max*1.0f);    // Top Left Of The Quad (Right)
        glVertex3f( x_max*1.0f,y_min*1.0f, z_max*1.0f);    // Bottom Left Of The Quad (Right)
        glVertex3f( x_max*1.0f,y_min*1.0f,z_min*1.0f);    // Bottom Right Of The Quad (Right)
        glEnd();

        // End Drawing The Cube - See more at: http://www.codemiles.com/c-opengl-examples/draw-3d-cube-using-opengl-t9018.html#sthash.vKmu1Epd.dpuf

}

void draw_bbox_only_withpose(TooN::Vector<3>& size_,
                             TooN::Vector<3>& center_,
                             TooN::SE3<>& thispose)
{

    glPushMatrix();

    CVD::glMultMatrix(thispose);

    glColor3f(1,0,1);

//    glTranslatef(center_[0],center_[1],center_[2]);
    glScalef(size_[0],size_[1],size_[2]);


    glBegin(GL_LINES);

    //front
    glVertex3f(-0.5f, 0.5f, 0.5f);
    glVertex3f(0.5f, 0.5f, 0.5f);

    glVertex3f(0.5f, 0.5f, 0.5f);
    glVertex3f(0.5f, -0.5f, 0.5f);

    glVertex3f(0.5f, -0.5f, 0.5f);
    glVertex3f(-0.5f, -0.5f, 0.5f);

    glVertex3f(-0.5f, -0.5f, 0.5f);
    glVertex3f(-0.5f, 0.5f, 0.5f);

    //right
    glVertex3f(0.5f, 0.5f, 0.5f);
    glVertex3f(0.5f, 0.5f, -0.5f);

    glVertex3f(0.5f, 0.5f, -0.5f);
    glVertex3f(0.5f, -0.5f, -0.5f);

    glVertex3f(0.5f, -0.5f, -0.5f);
    glVertex3f(0.5f, -0.5f, 0.5f);

    //back
    glVertex3f(0.5f, 0.5f, -0.5f);
    glVertex3f(-0.5f, 0.5f, -0.5f);

    glVertex3f(-0.5f, -0.5f, -0.5f);
    glVertex3f(0.5f, -0.5f, -0.5f);

    glVertex3f(-0.5f, -0.5f, -0.5f);
    glVertex3f(-0.5f, 0.5f, -0.5f);

    //left
    glVertex3f(-0.5f, 0.5f, -0.5f);
    glVertex3f(-0.5f, 0.5f, 0.5f);

    glVertex3f(-0.5f, -0.5f, 0.5f);
    glVertex3f(-0.5f, -0.5f, -0.5f);
    glEnd();

    glPopMatrix();

    glClearColor(0,0,0,0);

}


void draw_bbox_2D_withpose(TooN::Vector<3>& size_,
                             TooN::Vector<3>& center_,
                             TooN::SE3<>& thispose)
{

    glPushMatrix();

    CVD::glMultMatrix(thispose);

//    glColor3f(1,0,1);

//    glTranslatef(center_[0],center_[1],center_[2]);
    glScalef(size_[0],size_[1],size_[2]);


    glBegin(GL_LINES);

    float scale = 0.0f;

    //front
    glVertex3f(-0.5f, scale*0.5f, 0.5f);
    glVertex3f(0.5f, scale*0.5f, 0.5f);

    glVertex3f(0.5f, scale*0.5f, 0.5f);
    glVertex3f(0.5f, scale*-0.5f, 0.5f);

    glVertex3f(0.5f, scale*-0.5f, 0.5f);
    glVertex3f(-0.5f,scale*-0.5f, 0.5f);

    glVertex3f(-0.5f, scale*-0.5f, 0.5f);
    glVertex3f(-0.5f, scale*0.5f, 0.5f);

    //right
    glVertex3f(0.5f, scale*0.5f, 0.5f);
    glVertex3f(0.5f, scale*0.5f, -0.5f);

    glVertex3f(0.5f, scale*0.5f, -0.5f);
    glVertex3f(0.5f, scale*-0.5f, -0.5f);

    glVertex3f(0.5f, scale*-0.5f, -0.5f);
    glVertex3f(0.5f, scale*-0.5f, 0.5f);

    //back
    glVertex3f(0.5f, scale*0.5f, -0.5f);
    glVertex3f(-0.5f,scale*0.5f, -0.5f);

    glVertex3f(-0.5f, scale*-0.5f, -0.5f);
    glVertex3f(0.5f, scale*-0.5f, -0.5f);

    glVertex3f(-0.5f, scale*-0.5f, -0.5f);
    glVertex3f(-0.5f, scale*0.5f, -0.5f);

    //left
    glVertex3f(-0.5f, scale*0.5f, -0.5f);
    glVertex3f(-0.5f, scale*0.5f, 0.5f);

    glVertex3f(-0.5f, scale*-0.5f, 0.5f);
    glVertex3f(-0.5f, scale*-0.5f, -0.5f);
    glEnd();

    glPopMatrix();

    glClearColor(0,0,0,0);

}

void draw_bbox_only(TooN::Vector<3>& size_,
                    TooN::Vector<3>& center_)
{

    glPushMatrix();

    glColor3f(1,0,1);

    glTranslatef(center_[0],center_[1],center_[2]);
    glScalef(size_[0],size_[1],size_[2]);


    glBegin(GL_LINES);

    //front
    glVertex3f(-0.5f, 0.5f, 0.5f);
    glVertex3f(0.5f, 0.5f, 0.5f);

    glVertex3f(0.5f, 0.5f, 0.5f);
    glVertex3f(0.5f, -0.5f, 0.5f);

    glVertex3f(0.5f, -0.5f, 0.5f);
    glVertex3f(-0.5f, -0.5f, 0.5f);

    glVertex3f(-0.5f, -0.5f, 0.5f);
    glVertex3f(-0.5f, 0.5f, 0.5f);

    //right
    glVertex3f(0.5f, 0.5f, 0.5f);
    glVertex3f(0.5f, 0.5f, -0.5f);

    glVertex3f(0.5f, 0.5f, -0.5f);
    glVertex3f(0.5f, -0.5f, -0.5f);

    glVertex3f(0.5f, -0.5f, -0.5f);
    glVertex3f(0.5f, -0.5f, 0.5f);

    //back
    glVertex3f(0.5f, 0.5f, -0.5f);
    glVertex3f(-0.5f, 0.5f, -0.5f);

    glVertex3f(-0.5f, -0.5f, -0.5f);
    glVertex3f(0.5f, -0.5f, -0.5f);

    glVertex3f(-0.5f, -0.5f, -0.5f);
    glVertex3f(-0.5f, 0.5f, -0.5f);

    //left
    glVertex3f(-0.5f, 0.5f, -0.5f);
    glVertex3f(-0.5f, 0.5f, 0.5f);

    glVertex3f(-0.5f, -0.5f, 0.5f);
    glVertex3f(-0.5f, -0.5f, -0.5f);
    glEnd();

    glPopMatrix();

    glClearColor(0,0,0,0);

}

void get_bbox(tinyobj::mesh_t* mesh,
              TooN::Vector<3>& size_,
              TooN::Vector<3>& center_)
{
    GLfloat
      min_x, max_x,
      min_y, max_y,
      min_z, max_z;
    min_x = max_x = mesh->positions[0];
    min_y = max_y = mesh->positions[1];
    min_z = max_z = mesh->positions[2];
    for (int i = 0; i < mesh->positions.size()/3; i++) {
      if (mesh->positions[3*i+0] < min_x) min_x = mesh->positions[3*i+0];
      if (mesh->positions[3*i+0] > max_x) max_x = mesh->positions[3*i+0];
      if (mesh->positions[3*i+1] < min_y) min_y = mesh->positions[3*i+1];
      if (mesh->positions[3*i+1] > max_y) max_y = mesh->positions[3*i+1];
      if (mesh->positions[3*i+2] < min_z) min_z = mesh->positions[3*i+2];
      if (mesh->positions[3*i+2] > max_z) max_z = mesh->positions[3*i+2];
    }
    glm::vec3 size = glm::vec3(max_x-min_x, max_y-min_y, max_z-min_z);
    glm::vec3 center = glm::vec3((min_x+max_x)/2, (min_y+max_y)/2, (min_z+max_z)/2);

    size_   = TooN::makeVector(size.x,size.y,size.z);
    center_ = TooN::makeVector(center.x,center.y,center.z);
}

void get_bbox_vecf(float* mesh,
                  int mesh_size,
              TooN::Vector<3>& size_,
              TooN::Vector<3>& center_)
{
    GLfloat
      min_x, max_x,
      min_y, max_y,
      min_z, max_z;

    min_x = max_x = mesh[0];
    min_y = max_y = mesh[1];
    min_z = max_z = mesh[2];

    for (int i = 0; i < mesh_size/3; i++)
    {

      if (mesh[3*i+0] < min_x) min_x = mesh[3*i+0];
      if (mesh[3*i+0] > max_x) max_x = mesh[3*i+0];

      if (mesh[3*i+1] < min_y) min_y = mesh[3*i+1];
      if (mesh[3*i+1] > max_y) max_y = mesh[3*i+1];

      if (mesh[3*i+2] < min_z) min_z = mesh[3*i+2];
      if (mesh[3*i+2] > max_z) max_z = mesh[3*i+2];
    }

    glm::vec3 size   = glm::vec3(max_x-min_x, max_y-min_y, max_z-min_z);
    glm::vec3 center = glm::vec3((min_x+max_x)/2, (min_y+max_y)/2, (min_z+max_z)/2);

    size_   = TooN::makeVector(size.x,size.y,size.z);
    center_ = TooN::makeVector(center.x,center.y,center.z);
}

void get_bbox_vec(std::vector<float> mesh,
              TooN::Vector<3>& size_,
              TooN::Vector<3>& center_)
{
    GLfloat
      min_x, max_x,
      min_y, max_y,
      min_z, max_z;
    min_x = max_x = mesh.at(0);
    min_y = max_y = mesh.at(1);
    min_z = max_z = mesh.at(2);

    for (int i = 0; i < mesh.size()/3; i++)
    {

      if (mesh.at(3*i+0) < min_x) min_x = mesh.at(3*i+0);
      if (mesh.at(3*i+0) > max_x) max_x = mesh.at(3*i+0);

      if (mesh.at(3*i+1) < min_y) min_y = mesh.at(3*i+1);
      if (mesh.at(3*i+1) > max_y) max_y = mesh.at(3*i+1);

      if (mesh.at(3*i+2) < min_z) min_z = mesh.at(3*i+2);
      if (mesh.at(3*i+2) > max_z) max_z = mesh.at(3*i+2);
    }

    glm::vec3 size = glm::vec3(max_x-min_x, max_y-min_y, max_z-min_z);
    glm::vec3 center = glm::vec3((min_x+max_x)/2, (min_y+max_y)/2, (min_z+max_z)/2);

    size_   = TooN::makeVector(size.x,size.y,size.z);
    center_ = TooN::makeVector(center.x,center.y,center.z);
}

void draw_bbox(tinyobj::mesh_t* mesh) {
  if (mesh->positions.size() == 0)
    return;

  // Cube 1x1x1, centered on origin
//  GLfloat vertices[] = {
//    -0.5, -0.5, -0.5, 1.0,
//     0.5, -0.5, -0.5, 1.0,
//     0.5,  0.5, -0.5, 1.0,
//    -0.5,  0.5, -0.5, 1.0,
//    -0.5, -0.5,  0.5, 1.0,
//     0.5, -0.5,  0.5, 1.0,
//     0.5,  0.5,  0.5, 1.0,
//    -0.5,  0.5,  0.5, 1.0,
//  };
//  GLuint vbo_vertices;
//  glGenBuffers(1, &vbo_vertices);
//  glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
//  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
//  glBindBuffer(GL_ARRAY_BUFFER, 0);

//  GLushort elements[] = {
//    0, 1, 2, 3,
//    4, 5, 6, 7,
//    0, 4, 1, 5, 2, 6, 3, 7
//  };
//  GLuint ibo_elements;
//  glGenBuffers(1, &ibo_elements);
//  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_elements);
//  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);
//  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


  GLfloat
    min_x, max_x,
    min_y, max_y,
    min_z, max_z;
  min_x = max_x = mesh->positions[0];
  min_y = max_y = mesh->positions[1];
  min_z = max_z = mesh->positions[2];
  for (int i = 0; i < mesh->positions.size()/3; i++) {
    if (mesh->positions[3*i+0] < min_x) min_x = mesh->positions[3*i+0];
    if (mesh->positions[3*i+0] > max_x) max_x = mesh->positions[3*i+0];
    if (mesh->positions[3*i+1] < min_y) min_y = mesh->positions[3*i+1];
    if (mesh->positions[3*i+1] > max_y) max_y = mesh->positions[3*i+1];
    if (mesh->positions[3*i+2] < min_z) min_z = mesh->positions[3*i+2];
    if (mesh->positions[3*i+2] > max_z) max_z = mesh->positions[3*i+2];
  }
  glm::vec3 size = glm::vec3(max_x-min_x, max_y-min_y, max_z-min_z);
  glm::vec3 center = glm::vec3((min_x+max_x)/2, (min_y+max_y)/2, (min_z+max_z)/2);
//  glm::mat4 transform =  glm::translate(glm::mat4(1), center)* glm::scale(glm::mat4(1), size);


//  std::cout<<"size = " << size.x <<", "<< size.y <<", "<< size.z << std::endl;


  /* Apply object's transformation matrix */
//  glm::mat4 m = mesh->object2world * transform;
//  glUniformMatrix4fv(uniform_m, 1, GL_FALSE, glm::value_ptr(m));


  glPushMatrix();

  glColor3f(1,0,1);

  glTranslatef(center.x,center.y,center.z);
  glScalef(size.x,size.y,size.z);
//  glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
//  glEnableVertexAttribArray(attribute_v_coord);
//  glVertexAttribPointer(
//    attribute_v_coord,  // attribute
//    4,                  // number of elements per vertex, here (x,y,z,w)
//    GL_FLOAT,           // the type of each element
//    GL_FALSE,           // take our values as-is
//    0,                  // no extra data between each position
//    0                   // offset of first element
//  );

  // White side - BACK

  glBegin(GL_LINES);

  //front
  glVertex3f(-0.5f, 0.5f, 0.5f);
  glVertex3f(0.5f, 0.5f, 0.5f);

  glVertex3f(0.5f, 0.5f, 0.5f);
  glVertex3f(0.5f, -0.5f, 0.5f);

  glVertex3f(0.5f, -0.5f, 0.5f);
  glVertex3f(-0.5f, -0.5f, 0.5f);

  glVertex3f(-0.5f, -0.5f, 0.5f);
  glVertex3f(-0.5f, 0.5f, 0.5f);

  //right
  glVertex3f(0.5f, 0.5f, 0.5f);
  glVertex3f(0.5f, 0.5f, -0.5f);

  glVertex3f(0.5f, 0.5f, -0.5f);
  glVertex3f(0.5f, -0.5f, -0.5f);

  glVertex3f(0.5f, -0.5f, -0.5f);
  glVertex3f(0.5f, -0.5f, 0.5f);

  //back
  glVertex3f(0.5f, 0.5f, -0.5f);
  glVertex3f(-0.5f, 0.5f, -0.5f);

  glVertex3f(-0.5f, -0.5f, -0.5f);
  glVertex3f(0.5f, -0.5f, -0.5f);

  glVertex3f(-0.5f, -0.5f, -0.5f);
  glVertex3f(-0.5f, 0.5f, -0.5f);

  //left
  glVertex3f(-0.5f, 0.5f, -0.5f);
  glVertex3f(-0.5f, 0.5f, 0.5f);

  glVertex3f(-0.5f, -0.5f, 0.5f);
  glVertex3f(-0.5f, -0.5f, -0.5f);
  glEnd();

//  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_elements);
//  glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_SHORT, 0);
//  glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_SHORT, (GLvoid*)(4*sizeof(GLushort)));
//  glDrawElements(GL_LINES, 8, GL_UNSIGNED_SHORT, (GLvoid*)(8*sizeof(GLushort)));
//  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

////  glDisableVertexAttribArray(attribute_v_coord);
//  glBindBuffer(GL_ARRAY_BUFFER, 0);

//  glDeleteBuffers(1, &vbo_vertices);
//  glDeleteBuffers(1, &ibo_elements);

  glPopMatrix();

  glClearColor(0,0,0,0);
}

void DrawCamera(TooN::SE3<> world_from_cam, float end_pt, float line_width , bool do_inverse)
 {
     glPushMatrix();

     Vector<6> rot_trans = world_from_cam.ln();

     static int i = 0;

     world_from_cam = TooN::SE3<>(rot_trans);

//        if ( do_inverse )
//        {
//            glMultMatrix(world_from_cam.inverse());
//            if ( i == 0 )
//            {
//                std::cout << " world_from_cam inverse = " << world_from_cam << std::endl;
//                i++;
//            }

//        }
//        else
     {
         glMultMatrix(world_from_cam);
     }


     if ( do_inverse)
     {
         glColor3f(1.0,0.0,1.0);
     }
     else
     {
         glColor3f(0.0,1.0,1.0);
     }

     glShadeModel(GL_SMOOTH);
     glTranslatef(0,0,0);
 /// Ref: http://www.alpcentauri.info/glutsolidsphere.htm

     glutSolidSphere(0.02f, 10.0f, 2.0f);

     glLineWidth(line_width);

     glColor3f(1.0,0.0,0.0);
     glBegin(GL_LINES);
       glVertex3f(0.0f, 0.0f, 0.0f);
       glVertex3f(end_pt, 0.0f, 0.0f);
     glEnd();

     glColor3f(0.0,1.0,0.0);
     glBegin(GL_LINES);
       glVertex3f(0.0f, 0.0f, 0.0f);
       glVertex3f(0.0f, end_pt, 0.0f);
     glEnd();

     glColor3f(0.0,0.0,1.0);
     glBegin(GL_LINES);
       glVertex3f(0.0f, 0.0f, 0.0f);
       glVertex3f(0.0f, 0.0f, end_pt);
     glEnd();

     glPopMatrix();

 }
