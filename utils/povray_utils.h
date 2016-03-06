#include <GL/glew.h>

#include <GL/freeglut.h>
#include <GL/glut.h>

//#include <fstream>

#include<TooN/TooN.h>
#include<TooN/se3.h>

#include <cvd/gl_helpers.h>

//#include <vector_types.h>
#include <cvd/image_io.h>



using namespace TooN;
using namespace CVD;

using namespace std;

namespace povray_utils{

    void DrawCamera(TooN::SE3<> world_from_cam, float end_pt, float line_width , bool do_inverse)
    {
        glPushMatrix();

        Vector<6> rot_trans = world_from_cam.ln();

        static int i = 0;

        world_from_cam = TooN::SE3<>(rot_trans);

        {
                //glMultMatrix(world_from_cam);
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

};
