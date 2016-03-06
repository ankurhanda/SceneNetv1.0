/* Implementation of Perlin Noise taken from
 * http://www.dreamincode.net/forums/topic/66480-perlin-noise/
 */

#ifndef _PERLIN_NOISE_H_
#define _PERLIN_NOISE_H_
#include "noise.h"
#include "noiseutils.h"

#include <time.h>
#include <libnoise/noise.h>
using namespace noise;

namespace render_kinect
{
  class PerlinNoise : public Noise
  {
  public:
  PerlinNoise( int width, int height, float scale)
    : Noise(width , height), 
      scale_(scale)
      {
              assert(scale>0.0 && scale_<1.0 );
              //srand (time(NULL));
              myModule_.SetPersistence(0.7);
              heightMapBuilder_.SetSourceModule (myModule_);
              heightMapBuilder_.SetDestNoiseMap (heightMap_);
              heightMapBuilder_.SetDestSize (width_, height_);
              heightMapBuilder_.SetBounds (2.0, 6.0, -1, 1);
      };
    
    void generateNoiseField( cv::Mat &noise_field)
    {
      int i = rand();
      myModule_.SetSeed(i);
      heightMapBuilder_.Build ();
    
      noise_field = cv::Mat(height_,width_,CV_32FC1);
      for(int r=0; r<height_; ++r) {
	float* noise_i = noise_field.ptr<float>(r);
	for(int c=0; c<width_; ++c) {
	  noise_i[c] = scale_ * heightMap_.GetValue(c,r);
	}
      }
    }
    
  private:
    float scale_;
    module::Perlin myModule_;
    utils::NoiseMap heightMap_;
    utils::NoiseMapBuilderPlane heightMapBuilder_;
    
  };
}

#endif // _NOISE_H_

