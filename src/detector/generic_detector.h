////////////////////////////////////////////////////////////////////////////
//    File:        generic_detector.h
//    Author:      Ken Chatfield
//    Description: Generic interface to object detector
////////////////////////////////////////////////////////////////////////////

#ifndef FEATPIPE_GENERIC_DETECTOR_H_
#define FEATPIPE_GENERIC_DETECTOR_H_

#include <vector>
#include "opencv2/opencv.hpp"

using std::vector;
using cv::Mat;
using cv::Rect;

namespace featpipe {

  struct Detection {
    float score = 0.0;
    Rect rect = Rect();

    Detection() { }

    Detection(float score, Rect bb)
      : score(score)
      , rect(bb) { }

    bool operator<(const Detection& detection) const {
      return score > detection.score;
    }
  };

  class GenericDetector {
  public:
    /* empty virtual destructor, to allow overriding by derived
       classes */
    virtual ~GenericDetector() { }
    virtual GenericDetector* clone() const = 0;
    virtual vector<vector<Detection> > detect(const vector<Mat>& images) = 0;

    virtual vector<Detection> detect(cv::Mat& image) {
      std::vector<Mat> images;
      images.push_back(image);

      return detect(images)[0];
    }
  };
}

#endif
