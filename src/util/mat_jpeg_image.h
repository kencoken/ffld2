////////////////////////////////////////////////////////////////////////////
//    File:        mat_jpeg_image.h
//    Author:      Ken Chatfield
//    Description: Wraps OpenCV Mat to share interface with
//    ffld2/JPEGImage class
////////////////////////////////////////////////////////////////////////////

#ifndef FEATPIPE_MAT_JPEG_IMAGE_H_
#define FEATPIPE_MAT_JPEG_IMAGE_H_

#include <vector>
#include <string>
#include "opencv2/opencv.hpp"

using std::vector;
using std::string;
using cv::Mat;

#include "JPEGImage.h"

namespace featpipe {

  class MatJPEGImage : public FFLD::JPEGImage {
  public:
    MatJPEGImage();
    MatJPEGImage(Mat image);

    bool empty() const;
    int width() const;
    int height() const;
    int depth() const;

    const uint8_t* bits() const;
    uint8_t* bits();
    const uint8_t* scanLine(int y) const;
    uint8_t* scanLine(int y);

    void save(const string& filename, int quality = 100) const;
    MatJPEGImage* create_rescale(double scale) const;
    MatJPEGImage rescale(double scale) const;

  protected:
    Mat image_;
  };

}

#endif
