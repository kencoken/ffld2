#include "mat_jpeg_image.h"

#include <glog/logging.h>

namespace featpipe {

  MatJPEGImage::MatJPEGImage() {}

  MatJPEGImage::MatJPEGImage(Mat image)
    :image_(image) {
    CHECK_EQ(image_.type(), CV_8UC3);
  }

  bool MatJPEGImage::empty() const {
    return image_.empty();
  }

  int MatJPEGImage::width() const {
    return image_.cols;
  }

  int MatJPEGImage::height() const {
    return image_.rows;
  }

  int MatJPEGImage::depth() const {
    return image_.channels();
  }

  const uint8_t* MatJPEGImage::bits() const {
    return static_cast<uint8_t*>(image_.data);
  }

  uint8_t* MatJPEGImage::bits() {
    return static_cast<uint8_t*>(image_.data);
  }

  const uint8_t* MatJPEGImage::scanLine(int y) const {
    return image_.ptr(y);
  }

  uint8_t* MatJPEGImage::scanLine(int y) {
    return image_.ptr(y);
  }

  void MatJPEGImage::save(const string& filename, int quality) const {
    vector<int> params;
    params.push_back(CV_IMWRITE_JPEG_QUALITY);
    params.push_back(quality);

    cv::imwrite(filename, image_, params);
  }

  MatJPEGImage* MatJPEGImage::create_rescale(double scale) const {
    MatJPEGImage scaled = rescale(scale);
    return new MatJPEGImage(scaled);
  }

  MatJPEGImage MatJPEGImage::rescale(double scale) const {
    if (scale <= 0.0) return MatJPEGImage();

    if (scale == 1.0) return *this;

    Mat new_image;
    cv::resize(image_, new_image, cv::Size(), scale, scale);
    return MatJPEGImage(new_image);
  }

}
