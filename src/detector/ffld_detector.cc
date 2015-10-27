#include "ffld_detector.h"

#include "util/mat_jpeg_image.h"

#include "HOGPyramid.h"

#include <iostream>
#include <fstream>
#include <utility>
#include <algorithm>
#include <glog/logging.h>

using FFLD::HOGPyramid;
using FFLD::Mixture;
using FFLD::Model;
using FFLD::Intersector;
using FFLD::Rectangle;
using FFLD::Patchwork;

using std::pair;

namespace featpipe {

vector<vector<Detection> > FFLDDetector::detect(const vector<Mat>& images) {

  vector<vector<Detection> > detections;

  int i;
  #pragma omp parallel for private(i)
  for (i = 0; i < images.size(); ++i) {
    const auto& image_mat_full = images[i];

    // ensure image is below max_im_size_
    float sf = 1.0;
    if (image_mat_full.cols > max_im_size_) {
      sf = static_cast<float>(max_im_size_) / static_cast<float>(image_mat_full.cols);
    }
    if (image_mat_full.rows > max_im_size_) {
      float sf2 = static_cast<float>(max_im_size_) / static_cast<float>(image_mat_full.rows);
      if (sf2 < sf) sf = sf2;
    }

    Mat image_mat;
    if (sf == 1.0) {
      image_mat = image_mat_full;
    } else {
      cv::resize(image_mat_full, image_mat, cv::Size(), sf, sf);
    }

    // wrap in JPEGImage compatible interface
    MatJPEGImage image(image_mat);

    // construct HOG pyramid for current image
    HOGPyramid pyramid(image, config_.padding, config_.padding, config_.interval);
    vector<Detection> im_detections;

    // compute the scores
    vector<HOGPyramid::Matrix> scores;
    vector<Mixture::Indices> argmaxes;
    vector<vector<vector<Model::Positions> > > positions;

    mixture_.convolve(pyramid, scores, argmaxes, &positions);

    // Cache the size of the models
    vector<pair<int, int> > sizes(mixture_.models().size());

    for (int i = 0; i < sizes.size(); ++i)
      sizes[i] = mixture_.models()[i].rootSize();

    int im_width = image.width();
    int im_height = image.height();

    // For each scale
    for (int z = 0; z < scores.size(); ++z) {
      const double scale = pow(2.0, static_cast<double>(z) / pyramid.interval() + 2);

      const int rows = static_cast<int>(scores[z].rows());
      const int cols = static_cast<int>(scores[z].cols());

      for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
          const double score = scores[z](y, x);

          if (score > config_.threshold) {
            // Non-maxima suppresion in a 3x3 neighborhood
            if (((y == 0) || (x == 0) || (score >= scores[z](y - 1, x - 1))) &&
                ((y == 0) || (score >= scores[z](y - 1, x))) &&
                ((y == 0) || (x == cols - 1) || (score >= scores[z](y - 1, x + 1))) &&
                ((x == 0) || (score >= scores[z](y, x - 1))) &&
                ((x == cols - 1) || (score >= scores[z](y, x + 1))) &&
                ((y == rows - 1) || (x == 0) || (score >= scores[z](y + 1, x - 1))) &&
                ((y == rows - 1) || (score >= scores[z](y + 1, x))) &&
                ((y == rows - 1) || (x == cols - 1) ||
                 (score >= scores[z](y + 1, x + 1)))) {
              // store truncated bb
              int x = std::max(static_cast<int>((x - pyramid.padx()) * scale + 0.5), 0);
              int y = std::max(static_cast<int>((y - pyramid.pady()) * scale + 0.5), 0);
              int width = std::min(static_cast<int>(sizes[argmaxes[z](y, x)].second * scale + 0.5),
                                   im_width - x);
              int height = std::min(static_cast<int>(sizes[argmaxes[z](y, x)].first * scale + 0.5),
                                    im_height - y);
              Rect bb(x, y, width, height);

              if (bb.area() > 0)
                im_detections.push_back(Detection(score, bb));
            }
          }
        }
      }
    }

    // Non maxima suppression
    sort(im_detections.begin(), im_detections.end());

    for (int i = 1; i < im_detections.size(); ++i) {
      Rectangle ref(im_detections[i-1].rect.x, im_detections[i-1].rect.y,
                    im_detections[i-1].rect.width, im_detections[i-1].rect.height);
      Intersector intersector(ref, config_.overlap, true);

      im_detections.resize(remove_if(im_detections.begin() + i, im_detections.end(),
                                     [intersector](const Detection& d){
                                       Rectangle r(d.rect.x, d.rect.y, d.rect.width, d.rect.height);
                                       return intersector(r);
                                     }) -
                           im_detections.begin());
    }

    detections.push_back(im_detections);

  }

  return detections;

}

void FFLDDetector::initFromConfig_() {

  // load model
  std::ifstream in(config_.model_file.c_str(), std::ios::binary);
  CHECK(in.is_open());

  in >> mixture_;
  CHECK(!mixture_.empty());

  // init patchwork class for fast convolutions
  LOG(INFO) << "Initializing patchwork class...";
  CHECK(Patchwork::InitFFTW((max_im_size_ + 15) & ~15, (max_im_size_ + 15) & ~15)) <<
    "Error initializing Patchwork class";

  // pre-cache transformed filters
  mixture_.cacheFilters();

}

}
