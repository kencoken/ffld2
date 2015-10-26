#include "ffld_detector.h"

#include <iostream>
#include <glog/logging.h>

vector<vector<Rect> > featpipe::FFLDDetector::detect(const vector<Mat>& images) {

  // construct HOG pyramid for current image
  // TODO: image CURRENTLY JPEGImage instance - update to accept opencv matrix
  HOGPyramid pyramid(image, config_.padding, config_.padding, config_.interval);
  vector<Detection> detections;

  // compute the scores
  vector<HOGPyramid::Matrix> scores;
  vector<Mixture::Indices> argmaxes;
  vector<vector<vector<Model::Positions> > > positions;

  mixture_.convolve(pyramid, scores, argmaxes, &positions);

  // Cache the size of the models
  vector<pair<int, int> > sizes(mixture.models().size());

  for (int i = 0; i < sizes.size(); ++i)
    sizes[i] = mixture.models()[i].rootSize();

  // For each scale
  for (int z = 0; z < scores.size(); ++z) {
    const double scale = pow(2.0, static_cast<double>(z) / pyramid.interval() + 2);

    const int rows = static_cast<int>(scores[z].rows());
    const int cols = static_cast<int>(scores[z].cols());

    for (int y = 0; y < rows; ++y) {
      for (int x = 0; x < cols; ++x) {
        const double score = scores[z](y, x);

        if (score > threshold) {
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
            Rectangle bndbox((x - pyramid.padx()) * scale + 0.5,
                     (y - pyramid.pady()) * scale + 0.5,
                     sizes[argmaxes[z](y, x)].second * scale + 0.5,
                     sizes[argmaxes[z](y, x)].first * scale + 0.5);

            // Truncate the object
            bndbox.setX(max(bndbox.x(), 0));
            bndbox.setY(max(bndbox.y(), 0));
            bndbox.setWidth(min(bndbox.width(), width - bndbox.x()));
            bndbox.setHeight(min(bndbox.height(), height - bndbox.y()));

            if (!bndbox.empty())
              detections.push_back(Detection(score, x, y, z, bndbox));
          }
        }
      }
    }
  }

  // Non maxima suppression
  sort(detections.begin(), detections.end());

  for (int i = 1; i < detections.size(); ++i)
    detections.resize(remove_if(detections.begin() + i, detections.end(),
                                Intersector(detections[i - 1], overlap, true)) -
                      detections.begin());


}

void featpipe::FFLDDetector::initFromConfig_() {

  // load model
  ifstream in(config_.model_file.c_str(), ios::binary);
  CHECK(in.is_open());

  in >> mixture_;
  CHECK(!mixture_.empty());

  // init patchwork class for fast convolutions
  LOG(INFO) << "Initializing patchwork class...";
  CHECK(Patchwork::InitFFTW((max_im_size + 15) & ~15, (max_im_size + 15) & ~15)) <<
    "Error initializing Patchwork class";

  // pre-cache transformed filters
  mixture_.cacheFilters();

}
