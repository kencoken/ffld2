////////////////////////////////////////////////////////////////////////////
//    File:        ffld_config.h
//    Author:      Ken Chatfield
//    Description: Config for FFLD detector
////////////////////////////////////////////////////////////////////////////

#ifndef FEATPIPE_FFLD_CONFIG_H_
#define FEATPIPE_FFLD_CONFIG_H_

#include <string>

using std::string;

namespace featpipe {

  // configuration -----------------------

  struct FFLDConfig {
    string model_file;
    float threshold = 0.3; // min detection threshold
    float overlap = 0.2; // min overlap in non-maximal suppression

    int padding = 6; // amount of zero padding in HOG cells
    int interval = 5; // number of levels per octave in HOG pyramid

    /*inline virtual void configureFromProtobuf(const cpuvisor::CaffeConfig& proto_config) {
      param_file = proto_config.param_file();
      model_file = proto_config.model_file();
      mean_image_file = proto_config.mean_image_file();
      mean_vals = std::vector<float>(proto_config.mean_image_vals_size());
      for (size_t i = 0; i < mean_vals.size(); ++i) {
        mean_vals[i] = proto_config.mean_image_vals(i);
      }
      cpuvisor::DataAugType proto_data_aug_type = proto_config.data_aug_type();
      switch (proto_data_aug_type) {
      case cpuvisor::DAT_NONE:
        data_aug_type = DAT_NONE;
        break;
      case cpuvisor::DAT_ASPECT_CORNERS:
        data_aug_type = DAT_ASPECT_CORNERS;
        break;
      }
      output_blob_name = proto_config.output_blob_name();
      cpuvisor::CaffeMode proto_caffe_mode = proto_config.mode();
      switch (proto_caffe_mode) {
      case cpuvisor::CM_CPU:
        mode = CM_CPU;
        break;
      case cpuvisor::CM_GPU:
        mode = CM_GPU;
        break;
      }
      use_rgb_images = proto_config.use_rgb_images();
      netpool_sz = proto_config.netpool_sz();
      image_mul = proto_config.image_mul();
    }*/
  };

}

#endif
