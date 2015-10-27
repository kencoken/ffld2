////////////////////////////////////////////////////////////////////////////
//    File:        ffld_detector.h
//    Author:      Ken Chatfield
//    Description: Detector using pretrained DPM (ffld2 library)
////////////////////////////////////////////////////////////////////////////

#ifndef FEATPIPE_FFLD_DETECTOR_H_
#define FEATPIPE_FFLD_DETECTOR_H_

#include "generic_detector.h"
#include "ffld_config.h"

// FFLD headers
#include "Intersector.h"
#include "Mixture.h"
#include "Scene.h"

//#include "cpuvisor_config.pb.h"

namespace featpipe {

  // class definition --------------------

  class FFLDDetector : public GenericDetector {
  public:
    inline virtual FFLDDetector* clone() const {
      return new FFLDDetector(*this);
    }
    // constructors
    FFLDDetector(const FFLDConfig& config): config_(config) {
      initFromConfig_();
    }
    // TODO: Add constructors from protobuf config
    /*CaffeEncoder(const cpuvisor::CaffeConfig& proto_config) {
      config_ = CaffeConfig();
      config_.configureFromProtobuf(proto_config);
      initNetFromConfig_();
    }
    CaffeEncoder(const CaffeEncoder& other) {
      config_ = other.config_;
      initNetFromConfig_();
    }
    CaffeEncoder& operator= (const CaffeEncoder& rhs) {
      config_ = rhs.config_;
      initNetFromConfig_();
      return (*this);
      }*/
    // main functions
    virtual vector<vector<Detection> > detect(const vector<Mat>& images);
  protected:
    void initFromConfig_();
    FFLDConfig config_;

    FFLD::Mixture mixture_;

    const int max_im_size_ = 1000;
  };

}

#endif
