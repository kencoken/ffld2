//--------------------------------------------------------------------------------------------------
// Implementation of the papers "Exact Acceleration of Linear Object Detectors", 12th European
// Conference on Computer Vision, 2012 and "Deformable Part Models with Individual Part Scaling",
// 24th British Machine Vision Conference, 2013.
//
// Copyright (c) 2013 Idiap Research Institute, <http://www.idiap.ch/>
// Written by Charles Dubout <charles.dubout@idiap.ch>
//
// This file is part of FFLDv2 (the Fast Fourier Linear Detector version 2)
//
// FFLDv2 is free software: you can redistribute it and/or modify it under the terms of the GNU
// Affero General Public License version 3 as published by the Free Software Foundation.
//
// FFLDv2 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero
// General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with FFLDv2. If
// not, see <http://www.gnu.org/licenses/>.
//--------------------------------------------------------------------------------------------------

#include "SimpleOpt.h"

#include "Mixture.h"

#include <algorithm>
#include <fstream>
#include <iostream>

#include <iomanip>
#include <ctime>

#include <vector>
#include <string>

using namespace FFLD;
using namespace std;

// SimpleOpt array of valid options
enum
{
  OPT_C, OPT_DATAMINE, OPT_INTERVAL, OPT_HELP, OPT_J, OPT_RELABEL, OPT_MODEL, OPT_NAME,
  OPT_PADDING, OPT_RESULT, OPT_SEED, OPT_OVERLAP, OPT_NB_COMP, OPT_NB_NEG,
  OPT_VOC_BASE_DIR, OPT_AFLW_BASE_DIR, OPT_AFLW_GT_FILE
};

CSimpleOpt::SOption SOptions[] =
{
  { OPT_C, "-c", SO_REQ_SEP },
  { OPT_C, "--C", SO_REQ_SEP },
  { OPT_DATAMINE, "-d", SO_REQ_SEP },
  { OPT_DATAMINE, "--datamine", SO_REQ_SEP },
  { OPT_INTERVAL, "-e", SO_REQ_SEP },
  { OPT_INTERVAL, "--interval", SO_REQ_SEP },
  { OPT_HELP, "-h", SO_NONE },
  { OPT_HELP, "--help", SO_NONE },
  { OPT_J, "-j", SO_REQ_SEP },
  { OPT_J, "--J", SO_REQ_SEP },
  { OPT_RELABEL, "-l", SO_REQ_SEP },
  { OPT_RELABEL, "--relabel", SO_REQ_SEP },
  { OPT_MODEL, "-m", SO_REQ_SEP },
  { OPT_MODEL, "--model", SO_REQ_SEP },
  { OPT_NAME, "-n", SO_REQ_SEP },
  { OPT_NAME, "--name", SO_REQ_SEP },
  { OPT_PADDING, "-p", SO_REQ_SEP },
  { OPT_PADDING, "--padding", SO_REQ_SEP },
  { OPT_RESULT, "-r", SO_REQ_SEP },
  { OPT_RESULT, "--result", SO_REQ_SEP },
  { OPT_SEED, "-s", SO_REQ_SEP },
  { OPT_SEED, "--seed", SO_REQ_SEP },
  { OPT_OVERLAP, "-v", SO_REQ_SEP },
  { OPT_OVERLAP, "--overlap", SO_REQ_SEP },
  { OPT_NB_COMP, "-x", SO_REQ_SEP },
  { OPT_NB_COMP, "--nb-components", SO_REQ_SEP },
  { OPT_NB_NEG, "-z", SO_REQ_SEP },
  { OPT_NB_NEG, "--nb-negatives", SO_REQ_SEP },
  // dataset dirs
  { OPT_VOC_BASE_DIR, "--voc-dir", SO_REQ_SEP },
  { OPT_AFLW_BASE_DIR, "--aflw-dir", SO_REQ_SEP },
  { OPT_AFLW_GT_FILE, "--aflw-gt", SO_REQ_SEP },
  SO_END_OF_OPTIONS
};

void showUsage()
{
  cout << "Usage: train [options] image_set.txt\n\n"
      "Options:\n"
      "  -c,--C <arg>             SVM regularization constant (default 0.002)\n"
      "  -d,--datamine <arg>      Maximum number of data-mining iterations within each "
      "training iteration  (default 10)\n"
      "  -e,--interval <arg>      Number of levels per octave in the HOG pyramid (default 5)"
      "\n"
      "  -h,--help                Display this information\n"
      "  -j,--J <arg>             SVM positive regularization constant boost (default 2)\n"
      "  -l,--relabel <arg>       Maximum number of training iterations (default 8, half if "
      "no part)\n"
      "  -m,--model <file>        Read the initial model from <file> (default zero model)\n"
      "  -n,--name <arg>          Name of the object to detect (default \"person\")\n"
      "  -p,--padding <arg>       Amount of zero padding in HOG cells (default 6)\n"
      "  -r,--result <file>       Write the trained model to <file> (default \"model.txt\")\n"
      "  -s,--seed <arg>          Random seed (default time(NULL))\n"
      "  -v,--overlap <arg>       Minimum overlap in latent positive search (default 0.7)\n"
      "  -x,--nb-components <arg> Number of mixture components (without symmetry, default 3)\n"
      "  -z,--nb-negatives <arg>  Maximum number of negative images to consider (default all)"
      "  --voc-dir <arg>          Base directory of VOC2007"
      "  --aflw-dir <arg>         Base directory of AFLW"
      "  --aflw-gt <arg>          Ground truth file for AFLW"
     << endl;
}

struct Opts {
  double C = 0.002;
  int nbDatamine = 10;
  int interval = 5;
  double J = 2.0;
  int nbRelabel = 8;
  string model;
  Object::Name name = Object::FACE;
  int padding = 6;
  string result = "model.txt";
  int seed = static_cast<int>(time(0));
  double overlap = 0.7;
  int nbComponents = 3;
  int nbNegativeScenes = -1;

  string voc_base_dir = "/data/ken/datasets/VOC2007/";
  string aflw_base_dir = "/data/ken/datasets/aflw/aflw/data/flickr/";
  string aflw_gt_file = "/home/ken/aflw-extract/detections.txt";

  vector<string> files;
};

int parseArgs(int argc, char* argv[], Opts& opts) {
  // Parse the parameters
  CSimpleOpt args(argc, argv, SOptions);

  while (args.Next()) {
    if (args.LastError() == SO_SUCCESS) {
      if (args.OptionId() == OPT_C) {
        opts.C = atof(args.OptionArg());

        if (opts.C <= 0) {
          showUsage();
          cerr << "\nInvalid C arg " << args.OptionArg() << endl;
          return -1;
        }
      }
      else if (args.OptionId() == OPT_DATAMINE) {
        opts.nbDatamine = atoi(args.OptionArg());

        if (opts.nbDatamine <= 0) {
          showUsage();
          cerr << "\nInvalid datamine arg " << args.OptionArg() << endl;
          return -1;
        }
      }
      else if (args.OptionId() == OPT_INTERVAL) {
        opts.interval = atoi(args.OptionArg());

        if (opts.interval <= 0) {
          showUsage();
          cerr << "\nInvalid interval arg " << args.OptionArg() << endl;
          return -1;
        }
      }
      else if (args.OptionId() == OPT_HELP) {
        showUsage();
        return 0;
      }
      else if (args.OptionId() == OPT_J) {
        opts.J = atof(args.OptionArg());

        if (opts.J <= 0) {
          showUsage();
          cerr << "\nInvalid J arg " << args.OptionArg() << endl;
          return -1;
        }
      }
      else if (args.OptionId() == OPT_RELABEL) {
        opts.nbRelabel = atoi(args.OptionArg());

        if (opts.nbRelabel <= 0) {
          showUsage();
          cerr << "\nInvalid relabel arg " << args.OptionArg() << endl;
          return -1;
        }
      }
      else if (args.OptionId() == OPT_MODEL) {
        opts.model = args.OptionArg();
      }
      else if (args.OptionId() == OPT_NAME) {
        string arg = args.OptionArg();
        transform(arg.begin(), arg.end(), arg.begin(), static_cast<int (*)(int)>(tolower));

        const string Names[22] =
        {
          "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
          "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
          "sheep", "sofa", "train", "tvmonitor", "face", "nonface"
        };

        const string * iter = find(Names, Names + 22, arg);

        if (iter == Names + 22) {
          showUsage();
          cerr << "\nInvalid name arg " << args.OptionArg() << endl;
          return -1;
        }

        opts.name = static_cast<Object::Name>(iter - Names);
      }
      else if (args.OptionId() == OPT_PADDING) {
        opts.padding = atoi(args.OptionArg());

        if (opts.padding <= 1) {
          showUsage();
          cerr << "\nInvalid padding arg " << args.OptionArg() << endl;
          return -1;
        }
      }
      else if (args.OptionId() == OPT_RESULT) {
        opts.result = args.OptionArg();
      }
      else if (args.OptionId() == OPT_SEED) {
        opts.seed = atoi(args.OptionArg());
      }
      else if (args.OptionId() == OPT_OVERLAP) {
        opts.overlap = atof(args.OptionArg());

        if ((opts.overlap <= 0.0) || (opts.overlap >= 1.0)) {
          showUsage();
          cerr << "\nInvalid overlap arg " << args.OptionArg() << endl;
          return -1;
        }
      }
      else if (args.OptionId() == OPT_NB_COMP) {
        opts.nbComponents = atoi(args.OptionArg());

        if (opts.nbComponents <= 0) {
          showUsage();
          cerr << "\nInvalid nb-components arg " << args.OptionArg() << endl;
          return -1;
        }
      }
      else if (args.OptionId() == OPT_NB_NEG) {
        opts.nbNegativeScenes = atoi(args.OptionArg());

        if (opts.nbNegativeScenes < 0) {
          showUsage();
          cerr << "\nInvalid nb-negatives arg " << args.OptionArg() << endl;
          return -1;
        }
      }
      else if (args.OptionId() == OPT_VOC_BASE_DIR) {
        opts.voc_base_dir = args.OptionArg();
        if (opts.voc_base_dir[opts.voc_base_dir.length()-1] != '/') {
          opts.voc_base_dir += "/";
        }
      }
      else if (args.OptionId() == OPT_AFLW_BASE_DIR) {
        opts.aflw_base_dir = args.OptionArg();
        if (opts.aflw_base_dir[opts.aflw_base_dir.length()-1] != '/') {
          opts.aflw_base_dir += "/";
        }
      }
      else if (args.OptionId() == OPT_AFLW_GT_FILE) {
        opts.aflw_gt_file = args.OptionArg();
      }
    }
    else {
      showUsage();
      cerr << "\nUnknown option " << args.OptionText() << endl;
      return -1;
    }
  }

  // if (!args.FileCount()) {
  //   showUsage();
  //   cerr << "\nNo dataset provided" << endl;
  //   return -1;
  // }
  // else if (args.FileCount() > 1) {
  //   showUsage();
  //   cerr << "\nMore than one dataset provided" << endl;
  //   return -1;
  // }

  for (int i = 0; i < args.FileCount(); ++i) {
    opts.files.push_back(args.File(i));
  }

  return 0;
}

// Train a mixture model
int main(int argc, char * argv[])
{

  cout << "Starting..." << endl;

  // parse input arguments
  Opts opts;
  int retval = parseArgs(argc, argv, opts);
  if (retval != 0) {
    return retval;
  }

  cout << "Aruments parsed" << endl;

  srand(opts.seed);
  srand48(opts.seed);

  // // Open the image set file
  // const string file = opts.files[0];
  // const size_t lastDot = file.find_last_of('.');

  // if ((lastDot == string::npos) || (file.substr(lastDot) != ".txt")) {
  //   showUsage();
  //   cerr << "\nInvalid image set file " << file << ", should be .txt" << endl;
  //   return -1;
  // }

  // ifstream in(file.c_str());

  // if (!in.is_open()) {
  //   showUsage();
  //   cerr << "\nInvalid image set file " << file << endl;
  //   return -1;
  // }

  // Find the annotations folder
  const string voc_anno_dir = opts.voc_base_dir + "Annotations/";
  const string voc_people_anno_file = opts.voc_base_dir + "ImageSets/Main/person_trainval.txt";

  // Load all the scenes
  int maxRows = 0;
  int maxCols = 0;

  vector<Scene> scenes;

  // add positive scenes (from AFLW)

  cout << "Parsing AFLW ground truth from: " << opts.aflw_gt_file << endl;
  ifstream aflw_in(opts.aflw_gt_file.c_str());

  int min_bb_width = 50;
  int max_bb_width = 300;

  int num_pos = 0;
  int num_neg = 0;

  while (aflw_in) {
    string line;
    getline(aflw_in, line);

    // Skip empty lines
    if (line.size() < 3)
      continue;

    num_pos++;

    // parse line
    size_t next_pos = line.find(" ");

    string fname = line.substr(0, next_pos);
    vector<int> bb; // x1, y1, x2, y2, width, height

    for (int i = 0; i < 6; ++i) {
      size_t pos = next_pos;
      next_pos = line.find(" ", pos+1);
      if (next_pos == string::npos) {
        next_pos = line.length();
      }
      bb.push_back(stoi(line.substr(pos+1, pos+1+next_pos)));
    }

    // // keep track of min/max width (aspect ratio always 1)

    // int bb_width = bb[3] - bb[1];
    // if (bb_width < min_bb_width) {
    //   min_bb_width = bb_width;
    // } else if (bb_width > max_bb_width) {
    //   max_bb_width = bb_width;
    // }

    //cout << fname << "-" << bb[0] << "," << bb[1] << "," << bb[2] <<
    //"," << bb[3] << endl;

    vector<Object> objects;
    objects.push_back(Object(Object::FACE, Object::FRONTAL, false, false, Rectangle(bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1])));

    const int width = bb[4];
    const int height = bb[5];
    const int depth = 3;
    const string filename = opts.aflw_base_dir + fname;

    Scene scene(width, height, depth, filename, objects);

    maxRows = max(maxRows, (scene.height() + 3) / 4 + opts.padding);
    maxCols = max(maxCols, (scene.width() + 3) / 4 + opts.padding);
    
    scenes.push_back(scene);

  }

  // add negative scenes (from VOC)

  cout << "Parsing VOC ground truth from: " << voc_people_anno_file << endl;
  ifstream voc_in(voc_people_anno_file.c_str());

  while (voc_in) {
    int N_rand = 2;

    string line;
    getline(voc_in, line);

    // Skip empty lines
    if (line.size() < 3)
      continue;

    // parse line
    size_t split_start = line.find(" ");
    size_t split_end = split_start;
    while (line[split_end+1] == ' ') split_end++;

    string id = line.substr(0, split_start);
    string label = line.substr(split_end+1);

    // Skip if not negative
    if (label != "-1")
      continue;

    // get width/height from annotation file
    Scene scene_init(voc_anno_dir + id + ".xml");
    const int width = scene_init.width();
    const int height = scene_init.height();
    const int depth = scene_init.depth();
    const string filename = scene_init.filename();

    // Sample N_rand random bounding boxes from image

    vector<Object> objects;

    for (int i = 0; i < N_rand; ++i) {

      if (opts.nbNegativeScenes > 0) {
        num_neg++;
      }

      int max_bb_width_corr = (max_bb_width > width) ? width : max_bb_width;
      if (max_bb_width_corr > height) {
        max_bb_width_corr = height;
      }

      int bb_width = rand() % (max_bb_width_corr - min_bb_width) + min_bb_width;
      int bb_height = bb_width;

      int x = rand() % (width - bb_width);
      int y = rand() % (height - bb_height);

      //cout << id << ":" << label << " " << width << "," << height
      //     << " BB:" << x << "," << y << "," << bb_width << "," <<
      //     bb_height << endl;

      objects.push_back(Object(Object::NONFACE, Object::FRONTAL, false, false, Rectangle(x, y, bb_width, bb_height)));

    }

    if (opts.nbNegativeScenes > 0) {
      Scene scene(width, height, depth, filename, objects);

      maxRows = max(maxRows, (scene.height() + 3) / 4 + opts.padding);
      maxCols = max(maxCols, (scene.width() + 3) / 4 + opts.padding);

      scenes.push_back(scene);

      --opts.nbNegativeScenes;
    }

  }

  cout << num_pos << " positives, " << num_neg << " negatives" << endl;
  cout << "Total scenes: " << scenes.size() << endl;
  cout << "maxRows: " << maxRows << ", maxCols: " << maxCols << endl;

  if (scenes.empty()) {
    showUsage();
    cerr << "\nCould not read in scenes!" << endl;
    return -1;
  }

  {  // START TRAINING

    // Initialize the Patchwork class
    cout << "Initializing patchwork class..." << endl;
    if (!Patchwork::InitFFTW((maxRows + 15) & ~15, (maxCols + 15) & ~15)) {
      cerr << "Error initializing the FFTW library" << endl;
      return - 1;
    }

    // The mixture to train
    cout << "Initializing mixture..." << endl;
    Mixture mixture(opts.nbComponents, scenes, opts.name);

    if (mixture.empty()) {
      cerr << "Error initializing the mixture model" << endl;
      return -1;
    }

    // Try to open the mixture
    if (!opts.model.empty()) {
      ifstream in(opts.model.c_str(), ios::binary);

      if (!in.is_open()) {
        showUsage();
        cerr << "\nInvalid model file " << opts.model << endl;
        return -1;
      }

      in >> mixture;

      if (mixture.empty()) {
        showUsage();
        cerr << "\nInvalid model file " << opts.model << endl;
        return -1;
      }
    }

    if (opts.model.empty()) {
      auto t = time(nullptr);
      auto tm = *localtime(&t);
      cout << put_time(&tm, "%d-%m-%Y %H-%M-%S") << " Starting training without parts..." << endl;
      mixture.train(scenes, opts.name, opts.padding, opts.padding, opts.interval,
                    opts.nbRelabel / 2, opts.nbDatamine, 24000, opts.C, opts.J,
                    opts.overlap);
    }

    if (mixture.models()[0].parts().size() == 1) {
      auto t = time(nullptr);
      auto tm = *localtime(&t);
      cout << put_time(&tm, "%d-%m-%Y %H-%M-%S") << " Initializing parts..." << endl;
      mixture.initializeParts(8, make_pair(6, 6));
    }

    {
      auto t = time(nullptr);
      auto tm = *localtime(&t);
      cout << put_time(&tm, "%d-%m-%Y %H-%M-%S") << " Starting training with parts..." << endl;
      mixture.train(scenes, opts.name, opts.padding, opts.padding, opts.interval,
                    opts.nbRelabel, opts.nbDatamine, 24000, opts.C, opts.J,
                    opts.overlap);
    }

    // Try to open the result file
    cout << "Saving results..." << endl;
    ofstream out(opts.result.c_str(), ios::binary);

    if (!out.is_open()) {
      showUsage();
      cerr << "\nInvalid result file " << opts.result << endl;
      cout << mixture << endl; // Print the mixture as a last resort
      return -1;
    }

    out << mixture;

  }

  return EXIT_SUCCESS;
}
