#ifndef _CONFIGS_H_
#define _CONFIGS_H_

#include <string>
namespace Tn
{
    const int INPUT_CHANNEL = 3;
    const std::string INPUT_ONNXMODEL = "ped.onnx";
    const std::string ENGINE_FILE = "";

    const std::string INPUT_IMAGE = "./1.jpg";
    const std::string EVAL_LIST = "";
    const int INPUT_WIDTH = 608;
    const int INPUT_HEIGHT = 608;

    const int DETECT_CLASSES = 1;
    const float NMS_THRESH = 0.45;
}

#endif
