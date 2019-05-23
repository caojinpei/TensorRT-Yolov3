#include <opencv2/opencv.hpp>
#include "TrtNet.h"
#include "argsParser.h"
#include "configs.h"
#include <chrono>
#include "YoloLayer.h"
#include "dataReader.h"
#include "eval.h"

using namespace std;
using namespace argsParser;
using namespace Tn;
using namespace Yolo;


string onnxFile = "./ped3_608_1.onnx";
string engineFile = "./ped3_608_1.trt";
string fileList = "./list.txt";

vector<string> labels = { "people"};
vector<vector<int> > output_shape = { {1, 18, 19, 19}, {1, 18, 38, 38} };
vector<vector<int> > g_masks = { {3, 4, 5}, {0, 1, 2} };
vector<vector<int> > g_anchors = { {8, 34}, {14, 60}, {23, 94}, {39, 149}, {87,291}, {187,472} };
float obj_threshold = 0.10;
float nms_threshold = 0.45;

int CATEGORY = 1;
int BATCH_SIZE = 1;
//int INPUT_CHANNEL = 3;
int DETECT_WIDTH = 608;
int DETECT_HEIGHT = 608;


// Res struct & function
typedef struct DetectionRes {
	float x, y, w, h, prob;
} DetectionRes;

float sigmoid(float in) {
	return 1.f / (1.f + exp(-in));
}
float exponential(float in) {
	return exp(in);
}


vector<float> prepareImage(cv::Mat& img)
{
    using namespace cv;

    int c = 3;
    int h = 608;   //net h
    int w = 608;   //net w

    float scale = min(float(w)/img.cols,float(h)/img.rows);
    auto scaleSize = cv::Size(img.cols * scale,img.rows * scale);

    cv::Mat rgb ;
    cv::cvtColor(img, rgb, CV_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized,scaleSize,0,0,INTER_CUBIC);

    cv::Mat cropped(h, w,CV_8UC3, 127);
    Rect rect((w- scaleSize.width)/2, (h-scaleSize.height)/2, scaleSize.width,scaleSize.height); 
    resized.copyTo(cropped(rect));

    cv::Mat img_float;
    if (c == 3)
        cropped.convertTo(img_float, CV_32FC3, 1/255.0);
    else
        cropped.convertTo(img_float, CV_32FC1 ,1/255.0);

    //HWC TO CHW
    vector<Mat> input_channels(c);
    cv::split(img_float, input_channels);

    vector<float> result(h*w*c);
    auto data = result.data();
    int channelLength = h * w;
    for (int i = 0; i < c; ++i) {
        memcpy(data,input_channels[i].data,channelLength*sizeof(float));
        data += channelLength;
    }

    return result;
}

void DoNms(vector<DetectionRes>& detections, float nmsThresh) {
	auto iouCompute = [](float * lbox, float* rbox) {
		float interBox[] = {
			max(lbox[0], rbox[0]), //left
			min(lbox[0] + lbox[2], rbox[0] + rbox[2]), //right
			max(lbox[1], rbox[1]), //top
			min(lbox[1] + lbox[3], rbox[1] + rbox[3]), //bottom
		};

		if (interBox[2] >= interBox[3] || interBox[0] >= interBox[1])
			return 0.0f;

		float interBoxS = (interBox[1] - interBox[0] + 1) * (interBox[3] - interBox[2] + 1);
		return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
	};

	sort(detections.begin(), detections.end(), [=](const DetectionRes & left, const DetectionRes & right) {
		return left.prob > right.prob;
	});

	vector<DetectionRes> result;
	for (unsigned int m = 0; m < detections.size(); ++m) {
		result.push_back(detections[m]);
		for (unsigned int n = m + 1; n < detections.size(); ++n) {
			if (iouCompute((float *)(&detections[m]), (float *)(&detections[n])) > nmsThresh) {
				detections.erase(detections.begin() + n);
				--n;
			}
		}
	}
	detections = move(result);
}


vector<DetectionRes> postProcess(cv::Mat& image, float * output) {
	vector<DetectionRes> detections;
	int total_size = 0;
	for (int i = 0; i < output_shape.size(); i++) {
		auto shape = output_shape[i];
		int size = 1;
		for (int j = 0; j < shape.size(); j++) {
			size *= shape[j];
		}
		total_size += size;
	}

	int offset = 0;
	float * transposed_output = new float[total_size];
	float * transposed_output_t = transposed_output;
	for (int i = 0; i < output_shape.size(); i++) {
		auto shape = output_shape[i];  // nchw
		int chw = shape[1] * shape[2] * shape[3];
		int hw = shape[2] * shape[3];
		for (int n = 0; n < shape[0]; n++) {
			int offset_n = offset + n * chw;
			for (int h = 0; h < shape[2]; h++) {
				for (int w = 0; w < shape[3]; w++) {
					int h_w = h * shape[3] + w;
					for (int c = 0; c < shape[1]; c++) {
						int offset_c = offset_n + hw * c + h_w;
						*transposed_output_t++ = output[offset_c];
					}
				}
			}
		}
		offset += shape[0] * chw;
	}
	vector<vector<int> > shapes;
	for (int i = 0; i < output_shape.size(); i++) {
		auto shape = output_shape[i];
		vector<int> tmp = { shape[2], shape[3], 3, 6 };
		shapes.push_back(tmp);
	}

	offset = 0;
	for (int i = 0; i < output_shape.size(); i++) {
		auto masks = g_masks[i];
		vector<vector<int> > anchors;
		for (auto mask : masks)
			anchors.push_back(g_anchors[mask]);
		auto shape = shapes[i];
		for (int h = 0; h < shape[0]; h++) {
			int offset_h = offset + h * shape[1] * shape[2] * shape[3];
			for (int w = 0; w < shape[1]; w++) {
				int offset_w = offset_h + w * shape[2] * shape[3];
				for (int c = 0; c < shape[2]; c++) {
					int offset_c = offset_w + c * shape[3];
					float * ptr = transposed_output + offset_c;
					ptr[4] = sigmoid(ptr[4]);
					ptr[5] = sigmoid(ptr[5]);
					float score = ptr[4] * ptr[5];
					if (score < obj_threshold)
						continue;
					ptr[0] = sigmoid(ptr[0]);
					ptr[1] = sigmoid(ptr[1]);
					ptr[2] = exponential(ptr[2]) * anchors[c][0];
					ptr[3] = exponential(ptr[3]) * anchors[c][1];

					ptr[0] += w;
					ptr[1] += h;
					ptr[0] /= shape[0];
					ptr[1] /= shape[1];
					ptr[2] /= DETECT_WIDTH;
					ptr[3] /= DETECT_WIDTH;
					ptr[0] -= ptr[2] / 2;
					ptr[1] -= ptr[3] / 2;

					DetectionRes det;;
					det.x = ptr[0];
					det.y = ptr[1];
					det.w = ptr[2];
					det.h = ptr[3];
					det.prob = score;
					detections.push_back(det);
				}
			}
		}
		offset += shape[0] * shape[1] * shape[2] * shape[3];
	}
	delete[]transposed_output;

	int h = DETECT_WIDTH;   //net h
	int w = DETECT_WIDTH;   //net w

	//scale bbox to img
	int width = image.cols;
	int height = image.rows;
	float scale = min(float(w) / width, float(h) / height);
	float scaleSize[] = { width * scale, height * scale };

	//correct box
	for (auto& bbox : detections) {
		bbox.x = (bbox.x * w - (w - scaleSize[0]) / 2.f) / scale;
		bbox.y = (bbox.y * h - (h - scaleSize[1]) / 2.f) / scale;
		bbox.w *= w;
		bbox.h *= h;
		bbox.w /= scale;
		bbox.h /= scale;
	}

	//nms
	float nmsThresh = nms_threshold;
	if (nmsThresh > 0)
		DoNms(detections, nmsThresh);

	return detections;
}

vector<string> split(const string& str, char delim)
{
    stringstream ss(str);
    string token;
    vector<string> container;
    while (getline(ss, token, delim)) {
        container.push_back(token);
    }

    return container;
}

int main( int argc, char* argv[] )
{
    parser::ADD_ARG_STRING("onnxmodel",Desc("input yolov3 caffemodel"),DefaultValue(INPUT_ONNXMODEL),ValueDesc("file"));
    parser::ADD_ARG_INT("C",Desc("channel"),DefaultValue(to_string(INPUT_CHANNEL)));
    parser::ADD_ARG_INT("class",Desc("num of classes"),DefaultValue(to_string(DETECT_CLASSES)));
    parser::ADD_ARG_FLOAT("nms",Desc("non-maximum suppression value"),DefaultValue(to_string(NMS_THRESH)));
    parser::ADD_ARG_INT("batchsize",Desc("batch size for input"),DefaultValue("1"));
    parser::ADD_ARG_STRING("enginefile",Desc("load from engine"),DefaultValue(ENGINE_FILE),ValueDesc("file"));

    //input
    parser::ADD_ARG_STRING("input",Desc("input image file"),DefaultValue(INPUT_IMAGE),ValueDesc("file"));
    parser::ADD_ARG_STRING("evallist",Desc("eval gt list"),DefaultValue(EVAL_LIST),ValueDesc("file"));

    if(argc < 2){
        parser::printDesc();
        exit(-1);
    }

    parser::parseArgs(argc,argv);


    std::unique_ptr<trtNet> net;
    int batchSize = parser::getIntValue("batchsize"); 
    string engineName =  parser::getStringValue("enginefile");
    if(engineName.length() > 0)
    {
        net.reset(new trtNet(engineName));
        assert(net->getBatchSize() == batchSize);
    }
    else
    {
    string onnxmodelFile = parser::getStringValue("onnxmodel");

    //save Engine name
    string saveName = "./ped.trt";
        net.reset(new trtNet(onnxmodelFile,batchSize));
    cout << "save Engine..." << saveName <<endl;
        net->saveEngine(saveName);
    }

    int outputCount = net->getOutputSize()/sizeof(float);
    unique_ptr<float[]> outputData(new float[outputCount]);

    string listFile = parser::getStringValue("evallist");
    list<string> fileNames;
    list<vector<Bbox>> groundTruth;

    if(listFile.length() > 0)
    {
        std::cout << "loading from eval list " << listFile << std::endl; 
        tie(fileNames,groundTruth) = readObjectLabelFileList(listFile);
    }
    else
    {
        string inputFileName = parser::getStringValue("input");
        fileNames.push_back(inputFileName);
    }

    list<vector<DetectionRes>> outputs;
    int classNum = parser::getIntValue("class");
    int c = parser::getIntValue("C");
    int h = parser::getIntValue("H");
    int w = parser::getIntValue("W");
    int batchCount = 0;
    vector<float> inputData;
    inputData.reserve(h*w*c*batchSize);
    vector<cv::Mat> inputImgs;

    auto iter = fileNames.begin();
    for (unsigned int i = 0;i < fileNames.size(); ++i ,++iter)
    {
        const string& filename  = *iter;

        std::cout << "process: " << filename << std::endl;

        cv::Mat img = cv::imread(filename);
        std::cout << "process: " << filename << std::endl;
        vector<float> curInput = prepareImage(img);
        if (!curInput.data())
            continue;
        inputImgs.emplace_back(img);

        inputData.insert(inputData.end(), curInput.begin(), curInput.end());
        batchCount++;

        if(batchCount < batchSize && i + 1 <  fileNames.size())
            continue;

        net->doInference(inputData.data(), outputData.get(),batchCount);

        //Get Output    
        auto output = outputData.get();
        auto outputSize = net->getOutputSize()/ sizeof(float) / batchCount;
        for(int i = 0;i< batchCount ; ++i)
        {    

            auto boxes = postProcess(inputImgs[i],output);
            outputs.emplace_back(boxes);

        //print boxes
            for (int i = 0; i < boxes.size(); ++i)
            {
                cout << boxes[i].prob << ", " << boxes[i].x << ", " << boxes[i].y << ", " << boxes[i].w << ", " << boxes[i].h << endl;
	    }

            cout << "\n" << endl;

            output += outputSize;
        }
        //inputImgs.clear();
        inputData.clear();

        batchCount = 0;
    }
    // draw boxes
    int idx = 1;
    auto iterDet = outputs.begin();
    for (unsigned int i = 0; i < fileNames.size(); ++i, ++iterDet)
    {
            const vector<DetectionRes> &outputI = *iterDet;
            for (auto box : outputI)
            {
                int x = box.x,
                    y = box.y,
                    w = box.w,
                    h = box.h;
                cv::Rect rect = { x, y, w, h };
                cv::rectangle(inputImgs[i], rect, cv::Scalar(255, 255, 0), 2);
             }
             stringstream ss;
             ss << idx;
             string index = ss.str();
             idx++;
             cv::imwrite("./result_" + index + ".jpg", inputImgs[i]);
             cout << "save result to: " << "./result_" + index + ".jpg" << endl;
    }
       

    return 0;
}
