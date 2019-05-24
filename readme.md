# TRTForYolov3-tiny

## Desc

    tensorRT for Yolov3-tiny (onnx to tensorrt)

### Test Enviroments

    Ubuntu  18.04/ win10
    TensorRT 5.1.5.0/5.1.2.2/5.0.4.3
    CUDA 10.0/9.0

### Models

    Convert your model to onnx model, if your model is trained by DarkNet, check my repo https://github.com/faedtodd/Tensorrt-Yolov3-tiny  


### Run Sample

```bash
#build source code
git submodule update --init --recursive
mkdir build
cd build && cmake .. && make && make install
cd ..

#for fp16

#for yolov3-tiny-608 
./install/runYolov3 --onnxmodel=./ped.onnx --enginefile=./ped.trt --input=./test.jpg
```


### Performance

Model | GPU | Mode | Inference Time
-- | -- | -- | -- 
Yolov3-tiny-608 |  GTX 1060 | fp16 | 6.9335ms
Yolov3-tiny-608 |  Tx2 | fp16 | 20.3274ms
Yolov3-416 |  GTX 1060 | fp16 | 26.817ms
Yolov3-416 |  Tx2 | fp16 | 74.1937ms

