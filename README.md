# msc-exploring-neural-networks
This project aims to understand and investigates the design and building of different convolutional neural network architectures for video classification. It does this, using TensorFlow and the Keras API with python and the UCF101 dataset. It begins by constructing convolutional neural network models as described in [Karpathy et al., 2014]. It makes some improvements on these models by, for example, replacing the local response normalization layer used with the more modern batch nor- malization technique. It explores the use of different optimization methods and the use of different model inputs that improve on the use of temporal features. It also looks at implementing some pre-trained models from the Kera API and tries to improve on their architecture to include temporal features.
 
 ### Simple CNNs
 
 The project begins with constructing a neural network as described in https://ai.google/research/pubs/pub42455. T
 
 ### using Pre trained models 
 
It then goes on to use pretrained models got signle frame image classification but taking into account. the Temporal Fusion arecheture discribed in https://ai.google/research/pubs/pub42455. Such as 

* Early fusion 
* Late fusion
* Slow fusion

Below shows the accurancy of the various designs


| Pre-trained model|summary |Single Frame | Early fusion| Late Fusion | Slow Fusion | 
| ------------------|:-----------:|:-----------:|:-----------:| ----------:| -----------:|
|  [VGG19](https://keras.io/applications/#vgg19) |image classification model trained on ImageNet  | |              |           |             |
|  [mobilenetv2](https://keras.io/getting-started/functional-api-guide/)|image classification model trained on ImageNet |      |             |           |             |
|  [i3d-kinetics-400](https://tfhub.dev/deepmind/i3d-kinetics-400/1)  |video classification model based on trained for action recognition on Kinetics-400.              |     |            |           |             |

 
Future planes will be to explore RNNs and Capsules( as explained in https://arxiv.org/abs/1710.09829_
 
 
 
