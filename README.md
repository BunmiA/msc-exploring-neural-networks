# msc-exploring-neural-networks
 This project investigates different neural network archieture for emotion recognition
 
 ### Simple CNNs
 
 The project begins with constructing a neural network as described in https://ai.google/research/pubs/pub42455. The paper suggest using a Local response normalization layer but it was swapped 
 
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

 
 ### RNNs
 
 
 
 ### Capsules 
 
 The project also aims to invetigate the capsuleNet archetecure as explained in https://arxiv.org/abs/1710.09829
 
 
