# **Few-Shot Learning with Graph Neural Networks on CIFAR-100**

This is the PyTorch-0.4.0 implementation of few-shot learning on CIFAR-100 with graph neural networks (GNN). And the codes is on the basis of following paper/github/course. <br/>

- [FEW-SHOT LEARNING WITH GRAPH NEURAL NET-WORKS](https://arxiv.org/pdf/1711.04043.pdf)
- https://github.com/vgsatorras/few-shot-gnn
- Course: Deep Learning for Computer Vision, Spring 2018, National Taiwan University

Besides directly end-to-end training both CNN-embedding layers and graph convolution layers, I also tried different training methods and got better results sometimes. <br/>
1. Pretrain the CNN-embedding layers by do classification on CIFAR-100 (excluding few-shot data), and fine-tune them while training graph convolution layer.
2. Pretrain the CNN-embedding layers but fixing them while training graph convolution layer.

## **Dataset**

CIFAR-100 (Get the dataset from torchvision) <br/> 

## **Requirement**

python == 3.6 <br/>
pytorch == 0.4.0 <br/>
torchvision == 0.2.1 <br/>
scikit-image == 0.14.0 <br/>

OS == Linux system (Ubuntu 16.04LTS)

## **Execution**

### **training**

Train for N way M shots <br/>

`python3 main.py 
--data_root 'data' 
--nway N --shots M` <br/>

Pretrain CNN-embedding layer and train for N way M shots <br/>

`python3 main.py 
--data_root 'data' 
--nway N --shots M 
--pretrain` <br/>

Pretrain CNN-embedding layer and train for N way M shots while fixing the CNN-embedding layer <br/>

`python3 main.py 
--data_root 'data' 
--nway N --shots M 
--pretrain --freeze_cnn` <br/>

### **testing**

Test for N way M shot: <br/>
`python main.py --todo 'test' 
--data_root 'data' 
--nway N --shots M 
--load --load_dir [path for folder saving model.pth]` 


## **Experiment Result**

- 5 way (validation / test) <br/>


|                                |       1-shot      |       5-shot      |       10-shot     |
|              :---:             |       :---:       |       :---:       |       :---:       |
| **end-to-end**                 | 37.68 % / 35.60 % | 61.42 % / 63.20 % | 70.05 % / 68.60 % |
| **pretrain and fine-tune**     | 49.66 % / 49.60 % | 63.84 % / 59.00 % | 69.69 % / 67.20 % | 
| **pretrain and fixed**         | 48.30 % / 42.80 % | 63.74 % / 65.60 % | 68.01 % / 67.20 % |

- 20 way (validation / test) <br/>


|                                |       1-shot      |       5-shot      |       10-shot     |
|              :---:             |       :---:       |       :---:       |       :---:       |
| **end-to-end**                 | 19.85 % / 16.55 % | 36.58 % / 35.85 % | 38.34 % / 44.05 % |
| **pretrain and fine-tune**     | 20.85 % / 22.95 % | 35.50 % / 37.15 % | 42.61 % / 41.10 % | 
| **pretrain and fixed**         | 22.68 % / 19.25 % | 35.66 % / 30.75 % | 41.67 % / 42.25 % |

1. The seed for choosing few-shot class is 1. <br/>
2. Only run each experiment for 1 time. Running it for multiple times can get more convincing results. <br/>
3. Note that if increasing the layers of the model too much, end-to-end training might fail. Pretraining CNN-embedding layers can cure this problem. <br/>


## **Contact**
Yi-Lin Sung, r06942076@ntu.edu.tw