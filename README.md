# ReadingMaterials
Reading materials for Scientific Computation and Machine Learning

Contents:
- [ReadingMaterials](#readingmaterials)
  - [Scientific Computation](#scientific-computation)
    - [Finite element](#finite-element)
    - [Finite difference](#finite-difference)
    - [Neural Network-based](#neural-network-based)
      - [Deep neural network-based](#deep-neural-network-based)
      - [Extreme learning machine-based](#extreme-learning-machine-based)
    - [Operator Learning](#operator-learning)
    - [Computer Aided Design](#computer-aided-design)
      - [Mesh Generation](#mesh-generation)
      - [3D Vision](#3d-vision)
  - [Machine Learning](#machine-learning)
    - [Basic](#basic)
    - [CNN](#cnn)
    - [Transformer](#transformer)
    - [Generative Models](#generative-models)
  - [Coding Related](#coding-related)
    - [Python](#python)
    - [Deep learning framework 1: PyTorch](#deep-learning-framework-1-pytorch)
    - [Deep learning framework 2: Tensorflow](#deep-learning-framework-2-tensorflow)
    - [NVIDIA drivers](#nvidia-drivers)
    - [Some implementations](#some-implementations)

## Scientific Computation

### Finite element

- books

|Title|Year|Authors|Link|Description|
|:--:|:--:|:--:|:--:|:--:|
|有限元方法的数学基础|2004|王烈衡，许学军|[book](./assets/note_169745363673169539d.pdf)||

### Finite difference

|Title|Year|Authors|Link|Description|
|:--:|:--:|:--:|:--:|:--:|
||||||

### Neural Network-based

#### Deep neural network-based

|Title|Year|Authors|Link|Description|
|:--:|:--:|:--:|:--:|:--:|
|DGM: A deep learning algorithm for solving partial differential equations|2018|Justin Sirignano, Konstantinos Spiliopoulos|[JCP](https://www.sciencedirect.com/science/article/pii/S0021999118305527)|DGM|
|Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations|2019|Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis|[JCP](https://www.sciencedirect.com/science/article/pii/S0021999118307125)|PINN|
|Weak adversarial networks for high-dimensional partial differential equations|2020|Zang, Y., Bao, G., Ye, X., & Zhou, H|[JCP](https://www.sciencedirect.com/science/article/pii/S0021999120301832)|WAN|

#### Extreme learning machine-based

|Title|Year|Authors|Link|Description|
|:--:|:--:|:--:|:--:|:--:|
|Extreme learning machine: Theory and applications|2006|Guang-Bin Huang, Qin-Yu Zhu, Chee-Kheong Siew|[NeuroComputing](https://www.sciencedirect.com/science/article/pii/S0925231206000385)|ELM|
|Local Extreme Learning Machines and Domain Decomposition for Solving Linear and Nonlinear Partial Differential Equations|2021|Suchuan Dong, Zongwei Li|[JCP](https://www.sciencedirect.com/science/article/pii/S0045782521004606)|locELM|
|Bridging Traditional and Machine Learning-Based Algorithms for Solving PDEs: The Random Feature Method|2022|Jingrun Chen, Xurong Chi, Weinan E & Zhouwang Yang|[JML](https://global-sci.org/intro/article_detail/jml/21029.html)|RFM|
|The Random Feature Method for Time-Dependent Problems|2023|Jing-Run Chen, Weinan E & Yi-Xin Luo|[EAJAM](https://global-sci.org/intro/article_detail/eajam/21718.html)|ST-RFM|

### Operator Learning

|Title|Year|Authors|Link|Description|
|:--:|:--:|:--:|:--:|:--:|
|Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators|2021|L. Lu, P. Jin, G. Pang, Z. Zhang, & G. E. Karniadakis. |[Nature Machine Intelligence](https://doi.org/10.1038/s42256-021-00302-5)|DeepONet|
|Fourier Neural Operator for Parametric Partial Differential Equations|2021|Zongyi Li, Nikola Borislavov Kovachki, Kamyar Azizzadenesheli, Burigede liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar|[ICLR](https://openreview.net/forum?id=c8P9NQVtmnO)|FNO|
|Meta-Auto-Decoder: a Meta-Learning-Based Reduced Order Model for Solving Parametric Partial Differential Equations|2023|Zhanhong Ye, Xiang Huang, Hongsheng Liu & Bin Dong|[Communications on Applied Mathematics and Computation](https://link.springer.com/article/10.1007/s42967-023-00293-7)|MAD|
|Geometry-Informed Neural Operator for Large-Scale 3D PDEs|2023|Zongyi Li, Nikola Borislavov Kovachki, Chris Choy, Boyi Li, Jean Kossaifi, Shourya Prakash Otta, Mohammad Amin Nabian, Maximilian Stadler, Christian Hundt, Kamyar Azizzadenesheli, Anima Anandkumar|[Preprint](https://arxiv.org/abs/2309.00583)||

### Computer Aided Design

#### Mesh Generation

|Title|Year|Authors|Link|Description|
|:--:|:--:|:--:|:--:|:--:|
|Tetrahedral Meshing in the Wild|2018|Yixin Hu, Qingnan Zhou, Xifeng Gao, Alec Jacobson, Denis Zorin, Daniele Panozzo|[TOG](https://dl.acm.org/doi/10.1145/3197517.3201353)||
|Decoupling simulation accuracy from mesh quality|2018|TESEO SCHNEIDER, YIXIN HU, JÉRÉMIE DUMAS, XIFENG GAO, DANIELE PANOZZO, DENIS ZORIN|[TOG](https://dl.acm.org/doi/10.1145/3272127.3275067)||
|ABC: A Big CAD Model Dataset For Geometric Deep Learning|2019|Sebastian Koch, Albert Matveev, Zhongshi Jiang, Francis Williams, Alexey Artemov, Evgeny Burnaev, Marc Alexa, Denis Zorin, Daniele Panozzo|[CVPR](https://cs.nyu.edu/~zhongshi/publication/abc-dataset/)||
|Deep Geometric Prior for Surface Reconstruction|2019|Francis Williams, Teseo Schneider, Claudio Silva, Denis Zorin, Joan Bruna, and Daniele Panozzo|[CVPR](https://openaccess.thecvf.com/content_CVPR_2019/papers/Williams_Deep_Geometric_Prior_for_Surface_Reconstruction_CVPR_2019_paper.pdf)||
|Isogeometric High Order Mesh Generation|2021|Teseo Schneidera, Daniele Panozzob, Xianlian Zhou|[Computer Methods in Applied Mechanics and Engineering](https://www.sciencedirect.com/science/article/pii/S0045782521004357)||
|A Large-Scale Comparison of Tetrahedral and Hexahedral Elements for Solving Elliptic PDEs with the Finite Element Method|2022|TESEO SCHNEIDER, YIXIN HU, XIFENG GAO, JÉRÉMIE DUMAS, DENIS ZORIN, DANIELE PANOZZO|[TOG](https://dl.acm.org/doi/10.1145/3508372)||
|DEF: Deep Estimation of Sharp Geometric Features in 3D Shapes|2022|Albert Matveev, Ruslan Rakhimov, Alexey Artemov, Gleb Bobrovskikh, Vage Egiazarian, Emil Bogomolov, Daniele Panozzo, Denis Zorin, Evgeny Burnaev|[TOG](https://dl.acm.org/doi/10.1145/3528223.3530140)||
|In-Timestep Remeshing for Contacting Elastodynamics|2023|Zachary Ferguson, Teseo Schneider, Danny Kaufman, Daniele Panozzo|[TOG](https://dl.acm.org/doi/10.1145/3592428)||
|Constrained Delaunay Tetrahedrization: A Robust and Practical Approach|2023|LORENZO DIAZZI, DANIELE PANOZZO, AMIR VAXMAN, MARCO ATTENE|[Siggragh Asia](https://www.physicsbasedanimation.com/2023/09/25/constrained-delaunay-tetrahedrization-a-robust-and-practical-approach/)||

#### 3D Vision

|Title|Year|Authors|Link|Description|
|:--:|:--:|:--:|:--:|:--:|
|An Invitation to 3-D Vision|2010|Yi Ma, Stefano Soatto, Jana Košecká, S. Shankar Sastry|[Springer](https://link.springer.com/book/10.1007/978-0-387-21779-6)||

## Machine Learning

### Basic

- books

|Title|Year|Authors|Link|Description|
|:--:|:--:|:--:|:--:|:--:|
|Deep Learning|2016|Ian Goodfellow and Yoshua Bengio and Aaron Courville|[Web](https://www.deeplearningbook.org/)|花书|

- papers

|Title|Year|Authors|Link|Description|
|:--:|:--:|:--:|:--:|:--:|
|Approximation by superpositions of a sigmoidal function|1989|G. Cybenko|[Mathematics of Control, Signals and Systems](https://link.springer.com/article/10.1007/BF02551274)|万有逼近定理|
|Frequency Principle: Fourier Analysis Sheds Light on Deep Neural Networks|2020|Zhi-Qin John Xu, Yaoyu Zhang, Tao Luo, Yanyang Xiao & Zheng Ma|[CiCP](https://global-sci.org/intro/article_detail/cicp/18395.html)|频率原理|

### CNN

|Title|Year|Authors|Link|Description|
|:--:|:--:|:--:|:--:|:--:|
|ImageNet Classification with Deep Convolutional Neural Networks|2012|Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton|[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)|AlexNet|
|Going Deeper With Convolutions|2015|Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich|[CVPR](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html)|InceptionNet|
|Deep Residual Learning for Image Recognition|2016|Kaiming He,  Xiangyu Zhang, Shaoqing Ren, Jian Sun|[CVPR](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf?_hsenc=p2ANqtz-_QiyyY2TSWPAmgDMPMomcXCfnQJLAbgU7SPqAQszQnwme7O58FCu297FfN9yjmRJQa6K3h)|ResNet|

### Transformer

|Title|Year|Authors|Link|Description|
|:--:|:--:|:--:|:--:|:--:|
|Attention is All you Need|2017|Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin|[NeurIPS](https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)||
|An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale|2021|Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby|[ICLR](https://openreview.net/forum?id=YicbFdNTTy)|ViT|

### Generative Models

|Title|Year|Authors|Link|Description|
|:--:|:--:|:--:|:--:|:--:|
|Auto-Encoding Variational Bayes|2013|Diederik P. Kingma, Max Welling|[ICLR](https://openreview.net/forum?id=33X9fd2-9FyZd)|VAE|
|Generative Adversarial Nets|2014|Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio|[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2014/hash/5ca3e9b122f61f8f06494c97b1afccf3-Abstract.html)|GAN|
|Density estimation using Real NVP|2017|Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio|[ICLR](https://openreview.net/forum?id=HkpbnH9lx)|Real NVP (标准化流)|
|Denoising Diffusion Probabilistic Models|2020|Jonathan Ho, Ajay Jain, Pieter Abbeel|[NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)|DDPM 扩散模型|
|Score-Based Generative Modeling through Stochastic Differential Equations|2021|Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole|[ICLR](https://openreview.net/forum?id=PxTIG12RRHS)|Score SDE 扩散模型连续化|

## Coding Related

### Python

- Homepage: [Python](https://www.python.org/)
- 教程：[菜鸟教程](https://www.runoob.com/python/python-tutorial.html)
- Anaconda 环境管理器：[官网](https://www.anaconda.com/)，[介绍](https://zhuanlan.zhihu.com/p/123188004)
- Virtualenv 环境管理器：[官网](https://virtualenv.pypa.io/en/latest/)，[介绍](https://www.cnblogs.com/doublexi/p/15783355.html)

### Deep learning framework 1: PyTorch

- Homepage: [PyTorch](https://pytorch.org/)
- Official tutorials: [PyTorch tutorials](https://pytorch.org/tutorials/)
- Official documents: [PyTorch Documents](https://pytorch.org/docs/stable/index.html)
- Installation guidelines: [Installation](https://pytorch.org/get-started/locally/)
- Github 教程示例：[Github repo](https://github.com/yunjey/pytorch-tutorial)
- B站视频：[bilibili](https://www.bilibili.com/video/BV1hE411t7RN/?share_source=copy_web)

### Deep learning framework 2: Tensorflow

### NVIDIA drivers

- 驱动下载链接：[Nvidia drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us)
- 官方安装驱动指导：[Installation](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
- Ubuntu 系统安装 N 卡驱动介绍：[Ubuntu installation](https://blog.csdn.net/huiyoooo/article/details/128015155)

### Some implementations

- [Link](coding/some-implementations/)
