# Multi-Scale Deep Compressive Sensing Network 
## Abstract

With joint learning of the sampling and recovery, the deep learning-based compressive sensing (DCS) has shown significant improvement in performance and running time reduction. Its reconstructed image, however, losses high-frequency content especially at low subrates. It is understood due to relatively much low-frequency information captured into the sampling matrix. This behaviour happens similarly in the multi-scale sampling scheme which also samples more low-frequency components. This paper proposes a multi-scale DCS (MS-DCSNet) based on convolutional neural network. Firstly, we convert image signal using multiple scale-based wavelet transform. Then, the signal is captured through the convolution block by block across scales. The initial reconstructed image is directly recovered from multi-scale measurements. Multi-scale wavelet convolution is utilized to enhance the final reconstruction quality. The network learns to perform both multi-scale in sampling and reconstruction thus results in better reconstruction quality.

## Implementation 
This is the test source code implemented with MatconvNet [1] using DagNN network. The trained CSNet [2] are taken from [3], MWCNN is used from [4, 5]. This implementation is motivated from [6, 7]. 

## Results
### Set 5	
Set5 |    CSNet     |   MS-CSNet1   |   MS-CSNet2   |   MS-DCSNet3   |  

---------------------------------------------------------------------

rate | PSNR  SSIM   |  PSNR   SSIM  |  PSNR   SSIM  |  PSNR   SSIM  |

0.1  | 32.30	0.902	|  30.66	0.855	|  32.44	0.904 |  33.39	0.917 |

0.2  | 35.63	0.945	|  34.06	0.924	|  35.82	0.947	|  36.56	0.951 |

0.3  | 37.90	0.963	|  36.51	0.952	|  38.20	0.965	|  38.74	0.967 |


Set14
---------------------------------------------------------------------

0.1  | 28.91	0.812	|  27.81	0.778	|  29.10	0.815	|  29.67	0.828 |

0.2  | 31.86	0.891	|  30.69	0.874	|  32.05	0.893	|  32.51	0.900 |

0.3  | 33.99	0.928	|  32.86	0.917	|  34.30	0.930	|  34.71	0.934 |


## Usage
Please cite this work if you use our soure code. 
T. N. Canh and B. Jeon, "Multi-Scale Deep Compressive Sensing Network," IEEE International Conference on Visual Communication and Image Processing, 2018. 

@inproceedings{Canh_VCIP18,

  title={Multi-Scale Deep Compressive Sensing Network},
  
  author={Thuong, Nguyen Canh and Byeungwoo, Jeon},
  
  booktitle={IEEE International Conference on Visual Communication and Image Processing},
  
  pages={},
  
  year={2018}
  
}
  
## Reference 
[1] A. Vedaldi et al., “Matconvnet: Convolutional neural networks for Matlab,” Proc. ACM Inter. Conf. Multi., pp. 689 – 692, 2015.

[2] S. Wuzhen et al., “Deep network for compressed image sensing,” Proc. IEEE Inter. Conf. Mult. Expo, pp.  877 – 882, 2017.

[3] CSNet pre-trained network, available at https://github.com/wzhshi/CSNet

[4] P. Liu et al., “Multi-level Wavelet-CNN for Image Restoration,” [online] at arXiv:1805.07071, 2018. 

[5] MWCNN Source code, available at https://github.com/lpj0/MWCNN

[6] K. Zhang et al., “Beyond a gaussian denoiser: residual learning of deep CNN for image denoising,” IEEE Trans. Image Process., vol. 26, no. 7, pp. 3142 – 3155, 2017.

[7] DnCNN source code, available at https://github.com/cszn/DnCNN




## Disclaimer

Copyright (c) 2018 Thuong Nguyen Canh

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
