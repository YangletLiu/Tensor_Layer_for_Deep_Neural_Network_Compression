# Tensor_Layer_for_Deep_Neural_Network_Compression
Apply CP, Tucker, TT/TR, HT to compress neural networks. Train from scratch.

## NOTE: Tensorly's CP decomposition is temporarily down.
## Usage

First, import required modules by
```
from common import *
from decomposition import *
from nets import *
```

Then, specify a neural network model.  The user can choose from any model provided by the nets package, or can define a new architecture using pytorch.  Examples:
```
model0 = LeNet()
model1 = VGG('VGG16')
```

Finally, go through the training and testing process using the `run_all` function.  
The function has a few parameters:
* `dataset`: choose a dataset from mnist, cifar10, and cifar100
* `model`: the neural network model defined in the last step
* `decomp`: the method of decomposition; defaulted to be `None` (undecomposed)
* `i`: number of iterations for training; defaulted to be 100
* `rate`: learning rate; defaulted to be 0.05

This example below runs the CP-decomposed LeNet on the MNIST dataset for 150 iterations with a learning rate of 0.1:
```
run_all('mnist', model0, decomp='cp', i=150, rate=0.1)
```
This function creates three subdirectories:
* `data`: stores the dataset
* `models`: stores the trained network
* `cureves`: stores arrays of training and testing accuracy across different iterations in `.npy` format


## Method
I aim to decompose the neural network in both the convolutional portion and the fully connected portion, using popular tensor decomposition algorithms such as CP, Tucker, TT and HT.  In doing so, I hope to speedup both the training and the inference process and reduce the number of parameters without signicant sacrifices in terms of accuracy.

## CP 
CP decomposition works fine with classifying the MNIST dataset, it can compress the network without significant loss in accuracy compared to the original, uncompressed network. However, as noted in paper Lebedev et al., CP cannot has problems dealing with larger networks, and it is often unstable. In my experiments, the decomposition process not only takes an intolerably long time, but it also consumes a lot of RAM. For a convolutional layer with size larger than 512 x 512, the CP decompositon becomes infeasible in terms of memory. Moreover, the CP decomposed network is highly sensitive to the learning rate, and requires the learning rate to be as small as 1e-5 for learning to take place.

## Tucker
Tucker decomposition is strictly superior to CP in almost every way. It has more sucess decomposing larger networks, and requires less resources in terms of runtime and memory. It is also more tolerant to larger values of learning rates, allowing the network to learn faster. The network decomposed with Tucker also learns faster, i.e., yields greater accuracy in fewer epochs (see the analysis of performance graphs for details).

## Tensor Train (TT)
In my implementation of compression using tensor train, I picked out the two dimensions in the convolutional layer corresponding to the input/output channels, then I matricized the tensor, decomposed the result to matrix product state, and reshaped them back to 4-dimensional tensors.  This gives us two decomposed convolutional layer for every convolutional layer in the original network.  Experimentally, this method yields better results than Tucker, and has similar rate of compression and speedup as Tucker.  In the two papers by Novikov et al., the authors proposed using a transformation to higher-order tensor before applying TT decomposition.

## Tensor Ring (TR) UNDER CONSTRUCTION
TR decomposition is highly similar to TT, differing only in an additional non-trivial mode on the first and last tensor core. The way it is applied to neural networks is also similar, although researchers argue that TR has greater expressiveness.

## Hierarchical Tucker (HT)
Have not yet developed.

## Experiments
I tested the performance of the three compression methods against the uncompressed network on the MNIST and the CIFAR10 datasets.  I tried to keep all hyperparameters the same for all tests, including rank, number of epochs, and learning rate.  However, as CP is too sensitive to learning rate, I give it a much smaller value for learning rate.

<div align=center><img width="400" src="https://github.com/hust512/Tensor_Layer_for_Deep_Neural_Network_Compression/blob/master/asset/MNIST_train.png"/></div>

<div align=center>Figure 1. Training accuracy comparision on the MNIST dataset.</div>


<div align=center><img width="400" src="https://github.com/hust512/Tensor_Layer_for_Deep_Neural_Network_Compression/blob/master/asset/MNIST_test.png"/></div>

<div align=center>Figure 2. Testing accuracy comparision on the MNIST dataset.</div>



From this performance graph, we can see that even though the CP-decomposed network has higher training accuracy at the end, its testing accuracy is low, likely resulting from overfitting due to a finer learning rate.  TT-decomposed network learns faster than Tucker and yields better results.  In terms of run time, the four networks do not differ from each other significantly. 


<div align=center><img width="400" src="https://github.com/hust512/Tensor_Layer_for_Deep_Neural_Network_Compression/blob/master/asset/CIRAR10_train.png"/></div>

<div align=center>Figure 3. Training accuracy comparision on the CIAR10 dataset.</div>


<div align=center><img width="400" src="https://github.com/hust512/Tensor_Layer_for_Deep_Neural_Network_Compression/blob/master/asset/CIFAR10_test.png"/></div>

<div align=center>Figure 4. Testing accuracy comparision on the CIAR10 dataset.</div>



For the uncompressed network, the average time for each epoch is around 38 seconds, the average time for the Tucker-decomposed network is 26 seconds, and the average time for the TT-decomposed network is 27 seconds.  In terms of accuracy, the TT-decomposed network outperforms Tucker in both training and testing, and is almost comparable to the original network before compression.

## Profiling
In a typical training process, the profiling output is:
```
         510155235 function calls (507136221 primitive calls) in 2053.806 seconds

   Ordered by: internal time
   List reduced from 824 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    49401  743.334    0.015  743.334    0.015 {method 'item' of 'torch._C._TensorBase' objects}
    19550  222.799    0.011  222.799    0.011 {method 'run_backward' of 'torch._C._EngineBase' objects}
  5747602  107.058    0.000  107.058    0.000 {method 'add_' of 'torch._C._TensorBase' objects}
  1183200  102.777    0.000  102.777    0.000 {built-in method conv2d}
  3010000   59.049    0.000  140.632    0.000 functional.py:192(normalize)
  3010000   45.654    0.000  205.622    0.000 functional.py:43(to_tensor)
  1915802   45.251    0.000   45.251    0.000 {method 'mul_' of 'torch._C._TensorBase' objects}
   394400   41.219    0.000   41.219    0.000 {built-in method batch_norm}
  3010000   40.894    0.000   40.894    0.000 {method 'tobytes' of 'numpy.ndarray' objects}
  1915850   39.603    0.000   39.603    0.000 {method 'zero_' of 'torch._C._TensorBase' objects}
  3010000   32.589    0.000   32.589    0.000 {method 'div' of 'torch._C._TensorBase' objects}
  3010000   25.541    0.000   25.541    0.000 {method 'contiguous' of 'torch._C._TensorBase' objects}
  6020000   25.312    0.000   25.312    0.000 {built-in method as_tensor}
  3010000   24.033    0.000   24.033    0.000 {method 'sub_' of 'torch._C._TensorBase' objects}
  3010000   20.338    0.000  116.898    0.000 Image.py:2644(fromarray)
  6020000   19.078    0.000   19.078    0.000 {method 'transpose' of 'torch._C._TensorBase' objects}
  3034650   18.921    0.000   18.921    0.000 {method 'view' of 'torch._C._TensorBase' objects}
  3010000   18.467    0.000   18.467    0.000 {method 'float' of 'torch._C._TensorBase' objects}
  3010000   15.967    0.000   15.967    0.000 {method 'clone' of 'torch._C._TensorBase' objects}
    19550   15.331    0.001  168.502    0.009 sgd.py:71(step)
```

## References
### List of relevent papers:

* Lebedev, V., Ganin, Y., Rakhuba, M., Oseledets, I. and Lempitsky, V., 2015. Speeding-up convolutional neural networks using fine-tuned CP-decomposition. In 3rd International Conference on Learning Representations, ICLR 2015-Conference Track Proceedings.
  * *Notes: applies CP to convlayers.*
  
* Kim, Y.D., Park, E., Yoo, S., Choi, T., Yang, L. and Shin, D., 2015. Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications. arXiv, pp.arXiv-1511.
  * *Notes: applies Tucker to convlayers*

* Garipov, T., Podoprikhin, D., Novikov, A. and Vetrov, D., 2016. Ultimate tensorization: compressing convolutional and FC   layers alike. arXiv, pp.arXiv-1611.
  * *Notes: applies TT to both conv and FC layers* 

* Novikov, A., Podoprikhin, D., Osokin, A. and Vetrov, D.P., 2015. Tensorizing neural networks. In Advances in neural information processing systems (pp. 442-450).
  * *Notes: applies TT to FC layers*

* Wang, W., Sun, Y., Eriksson, B., Wang, W. and Aggarwal, V., 2018. Wide compression: Tensor ring nets. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 9329-9338).
  * *Notes: applies TR to both conv and FC layers*

* Cohen, N., Sharir, O., Levine, Y., Tamari, R., Yakira, D. and Shashua, A., 2017. Analysis and Design of Convolutional Networks via Hierarchical Tensor Decompositions. arXiv, pp.arXiv-1705.
  * *Notes: applies HT to convlayers*

* Yang, Y., Krompass, D. and Tresp, V., 2017. Tensor-train recurrent neural networks for video classification. arXiv preprint arXiv:1707.01786.
  * *Notes: applies TT to sequential models*

* Yin, M., Liao, S., Liu, X.Y., Wang, X. and Yuan, B., 2020. Compressing Recurrent Neural Networks Using Hierarchical Tucker Tensor Decomposition. arXiv, pp.arXiv-2005.
  * *Notes: applies HT to LSTMs*
  
### Related Github repos:

https://github.com/jacobgil/pytorch-tensor-decompositions

https://github.com/JeanKossaifi/tensorly-notebooks

https://github.com/vadim-v-lebedev/cp-decomposition

https://github.com/timgaripov/TensorNet-TF
