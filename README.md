# Tensor_Layer_for_Deep_Neural_Network_Compression
Apply CP, Tucker, TT/TR, HT to compress neural networks. Train from scratch.

## Method
I aim to decompose the neural network in both the convolutional portion and the fully connected portion, using popular tensor decomposition algorithms such as CP, Tucker, TT and HT.  In doing so, I hope to speedup the training process and reduce the number of parameters without signicant sacrifices in terms of accuracy.

## CP 
CP decomposition works fine with classifying the MNIST dataset, it can compress the network without significant loss in accuracy compared to the original, uncompressed network. However, CP cannot has many problems dealing with larger networks such as VGG for CIFAR10. The decomposition not only takes an intolerably long time, but it also consumes a lot of RAM. For a convolutional layer with size larger than 512 x 512, the CP decompositon becomes infeasible in terms of memory. Moreover, the CP decomposed network is highly sensitive to the learning rate, and requires the learning rate to be as small as 1e-5 for learning to take place.

## Tucker
Tucker decomposition is strictly superior to CP in almost every way. It has more sucess decomposing larger networks, and requires less resources in terms of runtime and memory. It is also more tolerant to larger values of learning rates, allowing the network to learn faster. The network decomposed with Tucker also learns faster, i.e., yields greater accuracy in fewer epochs (see the analysis of performance graphs for details).

## Tensor Train (TT)
In my implementation of compression using tensor train, I picked out the two dimensions in the convolutional layer corresponding to the input/output channels, then I matricized the tensor, decomposed the result to matrix product state, and reshaped them back to 4-dimensional tensors.  This gives us two decomposed convolutional layer for every convolutional layer in the original network.  Experimentally, this method yields better results than Tucker, and has similar rate of compression and speedup as Tucker.

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

Lebedev, V., Ganin, Y., Rakhuba, M., Oseledets, I. and Lempitsky, V., 2015. Speeding-up convolutional neural networks using fine-tuned CP-decomposition. In 3rd International Conference on Learning Representations, ICLR 2015-Conference Track Proceedings.

*Notes: applies CP to convlayers.*

Garipov, T., Podoprikhin, D., Novikov, A. and Vetrov, D., 2016. Ultimate tensorization: compressing convolutional and FC layers alike. NIPS 2016 workshop: Learning with Tensors: Why Now and How?

*Notes: applies TT to both conv and FC layers* 

Novikov, A., Podoprikhin, D., Osokin, A. and Vetrov, D.P., 2015. Tensorizing neural networks. In Advances in neural information processing systems (pp. 442-450).

*Notes: applies TT to FC layers*

Wang, W., Sun, Y., Eriksson, B., Wang, W. and Aggarwal, V., 2018. Wide compression: Tensor ring nets. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 9329-9338).

*Notes: applies TR to both conv and FC layers*

Cohen, N., Sharir, O., Levine, Y., Tamari, R., Yakira, D. and Shashua, A., 2017. Analysis and Design of Convolutional Networks via Hierarchical Tensor Decompositions. arXiv, pp.arXiv-1705.

*Notes: applies HT to convlayers*

Yang, Y., Krompass, D. and Tresp, V., 2017. Tensor-train recurrent neural networks for video classification. In International Conference on Machine Learning, pp. 3891-3900. 2017.

*Notes: applies TT to sequential models*

### Related Github repos:

https://github.com/jacobgil/pytorch-tensor-decompositions

https://github.com/JeanKossaifi/tensorly-notebooks

