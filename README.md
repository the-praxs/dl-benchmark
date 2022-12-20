# Benchmarking of Neural Network performance between PyTorch and Flux on Julia for High Performance Computing
## Team Members
Pratyush Shukla (ps4534) and Yufeng Duan (yd2284)

## Description
Our aim is to test the performance of Julia’s Deep Learning ecosystem using Flux.jl package against Python’s PyTorch and assess whether Julia’s motto of “Looks like Python, feels like Lisp, runs like C/Fortran” is justified. We aim to perform a detailed benchmarking analysis of PyTorch and Flux.jl performances on Neural Network training and hyperparameter optimization over the High Performance Computing system. 

## Code Structure
```
dl-benchmark
│   README.md
│   benchmark.jl
|   benchmark.py
│
└───model
    │   lenet.py
    │   resnet.py 
    │   lenet.jl
    │   resnet.jl  
```

## Usage
### Pre-requisites
Python requires the following dependencies -
```
Python >= 3.7.0
pytorch >= 1.4.0
argparse >= 1.2.0
time >= 1.6.0
```

Julia requires the following dependencies - 
```
Julia >= 1.5.1
Flux >= 0.12.4
MLDatasets >= 0.6.0
CUDA >= 3.7.0
```

### Running the scripts
Clone the repository: ```git clone https://github.com/the-praxs/dl-benchmark.git```<br/>

For benchmarking in Python: ```python benchmark.py```

| Args        | Values                    | Description                                               |
|-------------|---------------------------|-----------------------------------------------------------|
| device      | cuda (default) / cpu      | Select between CUDA-enabled GPU or CPU for model training |  
| data        | mnist (default) / fashion | Use MNIST or FashionMNIST dataset                         |
| num_workers | 2 (default) / _Integer_   | Number of sub-processes to use for data loading           |
| batch_size  | 128 (default) / _Integer_ | Batch size for data loading                               |
| model       | resnet (default) / lenet  | Select ResNet-18 or LeNet model to train                  |
| optimizer   | sgd (default) / adam      | Use Stochastic Gradient Descent (SGD) or Adam optimizer   |
| lr          | 0.1 (default) / _Integer_ | Learning Rate parameter for the optimizer                 |
| epochs      | 10 (default) / _Integer_  | Number of epochs for model training                       |

For benchmarking in Julia: ```julia benchmark.jl```

To use number of sub-processes use this command: ```julia -p <number of processes> benchmark.jl```

We use Adam as the default optimizer.

| __Args__    | __Values__                 | __Description__                                           |
|-------------|----------------------------|-----------------------------------------------------------|
| use_cuda    | true (default) / false     | Select between CUDA-enabled GPU or CPU for model training |
| batchsize   | 128 (default) / _Integer_  | Batch size for data loading                               |
| η           | 0.1 (default) / _Integer_  | Learning Rate parameter for the optimizer                 |
| λ           | 5e-4 (default) / _Integer_ | L2 regularizer parameter implemented as weight decay      |
| epochs      | 10 (default) / _Integer_   | Number of epochs for model training                       |
| seed        | 42 (default) / _Integer_   | Seeed for data reproducibility                            |
| infotime    | 1 (default) / _Integer_    | Report every _infotime_ epochs                            |

## Results and Observations
Benchmarking performed for MNIST dataset on LeNet model with default values of the scripts.

### Benchmarking TTA against number of workers
Python:
| __Workers__ | __Best Training Accuracy (%)__ | __Total Data Loading Time (s)__ | __Total Epoch Training Time (s)__ | __Total Training Function Time (s)__ | __Average Training Loss (s)__ |
|-------------|-------------------------------|---------------------------------|-----------------------------------|------------------------------------------|------------------------------------|
|  2 |	98.517 | 1.286 | 13.246 | 31.824 | 0.11
|4   |	98.38 |	1.203 |	11.462 | 22.798 |	0.105
|8   |	98.482 | 1.275 | 9.836 | 23.995 | 0.107
|16  |	98.922 |	1.275 |	8.856 |	29.945 | 0.079

Julia:
| __Workers__ | __Best Training Accuracy (%)__ | __Total Data Loading Time (s)__ | __Total Epoch Training Time (s)__ | __Total Training Function Time (s)__ | __Average Training Loss (s)__ |
|-------------|-------------------------------|---------------------------------|-----------------------------------|------------------------------------------|------------------------------------|
|2	|99.37	|1.198	|16.939	|18.624	|0.031
|4	|99.407	|1.122	|18.04	|19.61	|0.029
|8	|99.417	|1.139	|17.414	|19.051	|0.029
|16	|99.41	|1.228	|17.945	|19.647	|0.03

![Epoch training time against Number of Workers](/images/1.png?raw=true "TTA against Number of Workers")

![TTA against Number of Workers](/images/2.png?raw=true "TTA against Number of Workers")

### Benchmarking TTA against batch size
Python:
| __Workers__ | __Best Training Accuracy (%)__ | __Total Data Loading Time (s)__ | __Total Epoch Training Time (s)__ | __Total Training Function Time (s)__ | __Average Training Loss (s)__ |
|-------------|-------------------------------|---------------------------------|-----------------------------------|------------------------------------------|------------------------------------|
|32	|68.638	|2.757	|41.956	|70.38	|0.027
|128	|99.37	|1.198	|16.939	|18.624	|1.511
|512	|98.032	|0.889	|3.336	|16.747	|0.247
|2048	|94.113	|0.807	|1.109	|15.968	|1.426
|8196	|40.37	|0.76	|0.686	|17.325	|2.326

Julia:
| __Workers__ | __Best Training Accuracy (%)__ | __Total Data Loading Time (s)__ | __Total Epoch Training Time (s)__ | __Total Training Function Time (s)__ | __Average Training Loss (s)__ |
|-------------|-------------------------------|---------------------------------|-----------------------------------|------------------------------------------|------------------------------------|
|32	  |99.523	|1.346	|50.161	|52.54	|0.027
|128	|99.37	|1.198	|16.939	|18.624	|0.031
|512	|98.286	|1.169	|6.651	|8.145	|0.052
|2048	|96.7	|1.128	|4.455	|5.849	|0.099
|8196	|90.503	|1.144	|4.614	|6.031	|0.309

![Epoch training time against Batch Size](/images/3.png?raw=true "TTA against Batch Size")

![TTA against Batch Size](/images/4.png?raw=true "TTA against Batch Size")

From the above results, we oberve that total loop time is more for Julia than Python. This is because PyTorch utilizes CUDA libraries that are developed in C++ and communicate with other Python libraries that have underlying implementation in C++. Hence, communication between different parts of the function is faster in Python than Julia. Howevever, Julia outperforms Python in terms of total training time so TTA is overall better in case of Julia than Python. Julia requires lesser number of workers and batch size for obtaining higher accuracy than Python. Because Julia uses LLVM as the compiler and does Just-In-Time compilation it has faster raw computation speed. Julia also utilizes Multiple Dispatch that maps a tuple of arguments to a return value. This means a particular set of arguments will result in one return type that is selected by Julia at run-time from multiple calls. This type of polymorphism makes Julia exceptionally fast.<br/>

However Julia lacks in terms of documentation and strong community support vis-a-vis Python. Since Julia is a relatively new language, it depends on the communities to enhance its usability in various domains. As it focuses more on the scientific community, its more versatile to use in R&D enviornments rather than enterprise production systems. Volatile deprecation of methods in Julia also tends to affect its backwards compatibility with modules.<br/>

It can be concluded that Julia is suitable for those interested in exploring Artificial Intelligence as a topic of research on a deeper level than those who want to explore Applied Artificial Intelligence for the industries.

## Dataset and Code:
* [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
* [PyTorch based models](https://github.com/pytorch/examples)
* [Flux-based models](https://fluxml.ai/Flux.jl/v0.2/examples/logreg.html)
* [Deep Learning benchmarks](https://github.com/avik-pal/DeepLearningBenchmarks)

## References
* [Deep Learning with Julia using Flux.jl](https://deeplearningwithjulia.com/)
* [THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/)
* [Metalhead](https://github.com/FluxML/Metalhead.jl)
* [Hyperopt](https://github.com/baggepinnen/Hyperopt.jl)
* [Flux.jl on MNIST — Variations of a theme](https://towardsdatascience.com/flux-jl-on-mnist-variations-of-a-theme-c3cd7a949f8c)
* [Flux.jl on MNIST — A performance analysis](https://towardsdatascience.com/flux-jl-on-mnist-a-performance-analysis-c660c2ffd330)
* [A Swift Introduction To Flux For Julia (With CUDA)](https://towardsdatascience.com/a-swift-introduction-to-flux-for-julia-with-cuda-9d87c535312c)
* [Torch-TensorRT](https://pytorch.org/TensorRT/)
* [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
* [Neural Network Benchmarks](https://towardsdatascience.com/neural-network-benchmarks-82d48425c21b)
* [High-Performance GPU Computing in the Julia Programming Language](https://developer.nvidia.com/blog/gpu-computing-julia-programming-language/)
* [PyTorch from a Flux ML Perspective](https://python.plainenglish.io/python-experience-in-machine-learning-from-julia-perspective-fe24e42eee4a)
