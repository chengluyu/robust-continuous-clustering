# Robust Continuous Clustering #

## Introduction ##

This is a MATLAB implementation of the RCC and RCC-DR algorithms presented in the following paper ([paper](http://www.pnas.org/content/early/2017/08/28/1700770114.abstract)):

Sohil Atul Shah and Vladlen Koltun. Robust Continuous Clustering. Proceedings of the National Academy of Sciences (PNAS), 2017.

If you use this code in your research, please cite our paper.
```
@article{shah2017robust,
  title={Robust continuous clustering},
  author={Shah, Sohil Atul and Koltun, Vladlen},
  journal={Proceedings of the National Academy of Sciences},
  volume={114},
  number={37},
  pages={9814--9819},
  year={2017},
  publisher={National Acad Sciences}
}
```

The source code and dataset are published under the MIT license. See [LICENSE](LICENSE) for details. In general, you can use the code for any purpose with proper attribution. If you do something interesting with the code, we'll be happy to know. Feel free to contact us.

We include two external packages in the codebase ([CMG](http://www.cs.cmu.edu/~jkoutis/cmg.html) and [Geometry Processing Toolbox](https://github.com/alecjacobson/gptoolbox)). These packages are under a BSD-style license. See [External/README.txt](source/External/README.txt) for details.

##### The MATLAB code provided in this repository can be used to reproduce the accuracy results reported in the paper. The runtime reported in the paper was based on a faster C++ implementation. #####

## Setup ##

One should add the MEX files of CMG package to MATLAB path before running the RCC and RCC-DR algorithms. To do so, in the MATLAB console run the following command.

```
> cd External/CMG/
> MakeCMG
```

## Running Robust Continuous Clustering ##

The RCC and RCC-DR program takes three parameters: a file storing the features of the data samples and their edge set, a variable indicating the maximum total iteration and a variable indicating the maximum iteration for each graduated non-convexity level.

We have provided an MNIST dataset file in the [Data](Data) folder. For example, you can run RCC and RCC-DR from the MATLAB console as follows:

```
> [clustAssign,numcomponents,optTime,gtlabels,nCluster] = RCC('Data/MNIST.mat', 100, 4);
> [clustAssign,numcomponents,optTime,gtlabels,nCluster] = RCCDR('Data/MNIST.mat', 100, 4);
```
The other preprocessed datasets can be found in gdrive [folder](https://drive.google.com/drive/folders/1vN4IpmjJvRngaGkLSyKVsPaoGXL02mFf?usp=sharing).

### Evaluation ###
To evaluate the cluster assignment using various measures, use [evaluate.m](Toolbox/evaluate.m) from the Toolbox folder. In MATLAB console, run
```
[ARI,AMI,NMI,ACC] = evaluate(clustAssign,numcomponents,gtlabels,nCluster);
```

### Creating input ###

The input file is a .mat file that stores features of the 'N' data samples in a matrix format N x D. In the MNIST data provided in the repository, N=70000, D=784. It should also contains edge set stored under variable 'w' in a matrix format numpairs x 2 and a vector of ground truth label to be used for evaluation. 

To construct edge set and to create preprocessed input file from the raw feature file, use [edgeConstruction.py](Toolbox/edgeConstruction.py) from the Toolbox folder. Run the python program in console,
```
python edgeConstruction.py --dataset MNIST.pkl --samples 70000 --prep 'minmax' --k 10 --algo 'mknn'
```
Note that .pkl file should be placed in the Data folder.


## Other Implementation ##
1. [Python Implementation](https://github.com/yhenon/pyrcc) by Yann Henon