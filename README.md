# Gibbs Pruning

Implementation of Gibbs pruning for Keras/Tensorflow. arXiv link coming.

Code should work with Tensorflow 1.14, 1.15, and 2.1, but our results were generated with Tensorflow 1.14, so note that other versions may lead to slightly different results.

## Usage

Code and full documentation for Gibbs pruning on 2D convolutional layers is in `gibbs_pruning.py`. `example.py` gives an example of using Gibbs pruning on ResNet-20 with CIFAR-10. Run `python example.py --help` to see all options.
