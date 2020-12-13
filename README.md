# NN Convertor for converting Artificial Neural Network to Spiking Neural Network

The `README` document and comment in project is not prepared well yet, these will be supplemented later.

## Introduction

Deep-learning neural networks such as convolutional
neural network (CNN) have shown great potential
as a solution for difficult vision problems, such as object
recognition. Spiking neural networks (SNN)-based architectures
have shown great potential as a solution for realizing
ultra-low power consumption using spike-based neuromorphic
hardware.


This implementation of this tool referenced many conversion methods of [snn toobox](https://snntoolbox.readthedocs.io/) and it's related papaers. You can refer to it for more functionalities.

## Supports

Currently, the tool only supports models that created and trained by `Keras`, and supports conversion for `Dense`, `Conv2D`, `xxPooling2D`,`Flatten` and `BatchNormalization`. There are also some other restrictions for the ANN Model, which will be detailed later in `README`.


## References

[1] [SNN Toolbox](https://snntoolbox.readthedocs.io/en/latest/)

[2] [Spiking Deep Convolutional Neural Networks for Energy-Efficient
Object Recognition](https://link.springer.com/article/10.1007%2Fs11263-014-0788-3)

[3] [Spiking Deep Convolutional Neural Networks for Energy-Efficient
Object Recognition](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.721.2413&rep=rep1&type=pdf)

[4] [Conversion of Continuous-Valued
Deep Networks to Efficient
Event-Driven Networks for Image
Classification](https://www.frontiersin.org/articles/10.3389/fnins.2017.00682/full)