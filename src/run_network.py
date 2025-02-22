import mnist_loader
import network

import numpy as np

import torch

def test_image(net):
    myImage = mnist_loader.imageprepare("C:/Users/rhear/Rhea/Research/MPAS/nielsenpytorch/data/Number5.png");
    test_result = [5];
    test_inputs = [torch.reshape(torch.tensor(myImage).to(torch.float64), (784, 1))]
    my_image_test_data = list(zip(test_inputs, test_result))

    for i in range(0,5):
        test_results = [(np.argmax(net.feedforward(x)), y)
                                for (x, y) in my_image_test_data]
        print(test_results)
    

def run_network():
    training_data, test_data = mnist_loader.load_data_from_mnist();
    net = network.Network([784, 30, 10]);
    net.SGD(training_data, 10, 10, 3.0, test_data=test_data);

    test_image(net)



run_network()

