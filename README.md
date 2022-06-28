# RoadMap

- sketch the main structure of this project
    - separate computing on a device from an interface, but support choice and interchanging
    - device operations directly on data 
    - device-aware struct NDArray {shape, data, device} (ndarray offers the same API on different devices)
    - device unaware struct Tensor {data: NDArray, grad: NDArray}
    - trait GradFn with a function backward: change grad w.r.t. input
    - trait NeuralNetwork
- implement ndarray naively with CPU
- implement tensor with ndarray (tensor shouldn't be aware of the device)
- test backpropagation
- implement high-level API of neural network
- implement multi-threading training
- implement optimizing strategies

