package cnn;

public class FullyConnectedLayer extends Layer {
    private Matrix weights;
    private Matrix biases;
    private ActivationFunction activation;

    public FullyConnectedLayer(int numInputNodes, int numOutputNodes, ActivationFunction activation) {
        this.weights = Matrix.randomized(numOutputNodes, numInputNodes);
        this.biases = Matrix.randomized(numOutputNodes, 1);
        this.activation = activation;
    }
}
