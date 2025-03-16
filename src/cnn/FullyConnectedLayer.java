package cnn;

/**
 * Represents a fully connected layer in a neural network.
 * This layer consists of weights, biases, and an activation function,
 * and supports forward propagation, backpropagation, and gradient descent.
 *
 * <p>The layer applies a linear transformation followed by a non-linear activation function.
 * During backpropagation, it computs gradients for weights and biases and accumulates them
 * for batch gradient updates.</p>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * forward(Matrix input); // Computes the output of the layer given an input
 * backward(Matrix outputGradient); // Computes gradients during backpropagation
 * applyGradient(double learningRate, int batchSize); // Updates the weight and biases using accumulated gradients
 * clearGradient(); // Resets the accumulated gradients.
 * }</pre>
 *
 * @author Braeden West
 * @version 1.0 (2025-03-13)
 * @since 2025-03-13
 */

public class FullyConnectedLayer extends Layer {
    private final int numInputNodes, numOutputNodes;
    private Matrix weights, costGradientWeights;
    private Matrix biases, costGradientBiases;
    private ActivationFunction activation;
    private Matrix weightedInputs, inputs;

    public FullyConnectedLayer(int numInputNodes, int numOutputNodes, ActivationFunction activation) {
        this.numInputNodes = numInputNodes;
        this.numOutputNodes = numOutputNodes;
        this.weights = Matrix.randomized(numOutputNodes, numInputNodes);
        this.biases = Matrix.randomized(numOutputNodes, 1);
        this.activation = activation;

        clearGradient(); // Reset gradients
    }

    public FullyConnectedLayer(int numInputNodes, int numOutputNodes, Matrix weights, Matrix biases, ActivationFunction activation) {
        this.numInputNodes = numInputNodes;
        this.numOutputNodes = numOutputNodes;
        this.weights = weights;
        this.biases = biases;
        this.activation = activation;

        clearGradient(); // Reset gradients
    }

    @Override
    public LayerType getType() {
        return LayerType.FULLY_CONNECTED;
    }

    // Forward propagation through network
    @Override
    public Matrix forward(Matrix input) {
        this.inputs = input;
        weightedInputs = weights.multiply(input); // Add together all the weighted inputs for each node
        weightedInputs = weightedInputs.add(biases); // Add the bias to each node

        Matrix activations;
        if (activation == ActivationFunction.SOFTMAX) {
            activations = weightedInputs.softmax();
        } else {
            activations = weightedInputs.activate(activation);
        }
        return activations;
    }

    // Backpropagation through network with a parameter using derivative of Cost with respect to activation
    @Override
    public Matrix backward(Matrix dC_da) {
        // Derivative of activation with respect to weighted inputs
        Matrix da_dz = weightedInputs.activationDerivative(activation);

        // Derivative of Cost with respect to weighted inputs
        Matrix dC_dz = dC_da.elementWiseMultiply(da_dz);

        // Evaluate the partial derivative of cost with respect to weight
        Matrix dC_dw = dC_dz.multiply(inputs.transpose());

        // Update gradients
        costGradientWeights = costGradientWeights.add(dC_dw);
        costGradientBiases = costGradientBiases.add(dC_dz);

        // Backpropagation to previous layer
        return weights.transpose().multiply(dC_dz);
    }

    // Output layer backpropagation
    public Matrix backward(Matrix predicted, Matrix expected) {
        // Derivative of Cost with respect to weighted inputs
        Matrix dC_dz = predicted.subtract(expected);

        // Evaluate the partial derivative of cost with respect to weight
        Matrix dC_dw = dC_dz.multiply(inputs.transpose());

        // Update gradients
        costGradientWeights = costGradientWeights.add(dC_dw);
        costGradientBiases = costGradientBiases.add(dC_dz);

        // Backpropagation to previous layer
        return weights.transpose().multiply(dC_dz);
    }

    // Applies and clears the average gradients for weight and bias
    public void applyGradient(double learnRate) {
        weights = weights.subtract(costGradientWeights.scale(learnRate));
        biases = biases.subtract(costGradientBiases.scale(learnRate));
    }

    // Clears the gradients for weight and bias
    public void clearGradient() {
        costGradientWeights = new Matrix(weights.getRows(), weights.getCols());
        costGradientBiases = new Matrix(biases.getRows(), biases.getCols());
    }

    @Override
    public int getNumInputs() {
        return numInputNodes;
    }

    @Override
    public int getNumOutputs() {
        return numOutputNodes;
    }

    public ActivationFunction getActivationFunction() {
        return activation;
    }

    public Matrix getWeights() {
        return weights;
    }

    public Matrix getBiases() {
        return biases;
    }
}
