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
    private Matrix weights, costGradientWeights;
    private Matrix biases, costGradientBiases;
    private ActivationFunction activation;
    private Matrix previousOutput, activatedOutput;

    public FullyConnectedLayer(int numInputNodes, int numOutputNodes, ActivationFunction activation) {
        this.weights = Matrix.randomized(numOutputNodes, numInputNodes);
        this.biases = Matrix.randomized(numOutputNodes, 1);
        this.activation = activation;

        clearGradient(); // Reset gradients
    }

    // Forward propagation through network
    @Override
    public Matrix forward(Matrix input) {
        previousOutput = input;
        Matrix weightedInputs = weights.multiply(input); // Add together all the weighted inputs for each node
        Matrix biasedOutput = weightedInputs.add(biases); // Add the bias to each node
        activatedOutput = biasedOutput.activate(activation); // Pass output through activation function
        return activatedOutput;
    }

    // Backpropagation through network
    @Override
    public Matrix backward(Matrix outputGradient) {
        // Compute the derivative of the activation function
        Matrix activationDerivative = activatedOutput.activationDerivative(activation);

        // Compute delta (error) using element-wise multiplication
        Matrix delta = outputGradient.elementWiseMultiply(activationDerivative);

        // Compute gradients for weights and biases
        Matrix inputTranspose = previousOutput.transpose();
        Matrix weightGradient = delta.multiply(inputTranspose);
        Matrix biasGradient = delta;

        // Accumulate gradients
        costGradientWeights = costGradientWeights.add(weightGradient);
        costGradientBiases = costGradientBiases.add(biasGradient);


        // Compute and return gradient for previous layer
        return weights.transpose().multiply(delta);
    }

    // Applies and clears the average gradients for weight and bias
    public void applyGradient(double learnRate, int batchSize) {
        weights = weights.subtract(costGradientWeights.scale(learnRate));
        biases = biases.subtract(costGradientBiases.scale(learnRate));
    }

    // Clears the gradients for weight and bias
    public void clearGradient() {
        costGradientWeights = new Matrix(weights.getRows(), weights.getCols());
        costGradientBiases = new Matrix(biases.getRows(), biases.getCols());
    }
}
