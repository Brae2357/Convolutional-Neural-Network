package cnn;

public abstract class Layer {
    // Abstract Methods
    public abstract Matrix forward(Matrix input); // Forward propagation
    public abstract Matrix backward(Matrix input); // Backpropagation
}
