package cnn;

public abstract class Layer {
    public enum LayerType {
        FULLY_CONNECTED
    }
    // Abstract Methods
    public abstract LayerType getType();
    public abstract Matrix forward(Matrix input); // Forward propagation
    public abstract Matrix backward(Matrix input); // Backpropagation
    public abstract int getNumInputs(); // Number of input nodes
    public abstract int getNumOutputs(); // Number of output nodes
}
