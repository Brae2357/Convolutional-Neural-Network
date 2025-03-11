package cnn;

import java.util.List;

public class CNN {
    private List<Layer> layers; // Holds all layers (convolution, pooling, fully connected)
    private double learningRate;
    private int epochs;
    private int batchSize;

    public CNN(double learningRate, int epochs, int batchSize, List<Layer> layers) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.layers = layers;
    }
}
