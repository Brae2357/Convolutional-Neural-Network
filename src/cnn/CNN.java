package cnn;

import utilities.Dataset;

import java.util.List;

public class CNN {
    private List<Layer> layers; // Holds all layers (convolution, pooling, fully connected)
    private CostFunction costFunction; // Cost function for network
    private double learningRate;
    private int epochs;
    private int batchSize;

    public CNN(List<Layer> layers, CostFunction costFunction) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.layers = layers;
        this.costFunction = costFunction;
    }

    // Forward propagate through the entire network
    public Matrix forward(Matrix input) {
        Matrix output = input;
        output = output.flatten(); // TODO ADD FIX TO FLATTEN
        for (Layer layer : layers) {
            output = layer.forward(output); // Pass previous output as the input to the next layer
        }
        return output; // Final output
    }

    // Backward propagate through the entire network
    public void backward(Matrix outputError) {
        Matrix error = outputError;
        for (int i = layers.size() - 1; i >= 0; i--) {
            error = layers.get(i).backward(error);
        }
    }

    // Predict correct output
    public Matrix predict(Matrix input) {
        return forward(input);
    }

    // Train the network on dataset with default learningRate, epochs, and batchSize
    public void train(Dataset dataset) {
        train(dataset, learningRate, epochs, batchSize, null, false);
    }

    // Train the network on dataset with configurable settings
    public void train(Dataset dataset, double learningRate, int maxEpochs, int batchSize, Double targetCost, boolean continuous) {
        int epoch = 1;
        double currentCost = Double.MAX_VALUE;

        while (continuous || (epoch <= maxEpochs && (targetCost == null || currentCost > targetCost))) {
            dataset.shuffleDataset(); // Shuffle at the start of each epoch
            double totalCost = 0;
            int batchCount = 0;

            // Process dataset in mini batches
            while (dataset.hasNextBatch()) {
                Dataset.Batch batch = dataset.getNextBatch(batchSize);
                List<Matrix> inputs = batch.inputs();
                List<Matrix> expectedOutputs = batch.outputs();
                for (int i = 0; i < inputs.size(); i++) {
                    Matrix input = inputs.get(i);
                    Matrix expectedOutput = expectedOutputs.get(i);

                    // Forward pass
                    Matrix predictedOutput = forward(input);

                    // Compute cost
                    Matrix outputError = costFunction.derivative(predictedOutput, expectedOutput);
                    totalCost += costFunction.calculate(predictedOutput, expectedOutput);

                    // Backward pass
                    backward(outputError);
                }

                // Apply accumulated gradients
                for (Layer layer : layers) {
                    if (layer instanceof FullyConnectedLayer) {
                        ((FullyConnectedLayer) layer).applyGradient(learningRate, batchSize);
                        ((FullyConnectedLayer) layer).clearGradient();
                    }
                }
                batchCount++;
            }

            // Compute average cost for each epoch
            currentCost = totalCost / batchCount;
            System.out.println("Epoch " + epoch + ": Cost = " + currentCost);
            epoch++;
        }
    }

    // Test the network on a dataset
    public void test(Dataset dataset) {
        int correct = 0;
        for (int i = 0; i < dataset.getSize(); i++) {
            Matrix output = predict(dataset.getDataAtIndex(i));
            int[] sortedIndices = output.sortFlattenedByIndex();
            int[] expectedIndices = dataset.getOutputAtIndex(i).sortFlattenedByIndex();
            if (sortedIndices[0] == expectedIndices[0]) {
                correct++;
            }
        }

        // Calculate percentage and print results
        double percentage = ((double) correct / dataset.getSize()) * 100;
        String formattedPercentage = String.format("%.2f%%", percentage);
        System.out.println("Test Results: " + correct + "/" + dataset.getSize() + " = " + formattedPercentage);
    }
}
