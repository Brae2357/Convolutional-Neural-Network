package utilities;

import cnn.Matrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * Abstract class for handling datasets in a neural network.
 * Provides functionality to store, shuffle, and retrieve mini-batches
 * of data for training and evaluation.
 *
 * <p>Features:</p>
 * <ul>
 *     <li>Stores input matrices and corresponding output labels</li>
 *     <li>Shuffles data to improve training generalization</li>
 *     <li>Supports retrieval of mini-batches for stochastic gradient descent</li>
 * </ul>
 *
 * <p>Usage:</p>
 * <pre>
 * Dataset dataset = new CustomDataset();
 * Dataset.Batch batch = dataset.getNextBatch(32);
 * </pre>
 *
 * <p>Subclasses should implement data loading mechanisms specific to the dataset format.</p>
 *
 * @author Braeden West
 * @version 1.1 (2025-03-13)
 * @since 1.0 (2025-03-12)
 */

public abstract class Dataset {
    private List<Matrix> data; // List of input matrices
    private List<Matrix> outputs; // List of output matrices
    private int[] shuffledIndices; // For randomized batches
    private int currentIndex = 0; // Current index for batch
    private Random random = new Random();

    public Dataset(List<Matrix> data, List<Matrix> outputs) {
        this.data = data;
        this.outputs = outputs;
        shuffleDataset();
    }

    // Shuffles dataset to allow variation in gradient for batches
    public void shuffleDataset() {
        shuffledIndices = IntStream.range(0, data.size()).toArray(); // Sequential indices
        for (int i = shuffledIndices.length - 1; i > 0; i--) { // Shuffle
            int j = random.nextInt(i + 1);
            int temp = shuffledIndices[i];
            shuffledIndices[i] = shuffledIndices[j];
            shuffledIndices[j] = temp;
        }
        currentIndex = 0; // Reset batch index
    }

    // Returns true if there is untested data in this shuffle
    public boolean hasNextBatch() {
        return currentIndex < data.size();
    }

    // Returns next batch, which includes the data and expected outputs, returns null if all data has been used (reshuffle needed)
    public Batch getNextBatch(int batchSize) {
        List<Matrix> batchData = new ArrayList<>();
        List<Matrix> batchOutputs = new ArrayList<>();

        for (int i = 0; i < batchSize; i++) {
            if (currentIndex >= data.size()) {
                if (i == 0) {
                    return null; // Return null if there is no data in this batch
                }
                return new Batch(batchData, batchOutputs);
            }
            int dataIndex = shuffledIndices[currentIndex];
            batchData.add(data.get(dataIndex));
            batchOutputs.add(outputs.get(dataIndex));
            currentIndex++;
        }

        return new Batch(batchData, batchOutputs);
    }

    public Matrix getDataAtIndex(int index) {
        return data.get(index);
    }

    public Matrix getOutputAtIndex(int index) {
        return outputs.get(index);
    }

    public int getSize() { return data.size(); }

    // Batch record
        public record Batch(List<Matrix> inputs, List<Matrix> outputs) {
    }
}
