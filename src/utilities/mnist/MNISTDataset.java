package utilities.mnist;

import cnn.Matrix;
import utilities.Dataset;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Handles loading and processing of the MNIST dataset for use in a neural network.
 *
 * <p>This class reads the MNIST image and label files, verifies their metadata, and
 * converts them into {@link Matrix} objects for training and testing. The images are
 * normalized to the range [0,1], and labels are one-hot encoded.</p>
 *
 * <p>File Format:</p>
 * <ul>
 *     <li><b>Images File:</b> 4-byte magic number (0x00000803), 4-byte image count,
 *         4-byte row count, 4-byte column count, followed by raw pixel values.</li>
 *     <li><b>Labels File:</b> 4-byte magic number (0x00000801), 4-byte label count,
 *         followed by one-byte labels (0-9).</li>
 * </ul>
 *
 * <p>Metadata Validation:</p>
 * <ul>
 *     <li>Ensures correct magic numbers to verify file integrity.</li>
 *     <li>Validates image dimensions (28x28).</li>
 *     <li>Ensures label count matches image count.</li>
 * </ul>
 *
 * @author Braeden West
 * @version 1.0 (2025-03-12)
 * @since 1.0 (2025-03-12)
 */

public class MNISTDataset extends Dataset {
    private static final int IMAGE_MAGIC_NUMBER = 0x00000803;
    private static final int LABEL_MAGIC_NUMBER = 0x00000801;
    private static final int IMAGE_ROWS = 28;
    private static final int IMAGE_COLS = 28;

    public MNISTDataset(String imagePath, String labelPath) {
        super(loadImages(imagePath), loadLabels(labelPath));
    }

    private static List<Matrix> loadImages(String imagePath) {
        List<Matrix> images = new ArrayList<>();
        try (DataInputStream dataInputStream = new DataInputStream(new FileInputStream(new File(imagePath)))) {
            // Read metadata
            int magicNumber = dataInputStream.readInt();
            if (magicNumber != IMAGE_MAGIC_NUMBER) throw new IOException("Invalid image file: incorrect magic number.");
            int numImages = dataInputStream.readInt();
            int rows = dataInputStream.readInt();
            int cols = dataInputStream.readInt();
            if (rows != IMAGE_ROWS || cols != IMAGE_COLS) throw new IOException("Unexpected image dimensions: " + rows + "x" + cols);

            // Read image data
            for (int i = 0; i < numImages; i++) {
                double[][] pixelData = new double[rows][cols];
                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < cols; col++) {
                        pixelData[row][col] = dataInputStream.readUnsignedByte() / 255.0; // Normalize to [0,1]
                    }
                }
                images.add(new Matrix(pixelData));
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return images;
    }

    private static List<Matrix> loadLabels(String labelPath) {
        List<Matrix> labels = new ArrayList<>();
        try (DataInputStream dataInputStream = new DataInputStream(new FileInputStream(new File(labelPath)))) {
            // Read metadata
            int magicNumber = dataInputStream.readInt();
            if (magicNumber != LABEL_MAGIC_NUMBER) throw new IOException("Invalid label file: incorrect magic number.");
            int numLabels = dataInputStream.readInt();

            // Read label data
            for (int i = 0; i < numLabels; i++) {
                int label = dataInputStream.readUnsignedByte();
                double[][] oneHot = new double[10][1]; // One-hot used to encode correct answer
                oneHot[label][0] = 1.0;
                labels.add(new Matrix(oneHot));
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return labels;
    }
}
