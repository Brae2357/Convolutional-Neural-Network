package utilities;

import cnn.*;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.io.*;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;

/**
 * Handles saving and loading a network model to a custom binary file format.
 *
 * <p>The file format is structured as follows:</p>
 *
 * <pre>
 * [int] Magic Number (to verify format validity)
 * [int] Number of Layers
 * [int] Cost Function (enum)
 * [boolean] Data Augmentation Allowed
 *
 * Each Layer Entry:
 * [int] Layer Type (enum)
 *
 * Fully Connected Layer:
 * [int] Number of Input Nodes
 * [int] Number of Output Nodes
 * [int] Activation Function (enum)
 * [Matrix] Weights
 * [Matrix] Biases
 * </pre>
 *
 * <p>Layer Types:</p>
 * <ul>
 *     <li>0 - Fully Connected (FC)</li>
 * </ul>
 *
 * <p>Activation Functions:</p>
 * <ul>
 *     <li>0 - Leaky ReLU</li>
 *     <li>1 - ReLU</li>
 *     <li>2 - Sigmoid</li>
 *     <li>3 - Softmax</li>
 * </ul>
 *
 * <p>Cost Functions:</p>
 * <ul>
 *     <li>0 - Mean Standard Error (MSE)</li>
 *     <li>1 - Cross Entropy</li>
 *     <li>2 - Mean Absolute Error (MAE)</li>
 * </ul>
 *
 * <p>The file uses little-endian format for all numeric values.</p>
 */

public class NetworkFileHandler {
    private static final int MAGIC_NUMBER = 0x434E4E; // "CNN" in hex to identify file form
    private static final String DIRECTORY = System.getProperty("user.home") + "\\ConvNN\\models\\";

    static {
        // Create the directory if it doesn't exist
        File dir = new File(DIRECTORY);
        if (!dir.exists()) {
            dir.mkdirs();
        }
    }

    public static File getSaveFile() {
        FileDialog fileDialog = new FileDialog((Frame) null, "Save CNN Model", FileDialog.SAVE);
        fileDialog.setDirectory(DIRECTORY);
        fileDialog.setFile("*.cnn"); // Default extension filter
        fileDialog.setVisible(true);

        if (fileDialog.getFile() != null) {
            String filePath = fileDialog.getDirectory() + fileDialog.getFile();
            if (!filePath.toLowerCase().endsWith(".cnn")) {
                filePath += ".cnn"; // Ensure .cnn extension
            }
            return new File(filePath);
        }
        return null; // User canceled
    }

    public static File getLoadFile() {
        FileDialog fileDialog = new FileDialog((Frame) null, "Open CNN Model", FileDialog.LOAD);
        fileDialog.setDirectory(DIRECTORY);
        fileDialog.setFile("*.cnn"); // Only show .cnn files
        fileDialog.setVisible(true);

        if (fileDialog.getFile() != null) {
            return new File(fileDialog.getDirectory() + fileDialog.getFile());
        }
        return null; // User canceled
    }

    public static void saveNetwork(CNN network, File file) throws IOException {
        try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(file))) {
            dos.writeInt(MAGIC_NUMBER); // Magic Number
            dos.writeInt(network.getLayers().size()); // Layer count
            dos.writeInt(network.getCostFunction().ordinal()); // Cost Function (enum)
            dos.writeBoolean(network.isAllowAugmenting()); // Allow augmenting

            for (Layer layer : network.getLayers()) {
                dos.writeInt(layer.getType().ordinal()); // Write layer type

                if (layer instanceof FullyConnectedLayer fcLayer) {
                    writeFCLayer(fcLayer, dos); // Write FC Layer
                }

                // TODO: Implement other layer types
            }
        }
    }

    public static CNN loadNetwork(File file) throws IOException {
        try (DataInputStream dis = new DataInputStream(new FileInputStream(file))) {
            int magic = dis.readInt();
            if (magic != MAGIC_NUMBER) {
                throw new IOException("Invalid file format.");
            }

            int numLayers = dis.readInt();
            CostFunction costFunction = CostFunction.values()[dis.readInt()];
            boolean allowAugmenting = dis.readBoolean();

            List<Layer> layers = new ArrayList<>();
            for (int i = 0; i < numLayers; i++) {
                Layer.LayerType layerType = Layer.LayerType.values()[dis.readInt()];

                if (layerType == Layer.LayerType.FULLY_CONNECTED) {
                    FullyConnectedLayer fcLayer = readFCLayer(dis);
                    layers.add(fcLayer);
                }

                // TODO: Implement other layer types
            }

            return new CNN(layers, costFunction, allowAugmenting);
        }
    }

    private static void writeFCLayer(FullyConnectedLayer fcLayer, DataOutputStream dos) throws IOException {
        dos.writeInt(fcLayer.getNumInputs()); // Input Nodes
        dos.writeInt(fcLayer.getNumOutputs()); // Output Nodes
        dos.writeInt(fcLayer.getActivationFunction().ordinal()); // Activation Function

        // Write weights
        for (double[] row : fcLayer.getWeights().toArray()) {
            for (double weight : row) {
                dos.writeDouble(weight);
            }
        }

        // Write biases
        for (double[] row : fcLayer.getBiases().toArray()) {
            for (double bias : row) {
                dos.writeDouble(bias);
            }
        }
    }

    private static FullyConnectedLayer readFCLayer(DataInputStream dis) throws IOException {
        int numInputs = dis.readInt();
        int numOutputs = dis.readInt();
        ActivationFunction activationFunction = ActivationFunction.values()[dis.readInt()];

        // Read weights
        double[][] weightData = new double[numOutputs][numInputs];
        for (int row = 0; row < numOutputs; row++) {
            for (int col = 0; col < numInputs; col++) {
                weightData[row][col] = dis.readDouble();
            }
        }
        Matrix weights = new Matrix(weightData);

        // Read biases
        double[][] biasData = new double[numOutputs][1];
        for (int row = 0; row < numOutputs; row++) {
            biasData[row][0] = dis.readDouble();
        }
        Matrix biases = new Matrix(biasData);

        return new FullyConnectedLayer(numInputs, numOutputs, weights, biases, activationFunction);
    }
}
