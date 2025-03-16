package gui;

import cnn.*;
import utilities.NetworkFileHandler;
import utilities.mnist.MNISTDataset;
import utilities.mnist.MNISTDownloader;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class Main {

    public static void main(String[] args) throws IOException {
        CNN network;

        // Load network
        File loadFile = NetworkFileHandler.getLoadFile();
        if (loadFile != null) {
            network = NetworkFileHandler.loadNetwork(loadFile);
        } else {
            ArrayList<Layer> layers = new ArrayList<>();
            layers.add(new FullyConnectedLayer(784, 256, ActivationFunction.SIGMOID));
            layers.add(new FullyConnectedLayer(256, 128, ActivationFunction.SIGMOID));
            layers.add(new FullyConnectedLayer(128, 10, ActivationFunction.SOFTMAX));
            network = new CNN(layers, CostFunction.MSE, true);
        }

        boolean trainNetwork = false;
        if (trainNetwork) {
            MNISTDownloader.downloadMNIST();
            MNISTDataset train = new MNISTDataset("mnist_data/train-images-idx3-ubyte", "mnist_data/train-labels-idx1-ubyte");
            MNISTDataset test = new MNISTDataset("mnist_data/t10k-images-idx3-ubyte", "mnist_data/t10k-labels-idx1-ubyte");

            network.train(train, 0.01, 20, 64, 0.01, false);
            network.test(test);

            // Save network
            File saveFile = NetworkFileHandler.getSaveFile();
            if (saveFile != null) {
                NetworkFileHandler.saveNetwork(network, saveFile);
            }
        }

        DrawingPadGUI drawingPad = new DrawingPadGUI(network, 10);
    }

}