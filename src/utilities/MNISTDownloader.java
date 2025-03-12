/**
 * MNISTDownloader.java
 *
 * This class handles downloading and extracting the MNIST dataset. It retrieves the dataset
 * from Google's cloud storage, extracts the gzip-compressed files, and stores them in the
 * designated directory.
 *
 * Features:
 * - Downloads training and test images along with their labels.
 * - Extracts gzip-compressed files.
 * - Skips downloading if files already exist.
 * - Deletes compressed files after extraction.
 *
 * Author: Braeden West
 * Created: 3/12/2025
 * Updated: 3/12/2025
 * Version 1.0 (Initial implementation)
 *
 * Usage:
 *  MNISTDownloader.downloadMNIST(); // Downloads and extracts MNIST dataset
 *
 *  Notes:
 *  - Requires an internet connection for downloading.
 *  - Creates a directory "mnist_data/" to store extracted files.
 */

package utilities;

import java.io.*;
import java.net.URL;
import java.util.zip.GZIPInputStream;

public class MNISTDownloader {
    private static final String TRAIN_IMAGES_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz";
    private static final String TRAIN_LABELS_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz";
    private static final String TEST_IMAGES_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz";
    private static final String TEST_LABELS_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz";
    private static final String MNIST_DIR = "mnist_data/";

    public static void downloadMNIST() throws IOException {
        File dir = new File(MNIST_DIR);
        if (!dir.exists()) {
            dir.mkdir();
        }

        downloadAndExtract(TRAIN_IMAGES_URL, MNIST_DIR + "train-images-idx3-ubyte");
        downloadAndExtract(TRAIN_LABELS_URL, MNIST_DIR + "train-labels-idx1-ubyte");
        downloadAndExtract(TEST_IMAGES_URL, MNIST_DIR + "t10k-images-idx3-ubyte");
        downloadAndExtract(TEST_LABELS_URL, MNIST_DIR + "t10k-labels-idx1-ubyte");

        System.out.println("MNIST dataset downloaded successfully!");
    }

    private static void downloadAndExtract(String url, String outputFile) throws IOException {
        String compressedFile = outputFile + ".gz";
        if (new File(outputFile).exists()) {
            System.out.println(outputFile + " already exists, download skipped.");
            return;
        }

        // Compressed File
        System.out.println("Downloading " + url);
        try (BufferedInputStream in = new BufferedInputStream(new URL(url).openStream());
             FileOutputStream fileOutputStream = new FileOutputStream(compressedFile)) {
            byte[] dataBuffer = new byte[1024];
            int bytesRead;
            while ((bytesRead = in.read(dataBuffer, 0, 1024)) != -1) {
                fileOutputStream.write(dataBuffer, 0, bytesRead);
            }
        }

        // Extract gz file
        try (GZIPInputStream gzipInputStream = new GZIPInputStream(new FileInputStream(compressedFile));
             FileOutputStream out = new FileOutputStream(outputFile)) {
            byte[] buffer = new byte[1024];
            int len;
            while ((len = gzipInputStream.read(buffer)) > 0) {
                out.write(buffer, 0, len);
            }
        }

        new File(compressedFile).delete(); // Delete compressed file after extracted
    }
}
