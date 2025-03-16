package utilities;

import cnn.Matrix;

import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.util.Random;

public class ImageAugmentor {
    private static final Random random = new Random();

    public static Matrix augment(Matrix data) {
        BufferedImage image = matrixToImage(data); // Convert to image
        BufferedImage augmentedData = augment(image); // Apply augmentation
        return imageToMatrix(augmentedData); // Convert back to matrix
    }

    public static BufferedImage augment(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();

        BufferedImage transformedImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D graphics2D = transformedImage.createGraphics();

        // Apply random transformation
        AffineTransform transform = new AffineTransform();

        // Random rotation (-15 to 15 degrees)
        double angle = Math.toRadians(random.nextInt(31) - 15);
        transform.rotate(angle, width / 2.0, height / 2.0);

        // Random scaling (90% to 110%)
        double scale = 0.9 + (random.nextDouble() * 0.2);
        transform.scale(scale, scale);

        // Random translation (shift up to 3 pixels)
        int shiftX = random.nextInt(7) - 3;
        int shiftY = random.nextInt(7) - 3;
        transform.translate(shiftX, shiftY);

        // Apply transformations
        graphics2D.setTransform(transform);
        graphics2D.drawImage(image, 0, 0, null);
        graphics2D.dispose();

        return transformedImage;
    }

    public static BufferedImage matrixToImage(Matrix matrix) {
        BufferedImage image = new BufferedImage(matrix.getCols(), matrix.getRows(), BufferedImage.TYPE_BYTE_GRAY);
        double[][] data = matrix.toArray();
        for (int y = 0; y < matrix.getRows(); y++) {
            for (int x = 0; x < matrix.getCols(); x++) {
                int value = (int) (data[y][x] * 255); // Convert [0,1] to [0,255]
                int rgb = new Color(value, value, value).getRGB();
                image.setRGB(x, y, rgb);
            }
        }
        return image;
    }

    public static Matrix imageToMatrix(BufferedImage image) {
        double[][] pixels = new double[image.getHeight()][image.getWidth()];
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int color = new Color(image.getRGB(x, y)).getRed(); // Get grayscale value
                pixels[y][x] = (color / 255.0); // Normalize to [0,1]
            }
        }
        return new Matrix(pixels);
    }
}
