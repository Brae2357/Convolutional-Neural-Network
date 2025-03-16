package gui;

import cnn.CNN;
import cnn.Matrix;
import utilities.ImageAugmentor;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.image.BufferedImage;

public class DrawingPadGUI extends JFrame {
    private static final int PAD_SIZE = 280; // Size of drawing pad
    private static final int IMAGE_SIZE = 28; // Size for input data
    private static final int STROKE_SIZE = 15;

    private final DrawPanel drawPanel;
    private final JLabel previewLabel;
    private final PredictionPanel predictionPanel;
    private CNN network; // Trained network

    public DrawingPadGUI(CNN network, int outputSize) {
        this.network = network;

        setTitle("Drawing Pad");
        setSize(800, 400);
        setLayout(new BorderLayout());
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // Create drawing panel
        drawPanel = new DrawPanel(this, PAD_SIZE, IMAGE_SIZE, STROKE_SIZE);

        // Create preview panel
        previewLabel = new JLabel();
        previewLabel.setPreferredSize(new Dimension(PAD_SIZE, PAD_SIZE)); // Enlarge preview to pad size

        JPanel previewPanel = new JPanel();
        previewPanel.add(previewLabel);

        // Prediction panel
        predictionPanel = new PredictionPanel(outputSize);

        // Add components to frame
        add(drawPanel, BorderLayout.WEST);
        add(previewPanel, BorderLayout.CENTER);
        add(predictionPanel, BorderLayout.EAST);

        updatePreview();
        setVisible(true);
    }

    // Converts drawing to input data and updates the preview
    public void updatePreview() {
        BufferedImage dataImage = drawPanel.getScaledImage();
        previewLabel.setIcon(new ImageIcon(dataImage.getScaledInstance(PAD_SIZE, PAD_SIZE, Image.SCALE_FAST)));
        Matrix prediction = network.predict(drawPanel.getImageData());
        predictionPanel.updatePredictions(prediction.toArray(), prediction.sortFlattenedByIndex());
    }
}

// Panel for drawing
class DrawPanel extends JPanel {
    private final Color BACKGROUND_COLOR = Color.BLACK;
    private final Color DRAW_COLOR = Color.WHITE;
    private final double ERASING_MULTIPLIER = 2;

    private final DrawingPadGUI parent;
    private final BufferedImage canvas;
    private final Graphics2D graphics2D;

    private final int padSize;
    private final int imageSize;
    private final int strokeSize;

    public DrawPanel(DrawingPadGUI parent, int padSize, int imageSize, int strokeSize) {
        this.parent = parent;
        this.padSize = padSize;
        this.imageSize = imageSize;
        this.strokeSize = strokeSize;

        setPreferredSize(new Dimension(padSize, padSize));
        setBackground(getBackground());

        // Create blank image
        canvas = new BufferedImage(padSize, padSize, BufferedImage.TYPE_BYTE_GRAY);
        graphics2D = canvas.createGraphics();
        graphics2D.setColor(Color.black);
        graphics2D.fillRect(0, 0, padSize, padSize);

        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                draw(e.getX(), e.getY(), e.getButton() == MouseEvent.BUTTON3); // Check if right mouse button
            }
        });

        addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                draw(e.getX(), e.getY(), SwingUtilities.isRightMouseButton(e)); // Check if right mouse button
            }
        });
    }

    private void draw(int x, int y, boolean isErasing) {
        graphics2D.setColor(isErasing ? BACKGROUND_COLOR : DRAW_COLOR); // Use background color to erase
        graphics2D.setStroke(new BasicStroke(isErasing ? (float) (strokeSize * ERASING_MULTIPLIER) : strokeSize, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND)); // Erasing increases stroke size
        graphics2D.drawLine(x, y, x, y);
        repaint();
        parent.updatePreview();
    }

    @Override
    protected void paintComponent(Graphics graphics) {
        super.paintComponent(graphics);
        graphics.drawImage(canvas, 0, 0, null);
    }

    public void clear() {
        graphics2D.setColor(BACKGROUND_COLOR);
        graphics2D.fillRect(0, 0, padSize, padSize);
        repaint();
        parent.updatePreview();
    }

    public BufferedImage getScaledImage() {
        Image scaled = canvas.getScaledInstance(imageSize, imageSize, Image.SCALE_SMOOTH);
        BufferedImage image = new BufferedImage(imageSize, imageSize, BufferedImage.TYPE_BYTE_GRAY);
        Graphics graphics = image.getGraphics();
        graphics.drawImage(scaled, 0, 0, null);
        graphics.dispose();

        return image;
    }

    public Matrix getImageData() {
        BufferedImage image = getScaledImage();
        return ImageAugmentor.imageToMatrix(image);
    }
}

// Panel for predictions
class PredictionPanel extends JPanel {
    private final int numToShow;

    public PredictionPanel(int numToShow) {
        this.numToShow = numToShow;

        setLayout(new GridLayout(numToShow, 2, 5, 5));
        setBorder(BorderFactory.createTitledBorder("Predictions"));
    }

    public void updatePredictions(double[][] probabilities, int[] sortedIndices) {
        removeAll();

        for (int i = 0; i < numToShow; i++) {
            int index = sortedIndices[i];
            JLabel digitLabel = new JLabel(getWordForDigit(index) + ": "); // TODO add a way to work with multiple datasets
            JLabel percentageLabel = new JLabel(String.format("%.2f%%", probabilities[index][0] * 100));

            add(digitLabel);
            add(percentageLabel);
        }
        revalidate();
        repaint();
    }

    private static String getWordForDigit(int digit) {
        String[] words = { "Zero", "One", "Two", "Three", "Four",
                           "Five", "Six", "Seven", "Eight", "Nine" };
        return words[digit % 10];
    }
}