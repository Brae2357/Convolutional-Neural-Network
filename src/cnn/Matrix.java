package cnn;

import java.util.Arrays;

/**
 * Represents a mathematical matrix and provides various matrix operations.
 *
 * <p>This class supports fundamental matrix operations such as addition, scaling,
 * matrix multiplication, activations, and flattening. It ensures immutability
 * by copying data where necessary and provides a readable string representation.</p>
 *
 * <p>Features:</p>
 * <ul>
 *     <li>Supports creation from 2D arrays, copies of other matrices, and randomized initialization in the range [-1,1].</li>
 *     <li>Implements matrix addition, subtraction, transposition, scalar multiplication, matrix multiplication, and element-wise multiplication.</li>
 *     <li>Includes activation functions and flattening operations.</li>
 *     <li>Ensures immutability by copying data when necessary.</li>
 * </ul>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * Matrix A = new Matrix(3, 3); // Creates a 3x3 matrix with all zeros
 * Matrix B = Matrix.randomized(3, 3); // Creates a 3x3 matrix with values between -1 and 1
 * Matrix C = A.add(B); // Adds matrices A and B
 * Matrix D = A.subtract(B); // Subtracts matrices A and B
 * double E = A.sum(); // Sums all elements in A
 * Matrix F = A.transpose(); // Interchanges rows and columns
 * Matrix G = A.scale(2.5); // Multiplies matrix A by scalar 2.5
 * Matrix H = A.multiply(B); // Performs matrix multiplication
 * Matrix I = A.elementWiseMultiply(B); // Multiplies elements in A and B
 * Matrix J = A.exp(); // Apply e^x to each element
 * Matrix K = A.activate(ActivationFunction.RELU); // Applies ReLU activation to each element in A
 * Matrix L = A.activationDerivative(ActivationFunction.RELU); // Applies ReLU derivative to each element in A
 * Matrix M = A.softmax(); // Transforms matrix into probability distribution
 * Matrix N = A.flatten(); // Converts matrix to a column matrix
 * int[] O = A.sortFlattenedByIndex(); // Sorts matrix indices in descending order
 * }</pre>
 *
 * @author Braeden West
 * @version 1.3 (2025-03-15)
 * @since 2025-03-11
 */

public class Matrix {
    private final double[][] data;
    private final int rows, cols;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public Matrix(double[][] data) {
        this.rows = data.length;
        this.cols = data[0].length;
        this.data = Matrix.copy(data);
    }

    public Matrix(Matrix other) {
        this.rows = other.rows;
        this.cols = other.cols;
        this.data = Matrix.copy(other.data);
    }

    public static double[][] copy(double[][] source) {
        double[][] copy = new double[source.length][source[0].length];
        for (int row = 0; row < source.length; row++) {
            copy[row] = Arrays.copyOf(source[row], source[row].length);
        }
        return copy;
    }

    public static Matrix randomized(int rows, int cols) {
        Matrix matrix = new Matrix(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                matrix.data[row][col] = Math.random() * 2 - 1; // Random value between -1 and 1
            }
        }
        return matrix;
    }

    // Matrix addition
    public Matrix add(Matrix other) {
        // Check if matrices not equal sizes
        if (this.rows != other.rows || this.cols != other.cols) throw new IllegalArgumentException("Matrix dimensions do not match for addition.");

        Matrix result = new Matrix(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.data[row][col] = this.data[row][col] + other.data[row][col];
            }
        }
        return result;
    }

    // Matrix subtraction
    public Matrix subtract(Matrix other) {
        // Check if matrices not equal sizes
        if (this.rows != other.rows || this.cols != other.cols) throw new IllegalArgumentException("Matrix dimensions do not match for addition.");

        Matrix result = new Matrix(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.data[row][col] = this.data[row][col] - other.data[row][col];
            }
        }
        return result;
    }

    // Matrix sum
    public double sum() {
        double sum = 0;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                sum += data[row][col];
            }
        }
        return sum;
    }

    // Matrix transposition
    public Matrix transpose() {
        Matrix result = new Matrix(cols, rows); // Flips dimensions
        for (int row = 0; row < this.rows; row++) {
            for (int col = 0; col < this.cols; col++) {
                result.data[col][row] = this.data[row][col];
            }
        }
        return result;
    }

    // Matrix scalar multiplication
    public Matrix scale(double scalar) {
        Matrix result = new Matrix(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.data[row][col] = this.data[row][col] * scalar;
            }
        }
        return result;
    }

    // Matrix multiplication
    public Matrix multiply(Matrix other) {
        // Check if this column size does not equal other row size
        if (this.cols != other.rows) throw new IllegalArgumentException("Matrix dimensions do not match for multiplication.");

        // Matrix multiplication
        Matrix result = new Matrix(this.rows, other.cols); // Result matrix size
        for (int row = 0; row < this.rows; row++) {
            for (int col = 0; col < other.cols; col++) {
                double sum = 0;
                for (int thisCol = 0; thisCol < this.cols; thisCol++) {
                    sum += this.data[row][thisCol] * other.data[thisCol][col];
                }
                result.data[row][col] = sum;
            }
        }
        return result;
    }

    // Element wise matrix multiplication
    public Matrix elementWiseMultiply(Matrix other) {
        // Check if matrices are not the same size
        if (this.rows != other.rows || this.cols != other.cols) throw new IllegalArgumentException("Matrix dimensions do not match for element-wise multiplication.");

        // Element multiplication
        Matrix result = new Matrix(this.rows, this.cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.data[row][col] = this.data[row][col] * other.data[row][col];
            }
        }
        return result;
    }

    // Apply e^x to each element
    public Matrix exp() {
        Matrix result = new Matrix(rows, cols);
        for (int row = 0; row < this.rows; row++) {
            for (int col = 0; col < this.cols; col++) {
                result.data[row][col] = Math.exp(data[row][col]);
            }
        }
        return result;
    }

    // Pass each element through the activation function
    public Matrix activate(ActivationFunction activationFunction) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.data[row][col] = activationFunction.apply(this.data[row][col]);
            }
        }
        return result;
    }

    // Pass each element through the derivative of the activation function
    public Matrix activationDerivative(ActivationFunction activationFunction) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.data[row][col] = activationFunction.derivative(this.data[row][col]);
            }
        }
        return result;
    }

    // Transform elements into a probability distribution
    public Matrix softmax() {
        Matrix expMatrix = exp(); // Apply e^x to each element
        double sum = expMatrix.sum(); // Sum of all e^x values
        return expMatrix.scale(1.0 / sum);
    }

    // Flatten into a column vector
    public Matrix flatten() {
        Matrix result = new Matrix(this.rows * this.cols, 1);
        int index = 0;
        for(int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.data[index][0] = this.data[row][col];
                index++;
            }
        }
        return result;
    }

    // Sort the flattened matrix and return a list of the indices from largest to smallest
    public int[] sortFlattenedByIndex() {
        // Check if matrix is not flattened
        if (this.cols != 1) throw new IllegalArgumentException("Matrix dimensions are not flattened for sorting.");

        Integer[] indices = new Integer[this.rows];
        for (int i = 0; i < this.rows; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, (a, b) -> Double.compare(this.data[b][0], this.data[a][0]));
        return Arrays.stream(indices).mapToInt(i -> i).toArray();
    }

    public int getCols() {
        return cols;
    }

    public int getRows() {
        return rows;
    }

    public double[][] toArray() {
        return Matrix.copy(data);
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        for (double[] row : data) {
            stringBuilder.append(Arrays.toString(row)).append("\n");
        }
        return stringBuilder.toString();
    }
}
