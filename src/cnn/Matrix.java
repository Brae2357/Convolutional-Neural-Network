package cnn;

import java.util.Arrays;

public class Matrix {
    private double[][] data;
    private int rows, cols;

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
