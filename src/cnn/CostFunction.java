package cnn;

public enum CostFunction {
    MSE { // Mean Squared Error
        @Override
        public double calculate(Matrix predicted, Matrix expected) {
            // Check if matrices are not the same size
            if (predicted.getRows() != expected.getRows() || predicted.getCols() != expected.getCols()) throw new IllegalArgumentException("Matrix dimensions do not match for cost function.");

            Matrix difference = predicted.subtract(expected); // Get difference between predicted and expected
            Matrix squared = difference.elementWiseMultiply(difference); // Square results

            return squared.sum() / (predicted.getRows() * predicted.getCols()); // Get average cost
        }

        @Override
        public Matrix derivative(Matrix predicted, Matrix expected) {
            // Check if matrices are not the same size
            if (predicted.getRows() != expected.getRows() || predicted.getCols() != expected.getCols()) throw new IllegalArgumentException("Matrix dimensions do not match for cost function.");

            return predicted.subtract(expected).scale(2.0 / (predicted.getRows()) * predicted.getCols());
        }
    },

    CROSS_ENTROPY {
        @Override
        public double calculate(Matrix predicted, Matrix expected) {
            // Check if matrices are not the same size
            if (predicted.getRows() != expected.getRows() || predicted.getCols() != expected.getCols()) throw new IllegalArgumentException("Matrix dimensions do not match for cost function.");

            double cost = 0;
            double[][] predictedData = predicted.toArray();
            double[][] expectedData = expected.toArray();
            for (int row = 0; row < predicted.getRows(); row++) {
                for (int col = 0; col < predicted.getCols(); col++) {
                    double p = predictedData[row][col];
                    double y = expectedData[row][col];
                    p = Math.max(1e-9, Math.min(p, 1 - 1e-9)); // Avoid log(0) by adding small value
                    cost += y * Math.log(p) + (1 - y) * Math.log(1 - p);
                }
            }

            return -cost / predicted.getRows();// * predicted.getCols());
        }

        @Override
        public Matrix derivative(Matrix predicted, Matrix expected) {
            // Check if matrices are not the same size
            if (predicted.getRows() != expected.getRows() || predicted.getCols() != expected.getCols()) throw new IllegalArgumentException("Matrix dimensions do not match for cost function.");

            double[][] gradient = new double[predicted.getRows()][predicted.getCols()];
            double[][] predictedData = predicted.toArray();
            double[][] expectedData = expected.toArray();
            for (int row = 0; row < predicted.getRows(); row++) {
                for (int col = 0; col < predicted.getCols(); col++) {
                    double p = predictedData[row][col];
                    double y = expectedData[row][col];
                    p = Math.max(1e-9, Math.min(p, 1 - 1e-9)); // Prevent division by zero
                    gradient[row][col] = -(y / p) + ((1 - y) / (1 - p));
                }
            }
            return new Matrix(gradient);
        }
    },

    MAE { // Mean Absolute Error
        @Override
        public double calculate(Matrix predicted, Matrix expected) {
            // Check if matrices are not the same size
            if (predicted.getRows() != expected.getRows() || predicted.getCols() != expected.getCols()) throw new IllegalArgumentException("Matrix dimensions do not match for cost function.");

            double cost = 0;
            double[][] predictedData = predicted.toArray();
            double[][] expectedData = expected.toArray();
            for (int row = 0; row < predicted.getRows(); row++) {
                for (int col = 0; col < predicted.getCols(); col++) {
                    double difference = Math.abs(predictedData[row][col] - expectedData[row][col]); // Get absolute difference
                    cost += difference; // Add absolute difference to cost
                }
            }
            return cost / (predicted.getRows() * predicted.getCols()); // Divide by number of elements to get average cost
        }

        @Override
        public Matrix derivative(Matrix predicted, Matrix expected) {
            // Check if matrices are not the same size
            if (predicted.getRows() != expected.getRows() || predicted.getCols() != expected.getCols()) throw new IllegalArgumentException("Matrix dimensions do not match for cost function.");

            double[][] gradient = new double[predicted.getRows()][predicted.getCols()];
            double factor = 1.0 / (predicted.getRows() * predicted.getCols());

            double[][] predictedData = predicted.toArray();
            double[][] expectedData = expected.toArray();
            for (int row = 0; row < predicted.getRows(); row++) {
                for (int col = 0; col < predicted.getCols(); col++) {
                    double difference = predictedData[row][col] - expectedData[row][col];
                    gradient[row][col] = factor * (difference > 0 ? 1 : -1);
                }
            }
            return new Matrix(gradient);
        }
    };

    // Default methods
    public abstract double calculate(Matrix predicted, Matrix expected);
    public abstract Matrix derivative(Matrix predicted, Matrix expected);
}
