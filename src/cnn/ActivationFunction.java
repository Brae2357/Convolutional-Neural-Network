package cnn;

public enum ActivationFunction {
    LEAKY_RELU {
        @Override
        public double apply(double x) {
            return Math.max(0, x);
        }

        @Override
        public double derivative(double x) {
            return x > 0 ? x : 0.01 * x;
        }
    },

    RELU {
        @Override
        public double apply(double x) {
            return Math.max(0, x);
        }

        @Override
        public double derivative(double x) {
            return x > 0 ? 1 : 0;
        }
    },

    SIGMOID {
        @Override
        public double apply(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }

        @Override
        public double derivative(double x) {
            double activation = apply(x);
            return activation * (1 - activation);
        }
    },

    SOFTMAX {
        @Override
        public double apply(double x) {
            throw new UnsupportedOperationException("Softmax requires a matrix, rather than a single value.");
        }

        @Override
        public double derivative(double x) {
            throw new UnsupportedOperationException("Softmax derivative must be computed with matrix");
        }
    };

    // Default methods
    public abstract double apply(double x);
    public abstract double derivative(double x);
}
