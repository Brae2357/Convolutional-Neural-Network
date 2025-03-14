package cnn;

public enum ActivationFunction {
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

    SIGMOID {
        @Override
        public double apply(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }

        @Override
        public double derivative(double x) {
            double sigmoid = apply(x);
            return sigmoid * (1 - sigmoid);
        }
    };

    // Default methods
    public abstract double apply(double x);
    public abstract double derivative(double x);
}
