package cnn;

public enum ActivationFunction {
    RELU {
        @Override
        public double apply(double x) {
            return Math.max(0, x);
        }
    },

    SIGMOID {
        @Override
        public double apply(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }
    };

    // Default methods
    public abstract double apply(double x);
}
