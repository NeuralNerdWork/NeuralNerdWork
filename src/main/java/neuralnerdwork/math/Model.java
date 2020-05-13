package neuralnerdwork.math;

import java.util.stream.Stream;

public class Model {
    private int nextParameterIndex = 0;

    public ParameterMatrix createParameterMatrix(int rows, int cols) {
        final int start = nextParameterIndex;
        nextParameterIndex += rows * cols;

        return new ParameterMatrix(start, rows, cols);
    }

    public ParameterVector createParameterVector(int length) {
        final int start = nextParameterIndex;
        nextParameterIndex += length;

        return new ParameterVector(start, length);
    }

    public ScalarParameter createScalarParameter() {
        return new ScalarParameter(nextParameterIndex++);
    }

    public int size() {
        return nextParameterIndex;
    }

    public int[] variables() {
        return Stream.iterate(0, n -> n + 1)
                     .limit(nextParameterIndex)
                     .mapToInt(n -> n)
                     .toArray();
    }

    public Binder createBinder() {
        return new Binder(0, nextParameterIndex);
    }

    public static class Binder {
        private final int start;
        private final Double[] values;
        Binder(int start, int length) {
            this.start = start;
            values = new Double[length];
        }

        public int[] variables() {
            return Stream.iterate(start, n -> n + 1)
                         .limit(values.length)
                         .mapToInt(n -> n)
                         .toArray();
        }

        public double get(int key) {
            return values[key - start];
        }

        public Double put(int key, Double value) {
            if (key >= start && key < start + values.length) {
                final Double prev = values[key - start];
                values[key - start] = value;

                return prev;
            } else {
                throw new IllegalArgumentException("invalid index for key " + key);
            }
        }
    }
}
