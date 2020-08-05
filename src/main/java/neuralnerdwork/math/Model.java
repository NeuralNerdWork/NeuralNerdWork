package neuralnerdwork.math;

import java.util.Arrays;
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

    public ParameterBindings createBinder() {
        return new ParameterBindings(0, nextParameterIndex);
    }

    public static class ParameterBindings {
        private final int start;
        private final double[] values;
        ParameterBindings(int start, int length) {
            this.start = start;
            values = new double[length];
        }
        private ParameterBindings(int start, double[] values) {
            this.start = start;
            this.values = values;
        }

        public int size() {
            return values.length;
        }

        public Iterable<Integer> variables() {
            return () -> Stream.iterate(start, n -> n + 1)
                               .limit(values.length)
                               .mapToInt(n -> n)
                               .iterator();
        }

        public double get(int key) {
            return values[key - start];
        }

        public double put(int key, Double value) {
            if (key >= start && key < start + values.length) {
                final Double prev = values[key - start];
                values[key - start] = value;

                return prev;
            } else {
                throw new IllegalArgumentException("invalid index for key " + key);
            }
        }

        public ParameterBindings copy() {
            return new ParameterBindings(start, Arrays.copyOf(values, values.length));
        }
    }
}
