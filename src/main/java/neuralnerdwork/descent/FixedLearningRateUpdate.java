package neuralnerdwork.descent;

import neuralnerdwork.math.Vector;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public record FixedLearningRateUpdate(double learningRate) implements WeightUpdateStrategy {
    @Override
    public Vector updateVector(Vector rawGradient) {
        return new Vector() {
            @Override
            public double get(int index) {
                return -learningRate * rawGradient.get(index);
            }

            @Override
            public int length() {
                return rawGradient.length();
            }

            @Override
            public String toString() {
                return "Vector["
                       + IntStream.iterate(0, i -> i++)
                                  .limit(length())
                                  .mapToDouble(this::get)
                                  .mapToObj(Double::toString)
                                  .collect(Collectors.joining(","))
                       + "]";
            }
        };
    }
}
