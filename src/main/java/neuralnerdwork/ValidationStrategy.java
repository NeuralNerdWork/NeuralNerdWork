package neuralnerdwork;

@FunctionalInterface
public interface ValidationStrategy {
    boolean hasConverged(long iterationCount, NeuralNetwork network);
}