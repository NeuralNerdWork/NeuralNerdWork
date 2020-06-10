package neuralnerdwork;

@FunctionalInterface
public interface IterationObserver {
    void observe(long iterationCount, NeuralNetwork network);
}