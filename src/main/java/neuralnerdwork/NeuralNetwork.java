package neuralnerdwork;

@FunctionalInterface
public interface NeuralNetwork {

    double[] apply(double[] input);

}