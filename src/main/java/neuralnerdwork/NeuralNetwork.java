package neuralnerdwork;

@FunctionalInterface
public interface NeuralNetwork {

    public double[] apply(double[] input);
    
}