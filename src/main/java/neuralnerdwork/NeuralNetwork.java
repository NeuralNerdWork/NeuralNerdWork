package neuralnerdwork;

import neuralnerdwork.math.ConstantVector;
import neuralnerdwork.math.FeedForwardNetwork;
import neuralnerdwork.math.Model;
import neuralnerdwork.math.Vector;

public record NeuralNetwork(FeedForwardNetwork runtimeNetwork, Model.ParameterBindings parameterBindings) {

    public double[] apply(double[] input) {
        var inputVector = new ConstantVector(input);
        return runtimeNetwork.expression(inputVector).evaluate(parameterBindings).toArray();
    }
    
    public Vector apply(Vector input) {
        return new ConstantVector(apply(input.toArray()));
    }

}