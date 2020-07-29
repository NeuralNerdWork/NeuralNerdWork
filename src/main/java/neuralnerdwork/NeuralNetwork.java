package neuralnerdwork;

import neuralnerdwork.backprop.FeedForwardNetwork;
import neuralnerdwork.backprop.FullyConnectedLayer;
import neuralnerdwork.backprop.Layer;
import neuralnerdwork.math.*;
import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;

import java.util.Optional;
import java.util.function.Function;

public record NeuralNetwork(FeedForwardNetwork runtimeNetwork, Model.ParameterBindings parameterBindings) {

    public double[] apply(double[] input) {
        var inputVector = new DMatrixRMaj(input.length, 1, true, input);
        DMatrix result = apply(inputVector);
        if (result instanceof DMatrixRMaj m) {
            return m.getData();
        } else {
            double[] values = new double[result.getNumRows() * result.getNumCols()];
            for (int i = 0; i < values.length; i++) {
                values[i] = result.get(i, 0);
            }

            return values;
        }
    }
    
    public DMatrix apply(DMatrix input) {
        return runtimeNetwork.expression(input).evaluate(parameterBindings);
    }

    public static NeuralNetwork fullyConnectedClassificationNetwork(Function<Layer<?>, Double> initialWeightSupplier, int... layerSizes) {
        var modelBuilder = new Model();
        return fullyConnectedClassificationNetwork(initialWeightSupplier, modelBuilder, layerSizes);
    }

    public static NeuralNetwork fullyConnectedClassificationNetwork(Function<Layer<?>, Double> initialWeightSupplier, Model modelBuilder, int... layerSizes) {
        /*
          (inp)               (out)
           l0   l1   l2   l3  l4

           o         o
                o
           o         o
                o         o
           o         o         o
                o         o
           o         o
                o
           o         o

           b    b    b    b    b?

            l1 weights: 4x6 matrix (each row is the 6 weight multipliers of l0 including the bias term)

        */
        // build random-weighted network of the requested shape
        if (layerSizes.length < 2) {
            throw new IllegalArgumentException("layerSizes must be at least 2 (input+output)");
        }
        //start at first hidden layer; end at output layer (TODO: bias on output layer should be optional)
        // build weight matrices to reuse in layers
        var layers = new Layer[layerSizes.length - 1];
        LeakyRelu activation = new LeakyRelu(0.01);
        for (int l = 1; l < layerSizes.length; l++) {
            // columns: input size
            // rows: output size
            ParameterMatrix layerLWeights = modelBuilder.createParameterMatrix(layerSizes[l], layerSizes[l-1]);
            ParameterVector bias = modelBuilder.createParameterVector(layerSizes[l]);
            if(l == layerSizes.length - 1){
                layers[l - 1] = new FullyConnectedLayer(layerLWeights, Optional.ofNullable(bias), new LogisticFunction());
            } else {
                layers[l - 1] = new FullyConnectedLayer(layerLWeights, Optional.ofNullable(bias), activation);
            }
        }

        Model.ParameterBindings initialParameterBindings = initializeParameters(initialWeightSupplier, modelBuilder, layers);
        var feedforwardDefinition = new FeedForwardNetwork(layers);

        return new NeuralNetwork(feedforwardDefinition, initialParameterBindings);
    }

    private static Model.ParameterBindings initializeParameters(Function<Layer<?>, Double> initialWeightSupplier, Model modelBuilder, Layer<?>[] layers) {
        Model.ParameterBindings initialParameterBindings = modelBuilder.createBinder();
        // initialize weights
        for (var layer : layers) {
            layer.variables().forEach(v -> initialParameterBindings.put(v, initialWeightSupplier.apply(layer)));
        }
        return initialParameterBindings;
    }

}
