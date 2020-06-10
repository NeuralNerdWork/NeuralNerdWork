package neuralnerdwork;

import neuralnerdwork.descent.GradientDescentStrategy;
import neuralnerdwork.math.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NeuralNetworkTrainer {
    private final int[] layerSizes;
    private final GradientDescentStrategy gradientDescentStrategy;

    /**
     * @param layerSizes the number of neurons in each layer. Layer 0 is the input layer; layer (layerSizes.length-1) is the output layer.
     * Must have length at least 2.
     * @param gradientDescentStrategy runs a variation of gradient descent given an error expression.
     */
    public NeuralNetworkTrainer(int[] layerSizes, GradientDescentStrategy gradientDescentStrategy) {
        if (layerSizes.length < 2) {
            throw new IllegalArgumentException("layerSizes must be at least 2 (input+output)");
        }
        this.layerSizes = layerSizes;
        this.gradientDescentStrategy = gradientDescentStrategy;
    }

    /*
            final ConstantMatrix identity = new ConstantMatrix(new double[][]{
                {1.0, 0.0},
                {0.0, 1.0}
        });
    */
    
    NeuralNetwork train(List<TrainingSample> samples) {
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

        var modelBuilder = new Model();
        //start at first hidden layer; end at output layer (TODO: bias on output layer should be optional)
        // build weight matrices to reuse in layers
        var weightMatrices = new ArrayList<ParameterMatrix>();
        for (int l = 1; l < layerSizes.length; l++) {
            // columns: input size including bias
            // rows: output size
            ParameterMatrix layerLWeights = modelBuilder.createParameterMatrix(layerSizes[l], layerSizes[l-1] + 1);
            weightMatrices.add(layerLWeights);
        }

        // training cycle end
        // TODO - Stop when we have converged

        // Note from @ball - normally the "convergence test" 
        // is to keep a small subset of the data as a test 
        // set and check the error function against that 
        // test set every n iterations
        Model.ParameterBindings parameterBindings = gradientDescentStrategy.runGradientDescent(
                samples,
                modelBuilder.createBinder(),
                ts -> {
                    final ScalarExpression[] squaredErrors = new ScalarExpression[ts.size()];
                    for (int i = 0; i < ts.size(); i++) {
                        var sample = ts.get(i);
                        var inputLayer = sample.input();
                        if (inputLayer.length() != layerSizes[0]) {
                            throw new IllegalArgumentException("Sample " + i + " has wrong size (got " + sample.input().length() + "; expected " + inputLayer.length() + ")");
                        }
                        if (sample.output().length() != layerSizes[layerSizes.length - 1]) {
                            throw new IllegalArgumentException("Sample " + i + " has wrong size (got " + sample.output().length() + "; expected " + layerSizes[layerSizes.length - 1] + ")");
                        }

                        VectorExpression network = buildNetwork(weightMatrices, inputLayer);

                        // find (squared) error amount
                        squaredErrors[i] = squaredError(sample, network);
                    }

                    // this is a function that hasn't been evaluated yet
                    return new ScalarConstantMultiple(1.0 / (double) ts.size(), ScalarSum.sum(squaredErrors));
                });

        return input -> {
            var inputVector = new ConstantVector(input);
            var runtimeNetwork = buildNetwork(weightMatrices, inputVector);

            return runtimeNetwork.evaluate(parameterBindings).toArray();
        };
    }

    private VectorExpression buildNetwork(ArrayList<ParameterMatrix> weightMatrices, ConstantVector inputLayer) {
        //start at first hidden layer; end at output layer (TODO: bias on output layer should be optional)
        var logistic = new LogisticFunction();
        var relu = new ReluFunction();
        final FeedForwardNetwork.Layer[] layers = new FeedForwardNetwork.Layer[layerSizes.length - 1];
        for (int l = 1; l < layerSizes.length; l++) try {

            // columns: input size including bias
            // rows: output size
            var weightMatrix = weightMatrices.get(l - 1);
            // TODO: move out of loop
            var activation = (l == layerSizes.length - 1) ? logistic : relu;
            layers[l-1] = new FeedForwardNetwork.Layer(weightMatrix, activation);
        } catch(Exception e) {
            throw new RuntimeException("Exception building layer " + l, e);
        }
        
        return new FeedForwardNetwork(inputLayer, layers);
    }

    private static ScalarExpression squaredError(TrainingSample sample, VectorExpression network) {
        // difference between network output and expected output
        final VectorExpression inputError = VectorSum.sum(network, new ScaledVector(-1.0, sample.output()));

        final double[] ones = new double[sample.output().length()];
        Arrays.fill(ones, 1.0);
        return new DotProduct(new ConstantVector(ones),
                new VectorizedSingleVariableFunction(new SquaredSingleVariableFunction(), inputError));
    }


}
