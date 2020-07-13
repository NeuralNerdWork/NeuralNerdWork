package neuralnerdwork;

import neuralnerdwork.descent.GradientDescentStrategy;
import neuralnerdwork.math.*;
import neuralnerdwork.math.Model.ParameterBindings;

import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.Optional;

public class NeuralNetworkTrainer {
    private final int[] layerSizes;
    private final GradientDescentStrategy gradientDescentStrategy;
    private final ValidationStrategy validationStrategy;
    private final IterationObserver iterationObserver;
    private final BiFunction<Integer, Integer, Double> initialWeightSupplier;

    /**
     * @param layerSizes the number of neurons in each layer. Layer 0 is the input layer; layer (layerSizes.length-1) is the output layer.
     * Must have length at least 2.
     * @param gradientDescentStrategy runs a variation of gradient descent given an error expression.
     */
    public NeuralNetworkTrainer(int[] layerSizes, BiFunction<Integer, Integer, Double> initialWeightSupplier, GradientDescentStrategy gradientDescentStrategy, ValidationStrategy validationStrategy) {
        this(layerSizes, initialWeightSupplier, gradientDescentStrategy, validationStrategy, (a, b) -> {});
    }

    /**
     * @param layerSizes the number of neurons in each layer. Layer 0 is the input layer; layer (layerSizes.length-1) is the output layer.
     * Must have length at least 2.
     * @param gradientDescentStrategy runs a variation of gradient descent given an error expression.
     */
    public NeuralNetworkTrainer(int[] layerSizes, BiFunction<Integer, Integer, Double> initialWeightSupplier, GradientDescentStrategy gradientDescentStrategy, ValidationStrategy validationStrategy, IterationObserver iterationObserver) {
        if (layerSizes.length < 2) {
            throw new IllegalArgumentException("layerSizes must be at least 2 (input+output)");
        }
        this.layerSizes = layerSizes;
        this.gradientDescentStrategy = gradientDescentStrategy;
        this.validationStrategy = validationStrategy;
        this.iterationObserver = iterationObserver;
        this.initialWeightSupplier = initialWeightSupplier;
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
        var layers = new FeedForwardNetwork.Layer[layerSizes.length - 1];
        LeakyRelu activation = new LeakyRelu(0.01);
        for (int l = 1; l < layerSizes.length; l++) {
            // columns: input size
            // rows: output size
            ParameterMatrix layerLWeights = modelBuilder.createParameterMatrix(layerSizes[l], layerSizes[l-1]);
            ParameterVector bias = l < layerSizes.length - 1 ? modelBuilder.createParameterVector(layerSizes[l]) : null;
            layers[l - 1] = new FeedForwardNetwork.Layer(layerLWeights, Optional.ofNullable(bias), activation);
        }

        // training cycle end
        // TODO - Stop when we have converged

        // Note from @ball - normally the "convergence test" 
        // is to keep a small subset of the data as a test 
        // set and check the error function against that 
        // test set every n iterations

        ParameterBindings initialParameterBindings = modelBuilder.createBinder();
        // initialize weights
        for (var layer : layers) {
            var m = layer.weights();
            m.variables().forEach(v -> initialParameterBindings.put(v, initialWeightSupplier.apply(m.cols(), m.rows())));
        }
        var feedforwardDefinition = new FeedForwardNetwork(layers);

        Model.ParameterBindings parameterBindings = gradientDescentStrategy.runGradientDescent(
                samples,
                initialParameterBindings,
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

                        VectorExpression network = feedforwardDefinition.expression(inputLayer);

                        // find (squared) error amount
                        squaredErrors[i] = squaredError(sample, network);
                    }

                    // this is a function that hasn't been evaluated yet
                    return new ScalarConstantMultiple(1.0 / (double) ts.size(), ScalarSum.sum(squaredErrors));
                },
                (iterationCount, lastUpdateVector, currentParameters) -> {
                    NeuralNetwork network = new NeuralNetwork(feedforwardDefinition, currentParameters);
                    iterationObserver.observe(iterationCount, network);
                    return validationStrategy.hasConverged(iterationCount, network);
                });

        return new NeuralNetwork(feedforwardDefinition, parameterBindings);
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
