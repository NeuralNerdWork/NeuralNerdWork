package neuralnerdwork;

import java.beans.ParameterDescriptor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

import neuralnerdwork.math.*;

public class NeuralNetworkTrainer {
    private final int[] layerSizes;
    private final double trainingRate;
    private final Supplier<Double> initialWeightSupplier;

    /**
     * @param layerSizes the number of neurons in each layer. Layer 0 is the input layer; layer (layerSizes.length-1) is the output layer.
     * Must have length at least 2.
     */
    public NeuralNetworkTrainer(int[] layerSizes, double trainingRate, Supplier<Double> initialWeightSupplier) {
        if (layerSizes.length < 2) {
            throw new IllegalArgumentException("layerSizes must be at least 2 (input+output)");
        }
        this.layerSizes = layerSizes;
        this.trainingRate = trainingRate;
        this.initialWeightSupplier = initialWeightSupplier;
    }

    /*
            final ConstantMatrix identity = new ConstantMatrix(new double[][]{
                {1.0, 0.0},
                {0.0, 1.0}
        });
    */
    
    Object train(List<TrainingSample> samples) {
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

        ScalarExpression sumOfSquaredError = new ScalarConstant(0.0);
        for (int i = 0; i < samples.size(); i++) {
            var sample = samples.get(i);
            var inputLayer = sample.input();
            if (inputLayer.length() != layerSizes[0]) {
                throw new IllegalArgumentException("Sample " + i + " has wrong size (got " + sample.input().length() + "; expected " + inputLayer.length() + ")");
            }
            if (sample.output().length() != layerSizes[layerSizes.length-1]) {
                throw new IllegalArgumentException("Sample " + i + " has wrong size (got " + sample.output().length() + "; expected " + layerSizes[layerSizes.length-1] + ")");
            }
            
            //start at first hidden layer; end at output layer (TODO: bias on output layer should be optional)
            VectorExpression network = inputLayer;
            var logistic = new LogisticFunction();
            var biasComponent = new ConstantVector(new double[] {1.0});
            for (int l = 1; l < layerSizes.length; l++) {
                // columns: input size including bias
                // rows: output size
                var weightMatrix = weightMatrices.get(l);
                network = new VectorConcat(
                        new VectorizedSingleVariableFunction(
                                logistic,
                                new MatrixVectorProduct(
                                        weightMatrix,
                                        network
                                )
                        ),
                        biasComponent
                );
            }
            
            // find (squared) error amount
            final ScalarExpression squaredError = squaredError(sample, network);
            sumOfSquaredError = ScalarSum.sum(sumOfSquaredError, squaredError);
        }

        // this is a function that hasn't been evaluated yet
        ScalarExpression meanSquaredError = new ScalarConstantMultiple(samples.size(), sumOfSquaredError);

        // use derivative to adjust weights
        var binder = modelBuilder.createBinder();
        VectorExpression lossDerivative = meanSquaredError.computeDerivative(binder.variables());
        // initialize weights
        for (int w = 0; w < binder.variables().length; w++) {
            binder.put(w, initialWeightSupplier.get());
        }

        // Repeat this until converged
        // TODO - Update neuron weights based on gradient and learning parameter
        var weightUpdateVector = lossDerivative.evaluate(binder);
        for (int w = 0; w < binder.variables().length; w++) {
            binder.put(w, binder.get(w) - trainingRate * weightUpdateVector.get(w));
        }
        // training cycle end

        // TODO - Stop when we have converged
        return null;
    }

    private static ScalarExpression squaredError(TrainingSample sample, VectorExpression network) {
        // difference between network output and expected output
        final VectorSum inputError = new VectorSum(network, new ScaledVector(-1.0, sample.output()));

        final double[] ones = new double[sample.output().length()];
        Arrays.fill(ones, 1.0);
        return new DotProduct(new ConstantVector(ones),
                new VectorizedSingleVariableFunction(new SquaredSingleVariableFunction(), inputError));
    }


}
