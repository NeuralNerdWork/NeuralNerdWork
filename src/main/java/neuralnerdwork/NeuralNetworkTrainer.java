package neuralnerdwork;

import neuralnerdwork.descent.GradientDescentStrategy;
import neuralnerdwork.math.*;
import neuralnerdwork.math.Model.ParameterBindings;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import java.util.List;

public class NeuralNetworkTrainer {
    private final GradientDescentStrategy gradientDescentStrategy;
    private final ValidationStrategy validationStrategy;
    private final IterationObserver iterationObserver;
    private final NeuralNetwork network;

    public NeuralNetworkTrainer(NeuralNetwork network, GradientDescentStrategy gradientDescentStrategy, ValidationStrategy validationStrategy) {
        this(network, gradientDescentStrategy, validationStrategy, (a, b) -> {});
    }

    public NeuralNetworkTrainer(NeuralNetwork network, GradientDescentStrategy gradientDescentStrategy, ValidationStrategy validationStrategy, IterationObserver iterationObserver) {
        this.network = network;
        this.gradientDescentStrategy = gradientDescentStrategy;
        this.validationStrategy = validationStrategy;
        this.iterationObserver = iterationObserver;
    }

    NeuralNetwork train(List<TrainingSample> samples) {
        ParameterBindings initialParameterBindings = network.parameterBindings().copy();
        var feedforwardDefinition = network.runtimeNetwork();

        Model.ParameterBindings parameterBindings = gradientDescentStrategy.runGradientDescent(
                samples,
                initialParameterBindings,
                ts -> {
                    final ScalarExpression[] squaredErrors = new ScalarExpression[ts.size()];
                    for (int i = 0; i < ts.size(); i++) {
                        var sample = ts.get(i);
                        var inputLayer = sample.input();
                        if (inputLayer.length != feedforwardDefinition.inputLength()) {
                            throw new IllegalArgumentException("Sample " + i + " has wrong size (got " + sample.input().length + "; expected " + feedforwardDefinition.inputLength() + ")");
                        }
                        if (sample.output().length != feedforwardDefinition.outputLength()) {
                            throw new IllegalArgumentException("Sample " + i + " has wrong size (got " + sample.output().length + "; expected " + feedforwardDefinition.outputLength() + ")");
                        }

                        VectorExpression network = feedforwardDefinition.expression(new DMatrixRMaj(inputLayer.length, 1, true, inputLayer));

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
        final VectorExpression inputError = VectorSum.sum(
                network,
                new ScaledVector(-1.0,
                                 new DMatrixColumnVectorExpression(new DMatrixRMaj(sample.output().length, 1, true, sample
                                         .output())))
        );

        DMatrixRMaj ones = new DMatrixRMaj(sample.output().length, 1);
        CommonOps_DDRM.fill(ones, 1.0);
        return new DotProduct(new DMatrixColumnVectorExpression(ones),
                new ColumnVectorizedSingleVariableFunction(new SquaredSingleVariableFunction(), inputError));
    }
}
