package neuralnerdwork.backprop;

import neuralnerdwork.math.*;
import org.ejml.data.DMatrix;

import java.util.Optional;
import java.util.stream.IntStream;

import static neuralnerdwork.math.VectorSum.sum;

public record FullyConnectedLayer(ParameterMatrix weights, Optional<ParameterVector> bias, ActivationFunction activation) implements Layer<FullyConnectedLayer.PerceptronCache> {
    public record PerceptronCache(Vector activation, Vector activationInputs, DMatrix activationDerivativeWithRespectToWeightedSum) {
    }

    @Override
    public boolean containsVariable(int variable) {
        return weights.containsVariable(variable) || bias.filter(b -> b.containsVariable(variable)).isPresent();
    }

    @Override
    public int outputLength() {
        return weights.rows();
    }

    @Override
    public int inputLength() {
        return weights.cols();
    }

    @Override
    public IntStream variables() {
        return IntStream.concat(weights.variables(),
                                bias.map(ParameterVector::variables)
                                    .orElseGet(IntStream::empty));
    }

    @Override
    public Result<Vector, PerceptronCache> derivativeWithRespectLayerParameter(Vector layerInput, int variable, PerceptronCache cache, Model.ParameterBindings bindings) {
        SingleVariableFunction activationDerivative = activation.differentiateByInput();

        final VectorExpression weightedSumDerivative;
        if (weights.containsVariable(variable)) {
            int row = weights.rowIndexFor(variable);
            int col = weights.colIndexFor(variable);
            double[] values = new double[weights.rows()];
            values[row] = layerInput.get(col);

            weightedSumDerivative = new ConstantVector(values);
        } else {
            int index = bias.orElseThrow().indexFor(variable);
            double[] values = new double[weights.rows()];
            values[index] = 1.0;

            weightedSumDerivative = new ConstantVector(values);
        }

        Vector activationInputs = getActivationInputs(layerInput, cache, bindings);

        DMatrix activationDerivativeWithRespectWeightedSum =
                getActivationDerivativeWithRespectToWeightedSum(cache, bindings, activationDerivative, activationInputs);

        Vector output = new MatrixVectorProduct(
                new DMatrixExpression(activationDerivativeWithRespectWeightedSum),
                weightedSumDerivative
        ).evaluate(bindings);

        return new Result<>(output, new PerceptronCache(cache.activation(), activationInputs, activationDerivativeWithRespectWeightedSum));
    }

    private Vector getActivationInputs(Vector layerInput, PerceptronCache cache, Model.ParameterBindings bindings) {
        return cache.activationInputs() != null ?
                        cache.activationInputs() :
                        calculateWeightedSums(layerInput, bindings);
    }

    private DMatrix getActivationDerivativeWithRespectToWeightedSum(PerceptronCache cache, Model.ParameterBindings bindings, SingleVariableFunction activationDerivative, Vector activationInputs) {
        return (cache.activationDerivativeWithRespectToWeightedSum() != null) ?
                cache.activationDerivativeWithRespectToWeightedSum() :
                calculateActivationDerivativeWithRespectToWeightedSum(bindings, activationDerivative, activationInputs);
    }

    private DMatrix calculateActivationDerivativeWithRespectToWeightedSum(Model.ParameterBindings bindings, SingleVariableFunction activationDerivative, Vector activationInputs) {
        /*
         If you have vectors x and y, then
           x dot y == D(x) * y
         where `dot` is the vector dot product, `*` is matrix multiplication, and `D` is a function
         that turns a vector into a diagonal matrix.

         Why does this matter? Matrix multiplication is associative, so for complex expressions we have:
           x dot (A * y) == D(x) * (A * y) = (D(x) * A) * y

         We use this in the backpropogation so that we can build up delta terms from left to right.
         */
        return new DiagonalizedVector(
                new VectorizedSingleVariableFunction(
                        activationDerivative,
                        activationInputs
                )
        ).evaluate(bindings);
    }

    @Override
    public Result<DMatrix, PerceptronCache> derivativeWithRespectToLayerInput(Vector layerInput, PerceptronCache cache, Model.ParameterBindings bindings) {
        SingleVariableFunction activationDerivative = activation.differentiateByInput();

        Vector activationInputs = getActivationInputs(layerInput, cache, bindings);
        DMatrix activationDerivativeWithRespectToWeightedSum =
                getActivationDerivativeWithRespectToWeightedSum(cache, bindings, activationDerivative, activationInputs);

        DMatrix output = new MatrixProduct(
                new DMatrixExpression(activationDerivativeWithRespectToWeightedSum),
                weights
        ).evaluate(bindings);

        return new Result<>(output, new PerceptronCache(cache.activation(), activationInputs, activationDerivativeWithRespectToWeightedSum));
    }

    @Override
    public Result<Vector, PerceptronCache> evaluate(Vector layerInput, Model.ParameterBindings bindings) {
        final Vector weightedSums = calculateWeightedSums(layerInput, bindings);

        Vector output = new VectorizedSingleVariableFunction(
                activation,
                weightedSums
        ).evaluate(bindings);

        return new Result<>(output, new PerceptronCache(output, weightedSums, null));
    }

    private Vector calculateWeightedSums(Vector layerInput, Model.ParameterBindings bindings) {
        return bias.map(b ->
                                sum(
                                        new MatrixVectorProduct(
                                                weights,
                                                layerInput
                                        ),
                                        b
                                )
        ).orElseGet(() ->
                            new MatrixVectorProduct(
                                    weights,
                                    layerInput
                            )
        ).evaluate(bindings);
    }

    @Override
    public Vector getEvaluation(PerceptronCache cache) {
        return cache.activation();
    }
}
