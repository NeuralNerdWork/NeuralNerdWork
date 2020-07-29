package neuralnerdwork.backprop;

import neuralnerdwork.math.*;
import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixSparseCSC;

import java.util.Optional;
import java.util.stream.IntStream;

import static neuralnerdwork.math.VectorSum.sum;

public record FullyConnectedLayer(ParameterMatrix weights, Optional<ParameterVector> bias, ActivationFunction activation) implements Layer<FullyConnectedLayer.PerceptronCache> {
    public record PerceptronCache(DMatrix activation, DMatrix activationInputs, DMatrix activationDerivativeWithRespectToWeightedSum) {
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
    public Result<DMatrix, PerceptronCache> derivativeWithRespectLayerParameter(DMatrix layerInput, int variable, PerceptronCache cache, Model.ParameterBindings bindings) {
        SingleVariableFunction activationDerivative = activation.differentiateByInput();

        final VectorExpression weightedSumDerivative;
        if (weights.containsVariable(variable)) {
            int row = weights.rowIndexFor(variable);
            int col = weights.colIndexFor(variable);

            DMatrixSparseCSC matrix = new DMatrixSparseCSC(weights.rows(), 1, 1);
            matrix.set(row, 0, layerInput.get(col, 0));
            weightedSumDerivative = new DMatrixColumnVectorExpression(matrix);
        } else {
            int index = bias.orElseThrow().indexFor(variable);

            DMatrixSparseCSC matrix = new DMatrixSparseCSC(weights.rows(), 1, 1);
            matrix.set(index, 0, 1.0);
            weightedSumDerivative = new DMatrixColumnVectorExpression(matrix);
        }

        DMatrix activationInputs = getActivationInputs(layerInput, cache, bindings);

        DMatrix activationDerivativeWithRespectWeightedSum =
                getActivationDerivativeWithRespectToWeightedSum(cache, bindings, activationDerivative, activationInputs);

        DMatrix output = new MatrixVectorProduct(
                new DMatrixExpression(activationDerivativeWithRespectWeightedSum),
                weightedSumDerivative
        ).evaluate(bindings);

        return new Result<>(output, new PerceptronCache(cache.activation(), activationInputs, activationDerivativeWithRespectWeightedSum));
    }

    private DMatrix getActivationInputs(DMatrix layerInput, PerceptronCache cache, Model.ParameterBindings bindings) {
        return cache.activationInputs() != null ?
                        cache.activationInputs() :
                        calculateWeightedSums(layerInput, bindings);
    }

    private DMatrix getActivationDerivativeWithRespectToWeightedSum(PerceptronCache cache, Model.ParameterBindings bindings, SingleVariableFunction activationDerivative, DMatrix activationInputs) {
        return (cache.activationDerivativeWithRespectToWeightedSum() != null) ?
                cache.activationDerivativeWithRespectToWeightedSum() :
                calculateActivationDerivativeWithRespectToWeightedSum(bindings, activationDerivative, activationInputs);
    }

    private DMatrix calculateActivationDerivativeWithRespectToWeightedSum(Model.ParameterBindings bindings, SingleVariableFunction activationDerivative, DMatrix activationInputs) {
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
                new ColumnVectorizedSingleVariableFunction(
                        activationDerivative,
                        new DMatrixColumnVectorExpression(activationInputs)
                )
        ).evaluate(bindings);
    }

    @Override
    public Result<DMatrix, PerceptronCache> derivativeWithRespectToLayerInput(DMatrix layerInput, PerceptronCache cache, Model.ParameterBindings bindings) {
        SingleVariableFunction activationDerivative = activation.differentiateByInput();

        DMatrix activationInputs = getActivationInputs(layerInput, cache, bindings);
        DMatrix activationDerivativeWithRespectToWeightedSum =
                getActivationDerivativeWithRespectToWeightedSum(cache, bindings, activationDerivative, activationInputs);

        DMatrix output = new MatrixProduct(
                new DMatrixExpression(activationDerivativeWithRespectToWeightedSum),
                weights
        ).evaluate(bindings);

        return new Result<>(output, new PerceptronCache(cache.activation(), activationInputs, activationDerivativeWithRespectToWeightedSum));
    }

    @Override
    public Result<DMatrix, PerceptronCache> evaluate(DMatrix layerInput, Model.ParameterBindings bindings) {
        final DMatrix weightedSums = calculateWeightedSums(layerInput, bindings);

        DMatrix output = new ColumnVectorizedSingleVariableFunction(
                activation,
                new DMatrixColumnVectorExpression(weightedSums)
        ).evaluate(bindings);

        return new Result<>(output, new PerceptronCache(output, weightedSums, null));
    }

    private DMatrix calculateWeightedSums(DMatrix layerInput, Model.ParameterBindings bindings) {
        return bias.map(b ->
                                sum(
                                        new MatrixVectorProduct(
                                                weights,
                                                new DMatrixColumnVectorExpression(layerInput)
                                        ),
                                        b
                                )
        ).orElseGet(() ->
                            new MatrixVectorProduct(
                                    weights,
                                    new DMatrixColumnVectorExpression(layerInput)
                            )
        ).evaluate(bindings);
    }

    @Override
    public DMatrix getEvaluation(PerceptronCache cache) {
        return cache.activation();
    }
}
