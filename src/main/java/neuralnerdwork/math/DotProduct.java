package neuralnerdwork.math;

import com.sun.security.auth.UnixNumericGroupPrincipal;

import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public record DotProduct(VectorFunction left, VectorFunction right) implements ScalarFunction {

    @Override
    public double apply(VectorVariableBinding input) {
        final Vector lVector = left.apply(input);
        final Vector rVector = right.apply(input);

        double accum = 0.0;
        // TODO check matching length and throw with message
        for (int i = 0; i < Math.max(lVector.length(), rVector.length()); i++) {
            accum += lVector.get(i) * rVector.get(i);
        }

        return accum;
    }

    @Override
    public ScalarFunction differentiate(ScalarVariable variable) {
        final VectorFunction leftDerivative = left.differentiate(variable);
        final VectorFunction rightDerivative = right.differentiate(variable);

        return new ScalarSumFunction(new DotProduct(leftDerivative, right), new DotProduct(left, rightDerivative));
    }

    @Override
    public VectorFunction differentiate(VectorVariable variable) {
        throw new UnsupportedOperationException("Not yet implemented!");
    }

    @Override
    public Set<ScalarVariable> variables() {
        return Stream.concat(left.variables()
                                 .stream(),
                             right.variables()
                                  .stream())
                     .collect(Collectors.toSet());
    }
}
