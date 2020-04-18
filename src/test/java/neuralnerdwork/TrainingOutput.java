package neuralnerdwork;

import net.jqwik.api.constraints.DoubleRange;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

@Retention(RetentionPolicy.RUNTIME)
@DoubleRange(min = -1, max = 1)
public @interface TrainingOutput {
}
