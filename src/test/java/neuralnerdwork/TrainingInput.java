package neuralnerdwork;

import net.jqwik.api.constraints.DoubleRange;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

@Retention(RetentionPolicy.RUNTIME)
@DoubleRange(min = -10, max = 10)
public @interface TrainingInput {
}
