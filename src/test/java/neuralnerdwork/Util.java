package neuralnerdwork;

import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.function.Supplier;

public class Util {
    static boolean compareClassifications(double a, double b) {
        return Math.signum(a - 0.5) == Math.signum(b - 0.5);
    }

    static <T> T logTiming(String actionName, Supplier<T> action) {
        final Instant start = Instant.now();
        final T retVal = action.get();
        final Duration duration = Duration.between(start, Instant.now());
        System.out.printf("%s function in %d ms\n", actionName, duration.toMillis());
        return retVal;
    }

    static double logisticDerivative(double x) {
        return logistic(x) * logistic(-x);
    }

    static double logistic(double x) {
        double exp = Math.exp(-x);
        if (Double.isInfinite(exp)) {
            return 0;
        } else {
            return 1 / (1 + exp);
        }
    }

    static void logFunctionStructure(Object expression) {
        logTiming("Printed derivative", () -> {
            try {
                final File tmpFile = File.createTempFile("derivative", ".json");
                System.out.println("Writing to tmp file " + tmpFile.getAbsolutePath());
                new ObjectMapper()
                        .setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.NONE)
                        .setVisibility(PropertyAccessor.GETTER, JsonAutoDetect.Visibility.ANY)
                        .setVisibility(PropertyAccessor.IS_GETTER, JsonAutoDetect.Visibility.ANY)
                        .setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY)
                        .writeValue(tmpFile, expression);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            return null;
        });
    }
}
