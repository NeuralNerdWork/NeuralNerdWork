package neuralnerdwork;

import neuralnerdwork.backprop.ConvolutionLayer;
import neuralnerdwork.backprop.ConvolutionLayer.Convolution;
import neuralnerdwork.backprop.FeedForwardNetwork;
import neuralnerdwork.backprop.Layer;
import neuralnerdwork.backprop.MaxPoolLayer;
import neuralnerdwork.backprop.MaxPoolLayer.Channel;
import neuralnerdwork.descent.AdagradUpdate;
import neuralnerdwork.descent.NesterovMomentumGradientUpdate;
import neuralnerdwork.descent.RmsPropUpdate;
import neuralnerdwork.descent.StochasticGradientDescent;
import neuralnerdwork.math.ConvolutionFilterMatrix;
import neuralnerdwork.math.LeakyRelu;
import neuralnerdwork.math.Model;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static neuralnerdwork.NeuralNetwork.fullyConnectedClassificationNetwork;
import static neuralnerdwork.weight.VariableWeightInitializer.dumbRandomWeightInitializer;
import static neuralnerdwork.weight.VariableWeightInitializer.smartRandomWeightInitializer;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

public class ConvolutionalNetworkTrainingTest {

    static record FailurePercent(int failures, int total) { 
        FailurePercent merge(FailurePercent other) { 
            return new FailurePercent(failures() + other.failures(), total() + other.total());
        }
        
        double asPercent() {
            return ((double) failures) / ((double) total);
        }
        
    }

    @Test
    @Disabled
    void trainingForPointsInsideACircleShouldConverge() throws Exception {

        final int rows = 100;
        final int cols = 100;
        Random r = new Random(11);
        var trainingSet = generatePoints(rows, cols, r);
        var verificationSet = generatePoints(rows, cols, r);

        Model model = new Model();

        Layer<?>[] layers = new Layer[2 + 1];
        LeakyRelu relu = new LeakyRelu(0.01);
        int filterRows = 3;
        int filterCols = 3;
        Convolution[] convolutions = new Convolution[] {
                new Convolution(
                        new ConvolutionFilterMatrix(
                                model.createParameterMatrix(filterRows, filterCols),
                                rows,
                                cols
                        ),
                        model.createScalarParameter()
                ),
                new Convolution(
                        new ConvolutionFilterMatrix(
                                model.createParameterMatrix(filterRows, filterCols),
                                rows,
                                cols
                        ),
                        model.createScalarParameter()
                )
        };
        layers[0] = new ConvolutionLayer(1, convolutions, relu);
        layers[1] = new MaxPoolLayer(
                new Channel[]{
                        new Channel(rows - filterRows + 1, cols - filterCols + 1, 7, 7),
                        new Channel(rows - filterRows + 1, cols - filterCols + 1, 7, 7),
                }
        );
        Layer<?>[] fullyConnectedLayers = fullyConnectedClassificationNetwork(smartRandomWeightInitializer(r), model,
                                                                              layers[1].outputLength(),
                                                                              1)
                .runtimeNetwork()
                .layers();
        System.arraycopy(fullyConnectedLayers, 0, layers, 2, fullyConnectedLayers.length);

        FeedForwardNetwork networkDef = new FeedForwardNetwork(layers);

        Model.ParameterBindings binder = model.createBinder();
        var initializer = dumbRandomWeightInitializer(r);
        layers[0].variables()
                 .forEach(var -> binder.put(var, initializer.apply(layers[0])));
        layers[1].variables()
                 .forEach(var -> binder.put(var, initializer.apply(layers[1])));
        for (int i = 2; i < layers.length; i++) {
            Layer<?> layer = layers[i];
            layer.variables()
                 .forEach(var -> binder.put(var, layer.activation().generateInitialWeight(r, layer)));
        }

        NeuralNetwork untrainedNetwork = new NeuralNetwork(networkDef, binder);
        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(untrainedNetwork, new StochasticGradientDescent(
                50,
                () -> new RmsPropUpdate(0.001, 0.9, 1e-8)
        ), (iterationCount, network) -> {
            System.out.println("Iteration " + iterationCount);
            if (iterationCount > 10000) {
                return false;
            } else if (iterationCount != 0 && iterationCount % 10 == 0) {
                var fails = verificationSet.stream()
                                           .parallel()
                                           .map(i -> {
                                               return Util.compareClassifications(network.apply(i.input())[0], i
                                                       .output()[0]);
                                           })
                                           .map(b -> new FailurePercent(b ? 0 : 1, 1))
                                           .reduce(new FailurePercent(0, 0), FailurePercent::merge);

                System.out.println("Percentage of verification set failing: " + fails.asPercent());

                return fails.asPercent() > 0.05;
            } else {
                return true;
            }
        });

        NeuralNetwork trainedNetwork = trainer.train(trainingSet);

        var failures = verificationSet.stream()
                                   .parallel()
                                   .map(i -> {
                                       return Util.compareClassifications(trainedNetwork.apply(i.input())[0], i
                                               .output()[0]);
                                   })
                                   .map(b -> new FailurePercent(b ? 0 : 1, 1))
                                   .reduce(new FailurePercent(0, 0), FailurePercent::merge);

        assertTrue(failures.asPercent() <= 0.05, () -> String.format("%d/%d incorrect predictions", failures.failures(), failures.total()));
    }

    private List<TrainingSample> generatePoints(int rows, int cols, Random r) {
        return Stream.iterate(1, i -> i < 1000, i -> i + 1)
                     .map(i -> {
                         double x = r.nextDouble() * 2.0 - 1.0;
                         double y = r.nextDouble() * 2.0 - 1.0;
                         boolean inside = Math.sqrt(x * x + y * y) <= 0.75;
                         return new TrainingSample(new double[]{x, y}, new double[]{inside ? 1.0 : 0.0});
                     })
                     .map(sample -> {
                         double[] pixels = new double[rows * cols];
                         double[] input = sample.input();
                         int xToCol = (int) Math.round((input[0] / 2.0 + 0.5) * (cols - 1));
                         int yToRow = (int) Math.round((input[1] / 2.0 + 0.5) * (rows - 1));
                         int pixelIndex = xToCol * cols + yToRow;
                         pixels[pixelIndex] = 1.0;

                         return new TrainingSample(pixels, sample.output());
                     })
                     .collect(Collectors.toList());
    }

}
