package neuralnerdwork.viz;

import neuralnerdwork.IterationObserver;
import neuralnerdwork.NeuralNetwork;
import neuralnerdwork.TrainingSample;
import neuralnerdwork.math.ConstantVector;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Rectangle2D;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toList;

public class JFrameTrainingVisualizer implements IterationObserver, AutoCloseable {

    private final JFrame frame = new JFrame("Training Visualizer");
    private final JLabel iterationLabel = new JLabel("Not started");
    private final PlotPanel plotPanel = new PlotPanel();
    private final Collection<TrainingSample> validationPoints;
    private final BiFunction<TrainingSample, ConstantVector, Color> predictionTester;

    public JFrameTrainingVisualizer(Collection<TrainingSample> validationPoints,
                                    Rectangle2D viewport,
                                    BiFunction<TrainingSample, ConstantVector, Color> predictionTester) {
        this.validationPoints = validationPoints;
        this.predictionTester = predictionTester;
        plotPanel.setDataViewport(viewport);
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.getContentPane().add(iterationLabel, BorderLayout.NORTH);
        frame.getContentPane().add(plotPanel, BorderLayout.CENTER);
        frame.setLocationByPlatform(true);

        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
        frame.setLocationRelativeTo(null);
        frame.setSize(new Dimension(screenSize.height - 50, screenSize.height - 50));
        frame.setVisible(true);
    }

    public void close() {
        frame.dispose();
    }

    @Override
    public void observe(long iterationCount, NeuralNetwork network) {
        var mather = new SwingWorker<Void,Void>() {

            @Override
            protected Void doInBackground() throws Exception {
                System.out.println("Mathy iteration: " + iterationCount);
                iterationLabel.setText("Iteration " + iterationCount);
                Map<Paint, PointSet> pointSets = validationPoints.stream()
                        .map(sample -> {
                            ConstantVector prediction = new ConstantVector(network.apply(sample.input().values()));
                            return new ClassifiedPoint(predictionTester.apply(sample, prediction),
                                                       sample.input().values());
                        }) // TODO confusion between ConstantVector and double[] here - I always had the wrong one!
                        .collect(Collectors.toMap(
                                ClassifiedPoint::paint,
                                cp -> new PointSet(cp.paint(), List.of(cp.point())),
                                (ps1, ps2) -> new PointSet(ps1.paint(), concat(ps1.points(), ps2.points()))));
        
                plotPanel.updatePointSets(pointSets.values());
                return null;
            }
        };

        mather.execute();
    }

    private static <T> List<T> concat(Collection<? extends T> c1, Collection<? extends T> c2) {
        return Stream.of(c1, c2)
                .flatMap(Collection::stream)
                .collect(toList());
    }

    public void addShape(Shape shape) {
        plotPanel.addShape(shape);
    }

    private record ClassifiedPoint(Paint paint, double[]point) {
    }

}
