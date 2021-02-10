# gender classification

package ai.certifai.training.classification.transferlearning;

import ai.certifai.training.classification.GenderIterator;
import org.datavec.image.transform.*;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class GenderClassification {

    private static double lr = 1e-3;
    private static int nEpochs = 10;
    private static int batchSize =32;
    private static int seed = 123;
    private static int numClasses = 2;;
    private static int nChannels = 3;
    private static int nOutput = 2;
    private static int height = 224;
    private static int width = 224;
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(EditLastLayerOthersFrozen.class);


    public static void main(String[] args) throws IOException {

        File myFile = new ClassPathResource("GenderClassification").getFile();

        FlipImageTransform hFlip = new FlipImageTransform(0);
        ImageTransform rCrop = new RandomCropTransform(seed,50,50);
        ImageTransform rotate = new RotateImageTransform(5);

        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
                new Pair<>(hFlip,0.3),
                new Pair<>(rCrop,0.3),
                new Pair<>(rotate,0.1)
        );

        PipelineImageTransform transform = new PipelineImageTransform(pipeline,false);

        GenderIterator genderIterator = new GenderIterator();

        genderIterator.setup(myFile,nChannels,numClasses,transform,batchSize,0.7);

        DataSetIterator trainIter = genderIterator.trainIterator();
        DataSetIterator testIter = genderIterator.testIterator();

        ZooModel zoo = VGG16.builder().build();
        ComputationGraph vgg16 = (ComputationGraph) zoo.initPretrained();

        System.out.println(vgg16.summary());

        FineTuneConfiguration fTConf  = new FineTuneConfiguration.Builder()
                .seed(seed)
                .activation(Activation.LEAKYRELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(lr))
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .build();

        ComputationGraph vgg16model = new TransferLearning.GraphBuilder(vgg16)
                .fineTuneConfiguration(fTConf)
                .setFeatureExtractor("fc2")
                .nOutReplace("fc2",1024,WeightInit.XAVIER)
                .addLayer("output", new OutputLayer.Builder()
                        .nIn(1024)
                        .nOut(120)
                        .activation(Activation.LEAKYRELU)
                        .build(), "fc2")
                .removeVertexKeepConnections("predictions")
                .addLayer("output", new OutputLayer.Builder()
                        .nIn(120)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build(), "fc3")
                .setOutputs("output")
                .build();

        System.out.println(vgg16model.summary());

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        vgg16model.setListeners(
                new StatsListener( statsStorage),
                new ScoreIterationListener(5),
                new EvaluativeListener(trainIter,1, InvocationType.EPOCH_END),
                new EvaluativeListener(testIter,1, InvocationType.EPOCH_END)
        );

        vgg16model.fit(trainIter, nEpochs);

        Evaluation evalTrain = vgg16model.evaluate(trainIter);
        Evaluation evalTest = vgg16model.evaluate(testIter);

        System.out.println("Train Evaluation:\n" + evalTrain.stats());
        System.out.println("Test Evaluation:\n" + evalTest);

    }
}
