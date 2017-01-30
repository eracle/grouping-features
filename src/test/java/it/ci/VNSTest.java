package it.ci;

import org.junit.Test;
import upo.jcu.math.data.dataset.DataType;
import upo.jml.data.dataset.ClassificationDataset;
import upo.jml.data.dataset.DatasetUtils;
import upo.jml.data.dataset.ToyDatasets;
import upo.jml.prediction.classification.fss.algorithms.FCBFBagSearch;
import upo.jml.prediction.classification.fss.core.FSSolution;
import upo.jml.prediction.classification.fss.core.FeatureBag;

import java.io.File;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Created by eracle on 30/01/17.
 */
public class VNSTest {

    @Test
    public void testToyIris() throws Exception {
        int[][] data = ToyDatasets.iris_discrete;
        int[] labels = ToyDatasets.iris_labels;

        FCBFBagSearch algorithm = VNS.testToyIris(data, labels);

        FSSolution solution3 = algorithm.search();
        List<FeatureBag> bags = algorithm.getBags();
        System.out.println("#bags: " + bags.size());
        for (int i = 0; i < bags.size(); i++) {
            System.out.println(bags.get(i));
        }

        System.out.println(solution3);
    }

    @Test
    public void testFromArff() throws Exception {
        // Dataset to work on ------------------------------------------------------------------------------
        //String[] sdatasets = new String[]{"colon_tumor", "ionosphere", "glass"};
        //String dpath = "data/" + sdatasets[0] + ".arff";
        //String dpath = "data/glass.arff";

        //String dpath = "C:\\Users\\alnouraME\\Desktop\\Summer 2016\\Datasets\\dexter\\dexter_first_half.arff";

        //ClassificationDataset ddataset = com.jscilib.math.data.dataset.DatasetUtils.dicretizeViaFayyad(dataset);
        //logger.info(ddataset.toString());

        //System.out.println("path: " + dpath);
        //String dpath = "C:\\1.arff";
        //System.out.println("path: " +  dataset.getCategoricalValuesIndexes());




        ClassLoader classLoader = getClass().getClassLoader();
        File file = new File(classLoader.getResource("dexter_first_half.arff").getFile());

        ClassificationDataset dataset = DatasetUtils.loadArffDataset(file, -1);
        if (!dataset.getDataType().equals(DataType.CATEGORICAL)) {
            dataset = DatasetUtils.dicretizeViaFayyad(dataset);
        }
        FCBFBagSearch algorithm = VNS.FromArff(dataset);

        FSSolution solution3 = algorithm.search();

        List<FeatureBag> bags = algorithm.getBags();
        System.out.println("#bags: " + bags.size());
        for (int i = 0; i < bags.size(); i++) {
            System.out.println(bags.get(i));
        }

        System.out.println(solution3);
        //System.out.println("Total attributes: " + solution3.features().length + " | Max K: " + bvns.getKMax() + " | Iteration: " + bvns.getNumberOfIterations());

        //logger.info("best solution found: " + solution);

        //====================================================================================
        // NN: Online PGVNS
//        System.out.println("/n/n NN TEST: get only one bag: "
//        + algorithm.getThreshold() +
//                algorithm.getBags() );


        //====================================================================================
    }

}