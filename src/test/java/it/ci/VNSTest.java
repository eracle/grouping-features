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


}