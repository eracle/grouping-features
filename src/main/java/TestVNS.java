import java.io.File;
import java.util.List;
import java.util.logging.Logger;
import upo.jcu.math.data.dataset.DataType;
import upo.jml.data.dataset.ClassificationDataset;
import upo.jml.data.dataset.DatasetUtils;
import upo.jml.data.dataset.ToyDatasets;
import upo.jml.prediction.classification.classifiers.BayesClassifier;
import upo.jml.prediction.classification.fss.algorithms.FCBFBagSearch;
import upo.jml.prediction.classification.fss.algorithms.FSPredGroupsBasicVNS;
import upo.jml.prediction.classification.fss.core.FSObjectiveFunction;
import upo.jml.prediction.classification.fss.core.FSSolution;
import upo.jml.prediction.classification.fss.core.FeatureBag;
import upo.jml.prediction.classification.fss.evaluators.CfsEvaluator;
import upo.jml.prediction.classification.fss.evaluators.WrapperEvaluator;

/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
public class TestVNS {

    private static final Logger LOG = Logger.getLogger(TestVNS.class.getName());

    public static void testToyIris() throws Exception {
        int[][] data = ToyDatasets.iris_discrete;
        int[] labels = ToyDatasets.iris_labels;

        FSObjectiveFunction of = new CfsEvaluator(data, labels);
        of.buildEvaluator();
        FSPredGroupsBasicVNS bvns = new FSPredGroupsBasicVNS(data, labels, of);
        FSSolution solution = bvns.search();

        FCBFBagSearch algorithm = new FCBFBagSearch(data, labels, 0.);
        FSSolution solution3 = algorithm.search();
        List<FeatureBag> bags = algorithm.getBags();
        System.out.println("#bags: " + bags.size());
        for (int i = 0; i < bags.size(); i++) {
            System.out.println(bags.get(i));
        }

        System.out.println(solution);
    }

    public static void testArff() throws Exception {
        // Dataset to work on ------------------------------------------------------------------------------
        //String[] sdatasets = new String[]{"colon_tumor", "ionosphere", "glass"};
        //String dpath = "data/" + sdatasets[0] + ".arff";
        //String dpath = "data/glass.arff";
        String dpath = "C:\\Users\\alnouraME\\Desktop\\Summer 2016\\Datasets\\dexter\\dexter_first_half.arff";        
        
        ClassificationDataset dataset = DatasetUtils.loadArffDataset(new File(dpath), -1);
        if (!dataset.getDataType().equals(DataType.CATEGORICAL)) {
            dataset = DatasetUtils.dicretizeViaFayyad(dataset);
        }
        //ClassificationDataset ddataset = com.jscilib.math.data.dataset.DatasetUtils.dicretizeViaFayyad(dataset);
        //logger.info(ddataset.toString());

        //System.out.println("path: " + dpath);
        //String dpath = "C:\\1.arff";
        //System.out.println("path: " +  dataset.getCategoricalValuesIndexes());
        
        
        // FSObjectiveFunction ------------------------------------------------------------------------------
//        FSObjectiveFunction of = new CfsEvaluator(dataset.getCategoricalData(), dataset.getLabels());
//        of.buildEvaluator();
//        FSPredGroupsBasicVNS bvns = new FSPredGroupsBasicVNS(dataset.getCategoricalData(), dataset.getLabels(), of, true);
//        FSSolution solution = bvns.search();
//        System.out.println(solution);
//        System.out.println("------------------------------");
        
        
        // FSObjectiveFunction ------------------------------------------------------------------------------
        FCBFBagSearch algorithm = new FCBFBagSearch(dataset, 0.0);
        //FCBFBagSearch algorithm = new FCBFBagSearch(dataset.getCategoricalData(), dataset.getLabels(), 0.);
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

    public static void main(String[] args) throws Exception {
        //TestVNS.testToyIris();

        TestVNS.testArff();
    }

}
