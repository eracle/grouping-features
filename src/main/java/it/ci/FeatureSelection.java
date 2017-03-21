package it.ci;

import upo.jcu.math.data.dataset.DataType;
import upo.jml.data.dataset.DatasetUtils;
import upo.jml.prediction.classification.fss.algorithms.FCBFBagSearch;
import upo.jml.prediction.classification.fss.core.FSSolution;
import upo.jml.prediction.classification.fss.core.FeatureBag;
import weka.core.Attribute;

import weka.core.Instances;

import weka.core.converters.ArffLoader.ArffReader;

import upo.jml.data.dataset.ClassificationDataset;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RandomSubset;
import weka.filters.unsupervised.attribute.Remove;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;




import java.util.logging.Logger;


/**
 *
 * Allows to open an arff file and create 6 dataset by
 * dividing the attributes to 50%, 10%, 10%, 10%, 10%, 10%.
 * It uses a WEKA RandomSubset object and convert them to
 * upo.jml.data.dataset.ClassificationDataset instances.
 * Created by eracle on 30/01/17.
 */
public class FeatureSelection {

    private final static Logger logger = Logger.getLogger(FeatureSelection.class.getName());


    public static Instances openArff(File file) throws IOException {
        logger.info("Open file: "+ file.getAbsolutePath());
        BufferedReader reader = new BufferedReader(new FileReader(file));
        ArffReader arff = new ArffReader(reader);
        Instances data = arff.getData();
        return data;

    }

    /**
     * Shortcut for RandomSubset object.
     * @param data
     * @param percent
     * @return
     */
    @Deprecated
    private static Instances RandomSubsetWrapper(Instances data, double percent)throws Exception{
        RandomSubset subset = new RandomSubset();

        String[] options = new String[2];
        options[0] = "-N";
        options[1] = ""+percent;
        subset.setOptions(options);

        subset.setInputFormat(data);
        Instances newData = Filter.useFilter(data, subset);
        return newData;
    }

    /**
     * Return a dataset (WEKA Instances object) with only the attributes kept, all the others are deleted.
     *
     * @param attributes
     * @param data
     * @return
     */
    protected static Instances keepAttributes(ArrayList<Attribute> attributes, Instances data)  throws Exception{

        ArrayList<Attribute> data_attributes =  Collections.list(data.enumerateAttributes());
        StringBuffer indexes_list = new StringBuffer();
        boolean first = true;
        for(Attribute to_remove : attributes){
            // logger.info("Removing attribute: "+ to_remove.name()+" from the rest of the data");
            int index_to_remove = data_attributes.indexOf(to_remove)+1;
            if(first){
                indexes_list.append(""+index_to_remove);
                first = false;
            }else {
                indexes_list.append("," + index_to_remove);
            }
        }
        Remove remove = new Remove();
        String indexes_str = indexes_list.toString();
        logger.info(indexes_str);

        remove.setAttributeIndices(indexes_str);

        remove.setInvertSelection(true);
        remove.setInputFormat(data);
        return Filter.useFilter(data, remove);
    }

    private static double[] split_percentages_50_10 = {.5, .1, .1, .1, .1, .1};

    public static ArrayList<Instances> splitFeatures(Instances data)  throws Exception{
        return FeatureSelection.splitFeatures(data, FeatureSelection.split_percentages_50_10);
    }

    /**
     * Static method which splits the data (WEKA instances) passed as an argument by
     * following the percentages contained on the split_percentages array.
     * The splits are made by attributes (fields), are chosen randomly, and their intersection is null.
     * During this process, the class attribute is not chosen, and is included on every returned Istances dataset.
     * Is assumed that the last attribute of the data object (WEKA instances) is its class attribute.
     * The sum of the double values contained in the split_percentages must sum up to 1.
     * Is returned an ArrayList containing those WEKA instances objects.
     * @param data
     * @param split_percentages
     * @return
     * @throws Exception
     */
    public static ArrayList<Instances> splitFeatures(Instances data, double[] split_percentages)  throws Exception{
        ArrayList<Attribute> attributes = Collections.list(data.enumerateAttributes());

        Attribute class_att = attributes.remove(attributes.size()-1);

        int num_attributes = attributes.size();

        logger.info("Class attribute:" + class_att);
        Collections.shuffle(attributes);

        ArrayList<ArrayList> sub_sets = new ArrayList<ArrayList>();

        for(int k=0; k < split_percentages.length; k++){
            int num_atts_to_take = (int)(split_percentages[k] * num_attributes);
            logger.info("Number of attributes to take:" + num_atts_to_take);
            ArrayList<Attribute> subset = new ArrayList<Attribute>();
            for(int i = 0; i < num_atts_to_take; i++){
                subset.add(attributes.remove(attributes.size()-1));
            }
            subset.add(class_att);
            sub_sets.add(subset);
            logger.info(subset.toString());
        }

        ArrayList<Instances> returns = new ArrayList<Instances>();
        for(ArrayList att_arr : sub_sets){
            //todo: from here, print fucking ist
            Instances sliced_data = FeatureSelection.keepAttributes(att_arr, data);
            returns.add(sliced_data);
            logger.info(FeatureSelection.toStringAttributeNames(sliced_data));
        }
        return returns;

    }


    /**
     * Converts a Weka Instances dataset to a upo.jml.data.dataset.ClassificationDataset object.
     * @param data The Instances dataset to convert
     * @return
     */
    public static ClassificationDataset Instances2ClassificationDataset(Instances data) throws Exception {
        //ClassificationDataset ret = new ClassificationDataset(


        File tempFile = File.createTempFile("tmp_instances-", ".arff");
        logger.info("Created tmp file, path: "+tempFile.getAbsolutePath());
        tempFile.deleteOnExit();

        logger.info("Saving weka instances on the file");
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(tempFile);
        saver.writeBatch();

        logger.info("Opening the arff file with ClassificationDataset constructor");
        ClassificationDataset dataset = DatasetUtils.loadArffClassificationDataset(tempFile.getAbsolutePath(), -1);
        //System.out.println(dataset);


        return dataset;
    }


    private static String toStringAttributeNames(Instances data){
        StringBuilder buf = new StringBuilder();
        for(int i=0; i < data.numAttributes(); i++){
            buf.append(data.attribute(i).name()).append(" ");
        }
        return buf.toString();
    }


    public static FCBFBagSearch removeIrrilevantFeatures(ClassificationDataset dataset) throws Exception {
        if (!dataset.getDataType().equals(DataType.CATEGORICAL))
            dataset = DatasetUtils.dicretizeViaFayyad(dataset);

        // FSObjectiveFunction ------------------------------------------------------------------------------
        // FSObjectiveFunction of = new CfsEvaluator(dataset.getCategoricalData(), dataset.getLabels());
        // of.buildEvaluator();
        // FSPredGroupsBasicVNS bvns = new FSPredGroupsBasicVNS(dataset.getCategoricalData(), dataset.getLabels(), of, true);
        // FSSolution solution = bvns.search();
        // System.out.println(solution);
        // System.out.println("------------------------------");


        // FSObjectiveFunction ------------------------------------------------------------------------------
        FCBFBagSearch algorithm = new FCBFBagSearch(dataset, 0.0);
        //FCBFBagSearch algorithm = new FCBFBagSearch(dataset.getCategoricalData(), dataset.getLabels(), 0.);

        FSSolution solution3 = algorithm.search();

        List<FeatureBag> bags = algorithm.getBags();
        System.out.println("#bags: " + bags.size());
        for(int i = 0; i < bags.size(); i++) {
            System.out.println(bags.get(i));
        }

        System.out.println(solution3);
        return algorithm;
    }

}
