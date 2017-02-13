package it.ci;

import upo.jcu.io.Parameters;
import upo.jml.data.dataset.DatasetUtils;
import weka.core.Instance;
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
import java.util.Random;

import upo.jml.data.dataset.ClassificationDataset;

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

    // assumes the current class is called MyLogger
    private final static Logger log = Logger.getLogger(FeatureSelection.class.getName());

    private static final int seed = 15;



    public static Instances openArff(File file) throws IOException {
        Instances data;
        ArrayList<ClassificationDataset> datasets;

        log.info("Open file: "+ file.getAbsolutePath());
        BufferedReader reader = new BufferedReader(new FileReader(file));
        ArffReader arff = new ArffReader(reader);
        data = arff.getData();
        data.setClassIndex(data.numAttributes()-1);
        return data;

    }

    /**
     * Shortcut for RandomSubset object.
     * @param data
     * @param percent
     * @return
     */
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
     * Return a dataset the main dataset where were been removed the attributes contained
     * in smaller.
     * @param smaller
     * @param main
     * @return
     */
    private static Instances computeDifference(Instances smaller, Instances main)  throws Exception{
        for(int i=0; i < smaller.numAttributes(); i++){
            log.finer("Removing attribute:"+smaller.attribute(i).name()+" from the rest of the data");
            Remove remove = new Remove();

            for(int j=0; j< main.numAttributes(); j++){
                String att_name = main.attribute(j).name();
                if(att_name.equals(smaller.attribute(i).name())){
                    log.finer("Found attribute to remove: "+att_name);
                    remove.setAttributeIndices(""+j);
                    remove.setInputFormat(main);
                }
            }
            log.finer("Applying remove filter");
            main = Filter.useFilter(main, remove);
        }
        return main;
    }

    public static ArrayList<Instances> splitFeatures(Instances data)  throws Exception{
        ArrayList<Instances> returns = new ArrayList<Instances>();

        double[] percent_series = new double[5];
        percent_series[0] = 0.5;
        percent_series[1] = 0.2;
        percent_series[2] = 0.25;
        percent_series[3] = 0.3333;
        percent_series[4] = 0.5;

        for(int k = 0; k < 4 ; k++){
            log.fine("Extracting half of the attributes");
            Instances percentage = RandomSubsetWrapper(data, percent_series[k]);

            log.info((k+1)+"th percentage:\n"+toStringAttributeNames(percentage));
            returns.add(percentage);

            data = computeDifference(percentage, data);
        }

        log.info("last percentage:\n"+toStringAttributeNames(data));
        returns.add(data);
        return returns;

    }

    private static String toStringAttributeNames(Instances data){
        StringBuffer buf = new StringBuffer();
        for(int i=0; i < data.numAttributes(); i++){
            buf.append(data.attribute(i).name()+ " ");
        }
        return buf.toString();
    }


    /**
     * Converts a Weka Instances dataset to a upo.jml.data.dataset.ClassificationDataset object.
     * @param data The Instances dataset to convert
     * @return
     */
    public static ClassificationDataset Instances2ClassificationDataset(Instances data) throws Exception {
        //ClassificationDataset ret = new ClassificationDataset(


        File tempFile = File.createTempFile("tmp_instances-", ".arff");
        log.info("Created tmp file, path: "+tempFile.getAbsolutePath());
        tempFile.deleteOnExit();

        log.info("Saving weka instances on the file");
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(tempFile);
        saver.writeBatch();

        log.info("Opening the arff file with ClassificationDataset constructor");
        ClassificationDataset dataset = DatasetUtils.loadArffClassificationDataset(tempFile.getAbsolutePath(), -1);
        System.out.println(dataset);


        return dataset;
    }
}



