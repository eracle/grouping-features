package it.ci;

import weka.core.Debug;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import weka.core.converters.ArffLoader.ArffReader;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
/**
 * Adaptor which allows to convert a WEKA RandomSubset object to be converted to a
 * upo.jml.data.dataset.ClassificationDataset object.
 * Created by eracle on 30/01/17.
 */
public class DatasetConverter {

    private static final int seed = 15;

    public Instances data;

    public DatasetConverter(File file) throws IOException {
        BufferedReader reader =
                new BufferedReader(new FileReader(file));
        ArffReader arff = new ArffReader(reader);
        Instances data = arff.getData();
        // data.setClassIndex(data.numAttributes() - 1);

        Random random = new Random(seed);
        data.randomize(random);
        this.data = data;
    }
}
