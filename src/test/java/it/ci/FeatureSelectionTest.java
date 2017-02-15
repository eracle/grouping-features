package it.ci;

import org.junit.Assert;
import org.junit.Test;
import upo.jml.data.dataset.ClassificationDataset;
import weka.core.Instances;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.logging.Logger;

import static org.junit.Assert.*;

/**
 * Created by eracle on 30/01/17.
 */
public class FeatureSelectionTest {


    // assumes the current class is called MyLogger
    private final static Logger log = Logger.getLogger(FeatureSelectionTest.class.getName());

    private File get_ionosphere(){
        ClassLoader classLoader = getClass().getClassLoader();
        File file = new File(classLoader.getResource("ionosphere.arff").getFile());
        return file;
    }

    private Instances getIonosphereArff() throws IOException {
        File file = get_ionosphere();
        return FeatureSelection.openArff(file);
    }
    @Test
    public void testOpenArff() throws IOException {
        Instances data =  getIonosphereArff();
        assertTrue(data!=null);
        for(int i=0; i < data.numAttributes(); i++){
            log.info(data.attribute(i).name());
        }
        //System.out.println(sut.data);
    }

    @Test
    public void testSplitFeatures() throws Exception {
        File file = get_ionosphere();
        Instances data = FeatureSelection.openArff(file);

        ArrayList<Instances> list = FeatureSelection.splitFeatures(data);

        assertEquals(6, list.size());
        assertEquals(list.get(0).numAttributes()-1 , (data.numAttributes()-1)/2);

        int tenth_num_attr = (data.numAttributes()-1)/10;
        assertEquals(list.get(1).numAttributes()-1 , tenth_num_attr);

        assertEquals(list.get(2).numAttributes()-1 , tenth_num_attr);
        assertEquals(list.get(3).numAttributes()-1 , tenth_num_attr);
        assertEquals(list.get(4).numAttributes()-1 , tenth_num_attr);

        int last_num_attr = list.get(5).numAttributes()-1;

        assertTrue((last_num_attr == tenth_num_attr) || last_num_attr == (tenth_num_attr+1));
    }

    @Test
    public void testInstances2ClassificationDataset() throws Exception {
        Instances data = getIonosphereArff();
        ClassificationDataset c_dataset = FeatureSelection.Instances2ClassificationDataset(data);
        Assert.assertEquals(c_dataset.numberOfFeatures(), data.numAttributes()-1);
        Assert.assertEquals(c_dataset.numberOfInstances(), data.numInstances());
    }
}