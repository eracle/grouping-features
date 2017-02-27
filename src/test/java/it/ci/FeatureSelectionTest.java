package it.ci;

import org.junit.Assert;
import org.junit.Test;
import upo.jml.data.dataset.ClassificationDataset;
import weka.core.Attribute;
import weka.core.Instances;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.logging.Logger;


import static org.junit.Assert.*;

/**
 * Created by eracle on 30/01/17.
 */
public class FeatureSelectionTest {

    private final Logger log = Logger.getLogger(FeatureSelectionTest.class.getName());

    private File get_ionosphere(){
        ClassLoader classLoader = getClass().getClassLoader();
        File file = new File(classLoader.getResource("ionosphere.arff").getFile());
        return file;
    }

    private Instances getIonosphereArff() throws IOException {
        File file = get_ionosphere();
        Instances data = FeatureSelection.openArff(file);
        data.setClassIndex(data.numAttributes()-1);
        return data;
    }

    @Test
    public void test_OpenArff() throws IOException {
        Instances data =  getIonosphereArff();
        assertTrue(data!=null);
        for(int i=0; i < data.numAttributes(); i++){
            log.info(data.attribute(i).name());
        }
        //System.out.println(sut.data);
    }

    @Test
    public void test_SplitFeatures() throws Exception {
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
    public void test_keepAttributes() throws Exception{
        Instances data = getIonosphereArff();

        int num_attr = data.numAttributes();
        log.info(""+num_attr);
        log.info(""+data.enumerateAttributes().toString());

        Enumeration attribute_enum = data.enumerateAttributes();
        Attribute first_att = (Attribute)attribute_enum.nextElement();
        Attribute second_att = (Attribute)attribute_enum.nextElement();

        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        attributes.add(first_att);
        attributes.add(second_att);

        Instances rest_data = FeatureSelection.keepAttributes(attributes, data);
        log.info(""+rest_data.enumerateAttributes().toString());
        log.info(""+rest_data.numAttributes());

        Assert.assertEquals(2, rest_data.numAttributes());
    }


    @Test
    public void test_Instances2ClassificationDataset() throws Exception {
        Instances data = getIonosphereArff();
        ClassificationDataset c_dataset = FeatureSelection.Instances2ClassificationDataset(data);
        Assert.assertEquals(c_dataset.numberOfFeatures(), data.numAttributes()-1);
        Assert.assertEquals(c_dataset.numberOfInstances(), data.numInstances());
    }
}