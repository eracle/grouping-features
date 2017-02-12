package it.ci;

import org.junit.Test;
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

    @Test
    public void testOpenArff() throws IOException {
        File file = get_ionosphere();
        Instances data = FeatureSelection.openArff(file);
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
        FeatureSelection.splitFeatures(data);
    }
}