package it.ci;

import org.junit.Test;

import java.io.File;
import java.io.IOException;

/**
 * Created by eracle on 30/01/17.
 */
public class DatasetConverterTest {

    @Test
    public void testGetClassificationDataset() throws IOException {
        ClassLoader classLoader = getClass().getClassLoader();
        File file = new File(classLoader.getResource("dexter_first_half.arff").getFile());
        DatasetConverter sut = new DatasetConverter(file);
        System.out.println(sut.data);
    }
}