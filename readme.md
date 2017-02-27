# Clustering attributes - Feature selection - tools
Simple library containing some tools used for research purposes on the field of Feature Selection.
It is used for data pre-processing and as a conversion tool between WEKA library and the upo.jml library.

The project is build using gradle and it uses junit.

#### Usage
Inside the src/main/java folder is possible to find a static classes which implements some basic functionality,
inside the it.ci package:

- FeatureSelection.java
 1. openArff
 2. splitFeatures
 3. Instances2ClassificationDataset

#### Example

```java
import ...

Instances data = FeatureSelection.openArff(file);

ArrayList<Instances> list = FeatureSelection.splitFeatures(data);

for(Instances ins: list){
    ClassificationDataset class_dataset = Instances2ClassificationDataset(ins);
    ...
}

```

See the src/test/java/it/ci/FeatureSelectionTest.java class for more examples.

#### Tests

Is possible to run tests with the default gradle method:

```bash
./gradlew compileJava test
```

[1]: https://docs.gradle.org/current/userguide/tutorial_gradle_command_line.html
