package it.ci;

import java.io.File;
import java.util.List;
import java.util.logging.Logger;
import upo.jcu.math.data.dataset.DataType;
import upo.jml.data.dataset.ClassificationDataset;
import upo.jml.data.dataset.DatasetUtils;
import upo.jml.data.dataset.ToyDatasets;
import upo.jml.prediction.classification.fss.algorithms.FCBFBagSearch;
import upo.jml.prediction.classification.fss.algorithms.FSPredGroupsBasicVNS;
import upo.jml.prediction.classification.fss.core.FSObjectiveFunction;
import upo.jml.prediction.classification.fss.core.FSSolution;
import upo.jml.prediction.classification.fss.core.FeatureBag;
import upo.jml.prediction.classification.fss.evaluators.CfsEvaluator;

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
public class VNS {

    private static final Logger LOG = Logger.getLogger(VNS.class.getName());

    public static FCBFBagSearch testToyIris(int[][] data, int[] labels) throws Exception {


        FSObjectiveFunction of = new CfsEvaluator(data, labels);
        of.buildEvaluator();
        // FSPredGroupsBasicVNS bvns = new FSPredGroupsBasicVNS(data, labels, of);
        // FSSolution solution = bvns.search();

        FCBFBagSearch algorithm = new FCBFBagSearch(data, labels, 0.);


        return algorithm;
    }



}
