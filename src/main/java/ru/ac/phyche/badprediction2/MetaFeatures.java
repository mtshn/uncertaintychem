package ru.ac.phyche.badprediction2;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import org.openscience.cdk.exception.CDKException;

import ru.ac.phyche.badprediction2.ChemUtils.FingerprintsType;
import ru.ac.phyche.badprediction2.DatasetPredictions.Entry;
import ru.ac.phyche.badprediction2.clusters.KMeansPCA;
import ru.ac.phyche.badprediction2.featuregenerators.PreprocessedFeaturesGenerator;
import ru.ac.phyche.badprediction2.featurepreprocessors.FeaturesPreprocessor;
import ru.ac.phyche.badprediction2.featuregenerators.FeaturesGenerator;

public class MetaFeatures {

	private static float[][] empty2DArray(int length1) {
		float[][] result = new float[length1][];
		for (int i = 0; i < length1; i++) {
			result[i] = new float[0];
		}
		return result;
	}

	public static void calculateFeatures(boolean fitPreprocessorAndClusters, DatasetPredictions queryData,
			DatasetPredictions data, int nPCA, int kForKMeans, int nSimilarities, String outFeaturesFile,
			String fittedPreprocessorAndClustersDirectoryName, float[] thresholdDiscrepancy, boolean mcs, String python)
			throws IOException, CDKException {
		KMeansPCA km = null;
		PreprocessedFeaturesGenerator pfg = null;
		if (fitPreprocessorAndClusters) {
			Files.createDirectories(Paths.get(fittedPreprocessorAndClustersDirectoryName));
			pfg = FeatureGenerators.allDescriptorsTrained(data.smiles());
			FileWriter fw0 = new FileWriter(fittedPreprocessorAndClustersDirectoryName + "/preproc");
			pfg.getGenPreproc().getRight().save(fw0);
			fw0.close();
			pfg.precompute(data.smiles());
			pfg.precompute(queryData.smiles());
			km = new KMeansPCA();
			km.init(new float[] { kForKMeans, nPCA });
			km.train(data.smiles(), pfg);
			km.save(fittedPreprocessorAndClustersDirectoryName + "/clusters");
		} else {
			FeaturesGenerator fg = FeatureGenerators.allDescriptorsNoPreproc();
			BufferedReader br = new BufferedReader(
					new FileReader(fittedPreprocessorAndClustersDirectoryName + "/preproc"));
			FeaturesPreprocessor fpr = FeaturesPreprocessor.fromFile(br);
			br.close();
			pfg = new PreprocessedFeaturesGenerator(fg, fpr);
			pfg.precompute(data.smiles());
			pfg.precompute(queryData.smiles());
			km = new KMeansPCA();
			km.load(fittedPreprocessorAndClustersDirectoryName + "/clusters");
		}
		int[][] fp = MolecularSimilarity.calculateAllFingerprints(data, FingerprintsType.ECFP_6_8192_ADDITIVE);
		int[][] queryfp = MolecularSimilarity.calculateAllFingerprints(queryData,
				FingerprintsType.ECFP_6_8192_ADDITIVE);
		float[][] features = pfg.features(data.smiles());
		float[][] featuresPCA = ArUtls.toFloatArray2d(km.getPca().apply(ArUtls.toDoubleArray2d(features)));
		float[][] queryFeatures = pfg.features(queryData.smiles());
		float[][] queryFeaturesPCA = ArUtls.toFloatArray2d(km.getPca().apply(ArUtls.toDoubleArray2d(queryFeatures)));

		float[][] sMaxNMCSSimilarity = empty2DArray(data.length());
		if (mcs) {
			sMaxNMCSSimilarity = MolecularSimilarity.sMaxNMCSSimilarity(queryData, data, nSimilarities, queryfp, fp,
					python);
			System.out.println("MCS done");
		}
		float[][] sMaxN = MolecularSimilarity.sMaxN(queryData, data, nSimilarities, queryfp, fp);
		float[][] sMaxNCosine = MolecularSimilarity.sMaxNCosine(queryData, data, nSimilarities, queryfp, fp);
		float[][] sMaxNEuclide = MolecularSimilarity.dMinNEuclide(queryData, data, nSimilarities, queryFeaturesPCA,
				featuresPCA);
		int[] clusters = km.predict(queryFeatures);
		float[] distances = km.distanceFromCenterOfCluster(queryFeatures);
		float[] relativeDistances = km.relativeDistanceFromCenterOfCluster(queryFeatures, features);
		float[][] modelDiffs = queryData.modelsCoincidenceFeatures();
		FileWriter fw = new FileWriter(outFeaturesFile);
		fw.write("SMILES ");
		fw.write("Discrepancy ");
		for (int j = 0; j < thresholdDiscrepancy.length; j++) {
			fw.write("Label" + thresholdDiscrepancy[j] + " ");
		}
		fw.write("N_cluster ");
		fw.write("Distance_to_cluster_center ");
		fw.write("Relative_distance_to_cluster_center ");
		fw.write("MAE_in_cluster ");
		fw.write("MdAE_in_cluster ");
		fw.write("Mean_abs_models_difference ");
		fw.write("RMS_models_difference ");
		fw.write("MinMax_models_difference ");

		if ((Math.abs(sMaxN.length - sMaxNCosine.length) + Math.abs(sMaxN.length - sMaxNEuclide.length)
				+ Math.abs(sMaxN.length - clusters.length) + Math.abs(sMaxN.length - distances.length)
				+ Math.abs(sMaxN.length - relativeDistances.length)
				+ Math.abs(sMaxN.length - modelDiffs.length)) != 0) {
			fw.close();
			throw new RuntimeException("Lengths of arrays should be equal");
		}

		for (int j = 0; j < sMaxN[0].length; j++) {
			fw.write("Best_" + (j + 1) + "_TanimotoSimilarity ");
		}
		for (int j = 0; j < sMaxNCosine[0].length; j++) {
			fw.write("Best_" + (j + 1) + "_CosineSimilarity ");
		}
		for (int j = 0; j < sMaxNEuclide[0].length; j++) {
			fw.write("Best_" + (j + 1) + "_euclideDistance ");
		}
		for (int j = 0; j < sMaxNMCSSimilarity[0].length; j++) {
			fw.write("Best_" + (j + 1) + "_MCSSimilarity ");
		}
		for (int j = 0; j < data.get(0).prediction.length; j++) {
			fw.write("Prediction_by_model_" + j + " ");
		}
		fw.write("\n");

		float[] maeCl = new float[km.getClustersNum()];
		float[] mdaeCl = new float[km.getClustersNum()];
		int[] clustersData = km.predict(features);
		for (int j = 0; j < km.getClustersNum(); j++) {
			ArrayList<Float> errors = new ArrayList<Float>();
			for (int i = 0; i < data.length(); i++) {
				if (clustersData[i] == j) {
					Entry e = data.get(i);
					float err = Math.abs(e.reference - e.prediction[0]);
					errors.add(err);
				}
			}
			float[] errorsF = ArUtls.toFloatArray(errors);
			maeCl[j] = ArUtls.mean(errorsF);
			mdaeCl[j] = ArUtls.median(errorsF);
		}

		for (int i = 0; i < queryData.length(); i++) {
			Entry e = queryData.get(i);
			float err = Math.abs(e.reference - e.prediction[0]);
			String labels = "";
			for (int j = 0; j < thresholdDiscrepancy.length; j++) {
				int label = (err >= thresholdDiscrepancy[j]) ? 1 : 0;
				labels = labels + label + " ";
			}
			labels = labels.trim();
			String s = e.smiles + " " + err + " " + labels + " " + clusters[i] + " " + distances[i] + " "
					+ relativeDistances[i] + " " + maeCl[clusters[i]] + " " + mdaeCl[clusters[i]] + " ";
			for (int j = 0; j < modelDiffs[i].length; j++) {
				s = s + modelDiffs[i][j] + " ";
			}
			for (int j = 0; j < sMaxN[i].length; j++) {
				s = s + sMaxN[i][j] + " ";
			}
			for (int j = 0; j < sMaxNCosine[i].length; j++) {
				s = s + sMaxNCosine[i][j] + " ";
			}
			for (int j = 0; j < sMaxNEuclide[i].length; j++) {
				s = s + sMaxNEuclide[i][j] + " ";
			}
			for (int j = 0; j < sMaxNMCSSimilarity[i].length; j++) {
				s = s + sMaxNMCSSimilarity[i][j] + " ";
			}
			for (int j = 0; j < queryData.get(i).prediction.length; j++) {
				s = s + queryData.get(i).prediction[j] + " ";
			}
			s = s.trim() + "\n";
			fw.write(s);
		}
		fw.close();
	}

	public static void main(String[] args) throws Exception {


	}
}
