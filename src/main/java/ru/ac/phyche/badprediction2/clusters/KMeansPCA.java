package ru.ac.phyche.badprediction2.clusters;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import com.thoughtworks.xstream.XStream;
import com.thoughtworks.xstream.io.xml.StaxDriver;

import ru.ac.phyche.badprediction2.ArUtls;
import smile.clustering.KMeans;
import smile.feature.extraction.PCA;

public class KMeansPCA extends Clustering {
	private KMeans km = null;
	private int nPCA = 10;
	private PCA pca = null;

	public PCA getPca() {
		return pca;
	}

	@Override
	public int[] train(float[][] features) {
		double[][] featuresDouble = ArUtls.toDoubleArray2d(features);
		nPCA = Math.min(Math.min(nPCA, features.length), features[0].length);
		pca = PCA.fit(featuresDouble).getProjection(nPCA);
		double[][] pcaD = pca.apply(featuresDouble);
		if (pcaD[0].length != nPCA) {
			throw new RuntimeException("Wrong matrix dimension after PCA");
		}
		if (pcaD.length != features.length) {
			throw new RuntimeException("Wrong matrix dimension after PCA");
		}
		km = KMeans.fit(pcaD, this.getClustersNum());
		int[] r = new int[features.length];
		for (int i = 0; i < r.length; i++) {
			r[i] = km.predict(pcaD[i]);
		}
		return r;
	}

	@Override
	public void save(String directoryName) throws IOException {
		Files.createDirectories(Paths.get(directoryName));
		FileWriter fw = new FileWriter(new File(directoryName, "ModelType.txt"));
		fw.write(this.modelType());
		fw.close();
		fw = new FileWriter(new File(directoryName, "k.txt"));
		fw.write(this.getClustersNum() + "");
		fw.close();
		fw = new FileWriter(new File(directoryName, "kmeans.xml"));
		XStream xstream = new XStream(new StaxDriver());
		xstream.toXML(km, fw);
		fw.close();
		fw = new FileWriter(new File(directoryName, "pca.xml"));
		xstream = new XStream(new StaxDriver());
		xstream.toXML(pca, fw);
		fw.close();
	}

	@Override
	public void load(String directoryName) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(directoryName, "ModelType.txt")));
		String s = br.readLine();
		while (s.trim().equals("")) {
			s = br.readLine();
		}
		br.close();
		if (!s.trim().equals(this.modelType())) {
			throw new RuntimeException("Wrong model type");
		}
		br = new BufferedReader(new FileReader(new File(directoryName, "k.txt")));
		s = br.readLine();
		while (s.trim().equals("")) {
			s = br.readLine();
		}
		br.close();
		this.setClustersNum(Integer.parseInt(s.trim()));
		XStream xstream = new XStream();
		xstream.allowTypes(new String[] { "smile.clustering.KMeans" });
		km = (KMeans) xstream.fromXML(new File(directoryName, "kmeans.xml"));
		xstream = new XStream();
		xstream.allowTypes(new String[] { "smile.math.matrix.Matrix$RowMajor", "smile.data.type.DoubleType",
				"smile.feature.extraction.PCA", "smile.math.matrix.Matrix", "smile.math.matrix.Matrix$1",
				"smile.data.type.StructField" });
		pca = (PCA) xstream.fromXML(new File(directoryName, "pca.xml"));
		this.nPCA = pca.projection.nrow();
	}

	@Override
	public int[] predict(float[][] features) {
		double[][] featuresDouble = ArUtls.toDoubleArray2d(features);
		double[][] pcaD = pca.apply(featuresDouble);
		if (pcaD[0].length != nPCA) {
			throw new RuntimeException("Wrong matrix dimension after PCA");
		}
		if (pcaD.length != features.length) {
			throw new RuntimeException("Wrong matrix dimension after PCA");
		}
		int[] r = new int[features.length];
		for (int i = 0; i < r.length; i++) {
			r[i] = km.predict(pcaD[i]);
		}
		return r;
	}

	@Override
	public void init(float[] parameters) {
		this.setClustersNum(Math.round(parameters[0]));
		this.nPCA = Math.round(parameters[1]);
		if (parameters.length != 2) {
			throw new RuntimeException("Wrong number of parameters. KMeansPCA has 2 parameters.");
		}
	}

	@Override
	public void init() {
		init(new float[] { 10, 10 });
	}

	@Override
	public String modelType() {
		return "KMeansPCA";
	}

	@Override
	public String paramsNames() {
		return "k nPCA";
	}

	public float[] distanceFromCenterOfCluster(float[][] features) {
		double[][] featuresDouble = ArUtls.toDoubleArray2d(features);
		double[][] pcaD = pca.apply(featuresDouble);
		float[] r = new float[features.length];
		for (int i = 0; i < r.length; i++) {
			int clNum = km.predict(pcaD[i]);
			double[] clCenter = km.centroids[clNum];
			double distance = 0;
			for (int j = 0; j < clCenter.length; j++) {
				distance += (clCenter[j] - pcaD[i][j]) * (clCenter[j] - pcaD[i][j]);
			}
			distance = Math.sqrt(distance);
			r[i] = (float) distance;
		}
		return r;
	}

	public float[] meanDistancesFromCenterOfCluster(float[][] features) {
		int[] ns = new int[this.getClustersNum()];
		double[] sumDistances = new double[this.getClustersNum()];
		double[][] featuresDouble = ArUtls.toDoubleArray2d(features);
		double[][] pcaD = pca.apply(featuresDouble);
		for (int i = 0; i < pcaD.length; i++) {
			int clNum = km.predict(pcaD[i]);
			double[] clCenter = km.centroids[clNum];
			double distance = 0;
			for (int j = 0; j < clCenter.length; j++) {
				distance += (clCenter[j] - pcaD[i][j]) * (clCenter[j] - pcaD[i][j]);
			}
			distance = Math.sqrt(distance);
			ns[clNum] = ns[clNum] + 1;
			sumDistances[clNum] = sumDistances[clNum] + distance;
		}
		float[] result = ArUtls.toFloatArray(sumDistances);
		for (int i = 0; i < sumDistances.length; i++) {
			result[i] = result[i] / ns[i];
		}
		return result;
	}

	public float[] relativeDistanceFromCenterOfCluster(float[][] queryFeatures,float[][] dataFeatures) {
		double[][] featuresDouble = ArUtls.toDoubleArray2d(dataFeatures);
		double[][] pcaD = pca.apply(featuresDouble);
		float[] r = new float[queryFeatures.length];
		float[] meanDistances = meanDistancesFromCenterOfCluster(dataFeatures);
		float[] distances = distanceFromCenterOfCluster(queryFeatures);
		for (int i = 0; i < r.length; i++) {
			int clNum = km.predict(pcaD[i]);
			r[i] = distances[i] / meanDistances[clNum];
		}
		return r;
	}

}