package ru.ac.phyche.badprediction2.clusters;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import ru.ac.phyche.badprediction2.featuregenerators.FeaturesGenerator;

public abstract class Clustering {

	private int clustersNum = 10;

	public abstract int[] train(float[][] features);

	public abstract void save(String directoryName) throws IOException;

	public abstract void load(String directoryName) throws IOException;

	public abstract int[] predict(float[][] features);

	public abstract void init(float[] parameters);

	public abstract void init();

	public abstract String modelType();

	public abstract String paramsNames();

	public int[] predict(String[] smiles, FeaturesGenerator g) {
		g.precompute(smiles);
		float[][] f = g.features(smiles);
		int[] c = this.predict(f);
		return c;
	}

	public ArrayList<String[]> predictAndSplit(String[] d, FeaturesGenerator g) {
		g.precompute(d);
		float[][] f = g.features(d);
		int[] c = this.predict(f);
		@SuppressWarnings("unchecked")
		ArrayList<String>[] r = new ArrayList[clustersNum];
		for (int i = 0; i < clustersNum; i++) {
			r[i] = new ArrayList<String>();
		}
		if (c.length != d.length) {
			throw new RuntimeException("Unexcpected error. Array sizes should be equal");
		}
		for (int i = 0; i < d.length; i++) {
			r[c[i]].add(d[i]);
		}
		ArrayList<String[]> result = new ArrayList<String[]>();
		for (int i = 0; i < clustersNum; i++) {
			result.add(r[i].toArray(new String[r[i].size()]));
		}
		return result;
	}

	public int[] train(String[] smiles, FeaturesGenerator g) {
		g.precompute(smiles);
		float[][] f = g.features(smiles);
		int[] c = this.train(f);
		return c;
	}

	public ArrayList<String[]> trainAndSplit(String[] trainset, FeaturesGenerator g) {
		g.precompute(trainset);
		float[][] f = g.features(trainset);
		int[] c = this.train(f);
		@SuppressWarnings("unchecked")
		ArrayList<String>[] r = new ArrayList[clustersNum];
		for (int i = 0; i < clustersNum; i++) {
			r[i] = new ArrayList<String>();
		}
		if (c.length != trainset.length) {
			throw new RuntimeException("Unexcpected error. Array sizes should be equal");
		}
		for (int i = 0; i < trainset.length; i++) {
			r[c[i]].add(trainset[i]);
		}
		ArrayList<String[]> result = new ArrayList<String[]>();
		for (int i = 0; i < clustersNum; i++) {
			result.add(r[i].toArray(new String[r[i].size()]));
		}
		return result;
	}

	public int getClustersNum() {
		return clustersNum;
	}

	public void setClustersNum(int clustersNum) {
		this.clustersNum = clustersNum;
	}

	public static Clustering getModel(String clusteringType) {
		if (clusteringType.equals((new KMeansSimple()).modelType())) {
			return new KMeansSimple();
		}
		if (clusteringType.equals((new KMeansPCA()).modelType())) {
			return new KMeansPCA();
		}
		if (clusteringType.equals((new KMeansIterative()).modelType())) {
			return new KMeansIterative();
		}
		if (clusteringType.equals((new DBSCANClustering()).modelType())) {
			return new DBSCANClustering();
		}
		throw new RuntimeException("Unknkown clustering type");
	}

	public static Clustering loadModel(String directoryName) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(directoryName, "ModelType.txt")));
		String s = br.readLine();
		while (s.trim().equals("")) {
			s = br.readLine();
		}
		br.close();
		Clustering result = getModel(s.trim());
		result.load(directoryName);
		return result;
	}
}
