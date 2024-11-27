package ru.ac.phyche.badprediction2;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.lang3.tuple.Pair;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.interfaces.IAtomContainer;

import ru.ac.phyche.badprediction2.DatasetPredictions.Entry;
import ru.ac.phyche.badprediction2.ChemUtils.FingerprintsType;

public class MolecularSimilarity {

	public static final int N_PRESEARCH_SIMILAR_FOR_MCS = 250;

	public static int[] toInts(float fp[]) {
		int[] r = new int[fp.length];
		for (int i = 0; i < fp.length; i++) {
			r[i] = (int) Math.round(fp[i]);
		}
		return r;
	}

	public static int[] fp(String smiles, FingerprintsType fingerprintsType) throws CDKException {
		float[] f = ChemUtils.fingerprints(smiles, fingerprintsType);
		return toInts(f);
	}

	public static float tanimoto(int[] fp1, int[] fp2) {
		int all1 = 0;
		int all2 = 0;
		int both12 = 0;
		for (int i = 0; i < fp1.length; i++) {
			if (fp1[i] > 0) {
				all1++;
			}
			if (fp2[i] > 0) {
				all2++;
			}
			if ((fp1[i] > 0) && (fp2[i] > 0)) {
				both12++;
			}
		}
		return ((float) (both12)) / ((float) all1 + (float) all2 - ((float) (both12)));
	}

	public static float cosineSimilarity(int[] fp1, int[] fp2) {
		if (fp2.length != fp2.length) {
			throw new RuntimeException("For elementwise operations arrays should be equal-sized");
		}
		float ab = 0;
		float aa = 0;
		float bb = 0;
		for (int i = 0; i < fp1.length; i++) {
			ab = ab + fp1[i] * fp2[i];
			aa = aa + fp1[i] * fp1[i];
			bb = bb + fp2[i] * fp2[i];
		}
		float r = ab / ((float) Math.sqrt(aa * bb));
		return r;
	}

	public static int[][] calculateAllFingerprints(DatasetPredictions a, FingerprintsType fingerprintsType) {
		int[][] result = new int[a.length()][];
		AtomicInteger x2 = new AtomicInteger(0);
		Arrays.stream(ArUtls.intsrnd(a.length())).parallel().forEach(i -> {
			if (x2.incrementAndGet() % 1000 == 0) {
				System.out.println(x2.get());
			}
			try {
				result[i] = fp(a.get(i).smiles, fingerprintsType);
			} catch (CDKException e) {
				e.printStackTrace();
				throw new RuntimeException(e.getMessage());
			}
		});
		return result;
	}

	private static class ComparableResult implements Comparable<ComparableResult> {
		float similarity;
		int number;

		@Override
		public int compareTo(ComparableResult o) {
			ComparableResult o1 = (ComparableResult) o;
			return (-1) * ((Float) similarity).compareTo(o1.similarity);
		}
	}

	public static Pair<int[], float[]> mostSimilarMoleculesFromOtherSubsets(Entry query, int[] queryFingerprints,
			DatasetPredictions dataSet, int[][] allFingerprints, int n) {
		ComparableResult[] x = new ComparableResult[dataSet.length()];
		for (int i = 0; i < dataSet.length(); i++) {
			x[i] = new ComparableResult();
			x[i].number = i;
			x[i].similarity = tanimoto(queryFingerprints, allFingerprints[i]);
			if (query.subset == dataSet.get(i).subset) {
				x[i].similarity = -1;
			}
		}
		Arrays.sort(x);
		int[] result = new int[n];
		for (int i = 0; i < n; i++) {
			result[i] = x[i].number;
		}
		float[] similarities = new float[n];
		for (int i = 0; i < n; i++) {
			similarities[i] = x[i].similarity;
		}

		return Pair.of(result, similarities);
	}

	public static Pair<int[], float[]> mostSimilarMoleculesFromOtherSubsetsCosine(Entry query, int[] queryFingerprints,
			DatasetPredictions dataSet, int[][] allFingerprints, int n) {
		ComparableResult[] x = new ComparableResult[dataSet.length()];
		for (int i = 0; i < dataSet.length(); i++) {
			x[i] = new ComparableResult();
			x[i].number = i;
			x[i].similarity = cosineSimilarity(queryFingerprints, allFingerprints[i]);
			if (query.subset == dataSet.get(i).subset) {
				x[i].similarity = -1;
			}
		}
		Arrays.sort(x);
		int[] result = new int[n];
		for (int i = 0; i < n; i++) {
			result[i] = x[i].number;
		}
		float[] similarities = new float[n];
		for (int i = 0; i < n; i++) {
			similarities[i] = x[i].similarity;
		}

		return Pair.of(result, similarities);
	}

	public static float euclide(float[] a, float[] b) {
		float x = 0;
		if (a.length != b.length) {
			throw new RuntimeException("For elementwise operations arrays should be equal-sized");
		}
		for (int i = 0; i < a.length; i++) {
			x = x + (a[i] - b[i]) * (a[i] - b[i]);
		}
		x = (float) Math.sqrt(x);
		return x;
	}

	public static Pair<int[], float[]> mostSimilarMoleculesFromOtherSubsetsEuclideDistance(Entry query,
			float[] queryFeatures, DatasetPredictions dataSet, float[][] allFeatures, int n) {
		ComparableResult[] x = new ComparableResult[dataSet.length()];
		for (int i = 0; i < dataSet.length(); i++) {
			x[i] = new ComparableResult();
			x[i].number = i;
			x[i].similarity = 0 - euclide(queryFeatures, allFeatures[i]);
			if (query.subset == dataSet.get(i).subset) {
				x[i].similarity = Float.NEGATIVE_INFINITY;
			}
		}
		Arrays.sort(x);
		int[] result = new int[n];
		for (int i = 0; i < n; i++) {
			result[i] = x[i].number;
		}
		float[] similarities = new float[n];
		for (int i = 0; i < n; i++) {
			similarities[i] = 0 - x[i].similarity;
		}
		return Pair.of(result, similarities);
	}

	public static int[][] mostSimilarMoleculesFromOtherSubsets(DatasetPredictions querySet, DatasetPredictions dataSet,
			int n, int[][] queryFingerprints, int[][] fingerprints) {
		int[][] results = new int[querySet.length()][];
		AtomicInteger x2 = new AtomicInteger(0);
		Arrays.stream(ArUtls.intsrnd(querySet.length())).parallel().forEach(i -> {
			results[i] = mostSimilarMoleculesFromOtherSubsets(querySet.get(i), queryFingerprints[i], dataSet,
					fingerprints, n).getLeft();
			if (x2.incrementAndGet() % 1000 == 0) {
				System.out.println(x2.get());
			}
		});
		return results;
	}

	public static float[][] sMaxN(DatasetPredictions querySet, DatasetPredictions dataSet, int n,
			int[][] queryFingerprints, int[][] fingerprints) {
		float[][] results = new float[querySet.length()][];
		AtomicInteger x2 = new AtomicInteger(0);
		Arrays.stream(ArUtls.intsrnd(querySet.length())).parallel().forEach(i -> {
			results[i] = mostSimilarMoleculesFromOtherSubsets(querySet.get(i), queryFingerprints[i], dataSet,
					fingerprints, n).getRight();
			if (x2.incrementAndGet() % 1000 == 0) {
				System.out.println(x2.get());
			}
		});
		return results;
	}

	public static float[][] sMaxNCosine(DatasetPredictions querySet, DatasetPredictions dataSet, int n,
			int[][] queryFingerprints, int[][] fingerprints) {
		float[][] results = new float[querySet.length()][];
		AtomicInteger x2 = new AtomicInteger(0);
		Arrays.stream(ArUtls.intsrnd(querySet.length())).parallel().forEach(i -> {
			results[i] = mostSimilarMoleculesFromOtherSubsetsCosine(querySet.get(i), queryFingerprints[i], dataSet,
					fingerprints, n).getRight();
			if (x2.incrementAndGet() % 1000 == 0) {
				System.out.println(x2.get());
			}
		});
		return results;
	}

	private static float[] stringsToFloats(String[] s) {
		float[] result = new float[s.length];
		for (int i = 0; i < result.length; i++) {
			try {
				result[i] = Float.parseFloat(s[i]);
				if (result[i] < 0.0) {
					result[i] = Float.NaN;
				}
				if (result[i] > 1.0) {
					result[i] = Float.NaN;
				}
			} catch (Throwable e) {
				e.printStackTrace();
				result[i] = Float.NaN;
			}
		}
		return result;
	}

	public static float[][] sMaxNMCSSimilarity(DatasetPredictions querySet, DatasetPredictions dataSet, int n,
			int[][] queryFingerprints, int[][] fingerprints, String python) {
		int k = N_PRESEARCH_SIMILAR_FOR_MCS;
		if (k > dataSet.length() / 2) {
			k = dataSet.length() / 2;
		}
		int[][] preSelected = mostSimilarMoleculesFromOtherSubsets(querySet,dataSet, k, queryFingerprints,fingerprints);
		float[][] result = new float[querySet.length()][];

		int nProcs = Runtime.getRuntime().availableProcessors();
		nProcs = Math.min(nProcs, querySet.length());
		String[] querySmiles = querySet.smiles();
		String[] dataSmiles = dataSet.smiles();
		int subsetSize = querySmiles.length / nProcs;
		for (int i = 0; i < nProcs; i++) {
			int min = i * subsetSize;
			int max = (i + 1) * subsetSize;
			if (i == nProcs - 1) {
				max = querySmiles.length;
			}
			try {
				FileWriter fw = new FileWriter("tmpForRDKit_" + i + ".txt");
				for (int j = min; j < max; j++) {
					fw.write(querySmiles[j]);
					for (int l = 0; l < preSelected[j].length; l++) {
						fw.write(" " + dataSmiles[preSelected[j][l]]);
					}
					fw.write("\n");
				}
				fw.close();
			} catch (IOException e) {
				throw (new RuntimeException(e.getMessage()));
			}
		}
		Arrays.stream(ArUtls.ints(nProcs)).parallel().forEach(i -> {
			PythonRunner.runPython("mcs.py", "" + i, python);
		});
		float[][] similarities = new float[querySet.length()][k];
		for (int i = 0; i < nProcs; i++) {
			try {
				BufferedReader outFile = new BufferedReader(new FileReader("tmpForRDKit_out_" + i + ".txt"));
				int min = i * subsetSize;
				int max = (i + 1) * subsetSize;
				if (i == nProcs - 1) {
					max = querySmiles.length;
				}
				outFile.readLine();
				for (int j = min; j < max; j++) {
					String x = outFile.readLine();
					if (x.split("\\,").length != k) {
						throw (new RuntimeException("wrong output of python script"));
					}
					similarities[j] = stringsToFloats(x.split("\\,"));
				}
				outFile.close();
				deleteFileIfExist("tmpForRDKit_" + i + ".txt");
				deleteFileIfExist("tmpForRDKit_out_" + i + ".txt");
			} catch (Exception e) {
				throw (new RuntimeException(e.getMessage()));
			}
		}
		for (int i = 0; i < result.length; i++) {
			float[] sorted = similarities[i].clone();
			Arrays.sort(sorted);
			result[i] = new float[n];
			for (int j = 0; j < result[i].length; j++) {
				result[i][j] = sorted[sorted.length - 1 - j];
			}
		}
		return result;
	}

	private static void deleteFileIfExist(String filename) {
		File f = new File(filename);
		if (f.exists()) {
			f.delete();
		}
	}

	public static float[][] dMinNEuclide(DatasetPredictions querySet, DatasetPredictions dataSet, int n, float[][] queryFeatures, float[][] features) {
		float[][] results = new float[querySet.length()][];
		AtomicInteger x2 = new AtomicInteger(0);
		Arrays.stream(ArUtls.intsrnd(querySet.length())).parallel().forEach(i -> {
			if (x2.incrementAndGet() % 1000 == 0) {
				System.out.println(x2.get());
			}
			results[i] = mostSimilarMoleculesFromOtherSubsetsEuclideDistance(querySet.get(i), queryFeatures[i], dataSet,
					features, n).getRight();
		});
		return results;
	}

	public static float mscSimilarity(IAtomContainer m1, IAtomContainer m2) {
		try {
			org.openscience.smsd.Isomorphism comparison = new org.openscience.smsd.Isomorphism(m1, m2,
					org.openscience.smsd.interfaces.Algorithm.MCSPlus, true, true, true);
			return ((float) comparison.getTanimotoSimilarity());
		} catch (Exception e) {
			throw new RuntimeException(e.getMessage());
		}
	}

	@SuppressWarnings("deprecation")
	public static float mscSimilarityCDKOnly(IAtomContainer m1, IAtomContainer m2) {
		try {
			org.openscience.cdk.smsd.Isomorphism comparison = new org.openscience.cdk.smsd.Isomorphism(
					org.openscience.cdk.smsd.interfaces.Algorithm.MCSPlus, false);
			comparison.init(m1, m2, true, false);
			return ((float) comparison.getTanimotoSimilarity());
		} catch (Exception e) {
			throw new RuntimeException(e.getMessage());
		}
	}

}
