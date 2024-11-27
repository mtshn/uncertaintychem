package ru.ac.phyche.badprediction2;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import org.openscience.cdk.exception.CDKException;

public class DatasetPredictions {
	public static class Entry {
		public String smiles;
		public float reference;
		public float[] prediction;
		public int subset;
	}

	private Entry[] data = new Entry[0];

	public Entry get(int i) {
		return data[i];
	}

	public static Entry[] toArray(ArrayList<Entry> e) {
		return e.toArray(new Entry[e.size()]);
	}

	public DatasetPredictions() {

	}

	public int length() {
		return data.length;
	}

	public String[] smiles() {
		String[] result = new String[data.length];
		for (int i = 0; i < result.length; i++) {
			result[i] = data[i].smiles;
		}
		return result;
	}

	public DatasetPredictions(Entry[] e) {
		this.data = e;
	}

	public DatasetPredictions(ArrayList<Entry> e) {
		this.data = toArray(e);
	}

	public Entry[] getData() {
		return data;
	}

	public void setData(Entry[] data) {
		this.data = data;
	}

	public void setData(ArrayList<Entry> data) {
		this.data = toArray(data);
	}

	public static DatasetPredictions loadFromFile(String filename) throws IOException {
		return loadFromFile(filename, false, null);
	}

	public static DatasetPredictions loadFromFile(String filename, boolean removeBadMols, String logFileBadMolecules)
			throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(filename)));
		int nPredictions = -100;
		String s = br.readLine();
		ArrayList<Entry> d = new ArrayList<Entry>();
		while (s != null) {
			if (!s.trim().equals("")) {
				String[] spl = s.trim().split("\\s+");
				Entry e = new Entry();
				e.smiles = spl[0];
				e.subset = Integer.parseInt(spl[1]);
				e.reference = Float.parseFloat(spl[2]);
				if (nPredictions == -100) {
					nPredictions = spl.length - 3;
				}
				if (nPredictions <= 0) {
					br.close();
					throw new RuntimeException("Wrong file format! " + s);
				}
				if (nPredictions != spl.length - 3) {
					br.close();
					throw new RuntimeException(
							"Wrong file format! Number of predicitions should be equal for all lines " + s);
				}
				e.prediction = new float[nPredictions];
				for (int i = 0; i < nPredictions; i++) {
					e.prediction[i] = Float.parseFloat(spl[3 + i]);
				}
				d.add(e);
			}
			s = br.readLine();
		}
		br.close();
		Entry[] result = DatasetPredictions.toArray(d);
		AtomicInteger x2 = new AtomicInteger(0);
		Arrays.stream(ArUtls.intsrnd(result.length)).parallel().forEach(i -> {
			if (x2.incrementAndGet() % 1000 == 0) {
				System.out.println(x2.get());
			}
			try {
				String newSmiles = ChemUtils.canonical(ChemUtils.canonical(result[i].smiles, false), false);
				if (newSmiles != null) {
					result[i].smiles = newSmiles;
				} else {
					throw new CDKException("SMILES null");
				}
			} catch (CDKException e) {
				result[i].smiles = "Unsupported_smiles_string_" + result[i].smiles + "_"
						+ e.getMessage().replace(" ", "_");
			}

		});

		if (removeBadMols) {
			ArrayList<Entry> d2 = new ArrayList<Entry>();
			FileWriter fw = new FileWriter(new File(logFileBadMolecules));
			if (logFileBadMolecules != null) {
				for (int i = 0; i < result.length; i++) {
					if (result[i].smiles.contains("Unsupported_smiles_string_")) {
						if (fw != null) {
							fw.write(result[i].smiles);
						}
					} else {
						d2.add(result[i]);
					}
				}
			}
			fw.close();
			return new DatasetPredictions(d2);
		} else {
			for (int i = 0; i < result.length; i++) {
				if (result[i].smiles.contains("Unsupported_smiles_string_")) {
					throw new RuntimeException(result[i].smiles);
				}
			}
			return new DatasetPredictions(result);
		}
	}

	public float[][] modelsCoincidenceFeatures() {
		float[][] result = new float[data.length][];
		for (int i = 0; i < data.length; i++) {
			float s1 = 0;
			float s2 = 0;
			for (int j = 1; j < data[i].prediction.length; j++) {
				s1 = s1 + Math.abs(data[i].prediction[j] - data[i].prediction[0]);
				s2 = s2 + (data[i].prediction[j] - data[i].prediction[0])
						* (data[i].prediction[j] - data[i].prediction[0]);
			}
			s1 = s1 == 0 ? 0 : s1 / (data[i].prediction.length - 1);
			s2 = s2 == 0 ? 0 : s2 / (data[i].prediction.length - 1);
			s2 = (float) Math.sqrt(s2);
			float min = (float) ArUtls.minmax(data[i].prediction).getLeft();
			float max = (float) ArUtls.minmax(data[i].prediction).getRight();
			result[i] = new float[] { s1, s2, max - min};
		}
		return result;
	}

	public void saveToFile(String filename) throws IOException {
		FileWriter fw = new FileWriter(new File(filename));
		for (int i = 0; i < data.length; i++) {
			String s = data[i].smiles + " " + data[i].subset + " " + data[i].reference;
			for (int j = 0; j < data[i].prediction.length; j++) {
				s = s + " " + data[i].prediction[j];
			}
			fw.write(s + "\n");
		}
		fw.close();
	}

	public DatasetPredictions simpleSplit(float fraction) {
		return simpleSplit((int) Math.round(fraction * this.length()));
	}

	public DatasetPredictions simpleSplit(int nToSplit) {
		ArrayList<Entry> d1 = new ArrayList<Entry>();
		ArrayList<Entry> d2 = new ArrayList<Entry>();
		for (int i = 0; i < this.length(); i++) {
			if (i < nToSplit) {
				d1.add(this.data[i]);
			} else {
				d2.add(this.data[i]);
			}
		}
		this.data = toArray(d2);
		return new DatasetPredictions(d1);
	}

	public DatasetPredictions compoundBasedSplit(float fraction) {
		String[] sm = this.smiles();
		HashSet<String> smi = new HashSet<String>();
		for (String s : sm) {
			smi.add(s);
		}
		int n = Math.round(smi.size()*fraction);
		return this.compoundBasedSplit(n);
	}

	public DatasetPredictions compoundBasedSplit(int nCompoundsToSplit) {
		String[] sm = this.smiles();
		HashSet<String> smi = new HashSet<String>();
		for (String s : sm) {
			smi.add(s);
		}
		HashSet<String> smi1 = new HashSet<String>();

		int j = 0;
		for (String s : smi) {
			if (j < nCompoundsToSplit) {
				smi1.add(s);
			}
			j = j + 1;
		}

		ArrayList<Entry> d1 = new ArrayList<Entry>();
		ArrayList<Entry> d2 = new ArrayList<Entry>();
		for (int i = 0; i < this.length(); i++) {
			if (smi1.contains(data[i].smiles)) {
				d1.add(this.data[i]);
			} else {
				d2.add(this.data[i]);
			}
		}
		this.data = toArray(d2);
		return new DatasetPredictions(d1);
	}

	public void shuffle() {
		Random rnd = new Random();
		for (int c = 0; c < 10; c++) {
			for (int i = 0; i < data.length; i++) {
				int j = rnd.nextInt(data.length);
				Entry a = data[i];
				Entry b = data[j];
				data[j] = a;
				data[i] = b;
			}
		}
	}

	public static Entry deepCopy(Entry e) {
		Entry result = new Entry();
		result.prediction = new float[e.prediction.length];
		for (int i = 0; i < e.prediction.length; i++) {
			result.prediction[i] = e.prediction[i];
		}
		result.smiles = e.smiles + "";
		result.reference = e.reference;
		result.subset = e.subset;
		return result;
	}

	public DatasetPredictions deepCopy() {
		Entry[] n = new Entry[data.length];
		for (int i = 0; i < n.length; i++) {
			n[i] = deepCopy(data[i]);
		}
		return new DatasetPredictions(n);
	}
}
