package ru.ac.phyche.badprediction2;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import ru.ac.phyche.badprediction2.featuregenerators.CDKDescriptorsGenerator;
import ru.ac.phyche.badprediction2.featuregenerators.CDKFingerprintsGenerator;
import ru.ac.phyche.badprediction2.featuregenerators.CombinedFeaturesGenerator;
import ru.ac.phyche.badprediction2.featuregenerators.FeaturesGenerator;
import ru.ac.phyche.badprediction2.featuregenerators.FuncGroupsCDKGenerator;
import ru.ac.phyche.badprediction2.featuregenerators.PreprocessedFeaturesGenerator;
import ru.ac.phyche.badprediction2.featurepreprocessors.CombinedFeaturesPreprocessor;
import ru.ac.phyche.badprediction2.featurepreprocessors.DropConstantFeaturesPreprocessor;
import ru.ac.phyche.badprediction2.featurepreprocessors.DropFeaturesWithNaNsPreprocessor;
import ru.ac.phyche.badprediction2.featurepreprocessors.DropHighCorrPreprocessor;
import ru.ac.phyche.badprediction2.featurepreprocessors.FeaturesPreprocessor;
import ru.ac.phyche.badprediction2.featurepreprocessors.ReplaceNaNsPreprocessor;
import ru.ac.phyche.badprediction2.featurepreprocessors.Scale01FeaturesPreprocessor;

public class FeatureGenerators {

	public static FeaturesPreprocessor defaultPreproc() {
		CombinedFeaturesPreprocessor p = new CombinedFeaturesPreprocessor();
		p.addPreprocessor(new DropFeaturesWithNaNsPreprocessor(0.001F));
		p.addPreprocessor(new ReplaceNaNsPreprocessor());
		p.addPreprocessor(new DropConstantFeaturesPreprocessor());
		// features with correlation >0.999 are always excluded
		p.addPreprocessor(new DropHighCorrPreprocessor(0.999f));
		p.addPreprocessor(new Scale01FeaturesPreprocessor());
		return p;
	}

	public static void trainPrecomputePreproc(PreprocessedFeaturesGenerator gen, String[] train) {
		FeaturesGenerator g = gen.getGenPreproc().getLeft();
		FeaturesPreprocessor p = gen.getGenPreproc().getRight();
		g.precompute(train);
		p.train(g, train);
	}

	public static void savePreproc(PreprocessedFeaturesGenerator gen, String filename) throws IOException {
		FeaturesPreprocessor p = gen.getGenPreproc().getRight();
		FileWriter fw = new FileWriter(filename);
		p.save(fw);
		fw.close();
	}

	public static void loadPreproc(PreprocessedFeaturesGenerator gen, String filename) throws IOException {
		FeaturesPreprocessor p = gen.getGenPreproc().getRight();
		BufferedReader br = new BufferedReader(new FileReader(filename));
		p.load(br);
		br.close();
	}

	public static PreprocessedFeaturesGenerator allDescriptors() {
		FeaturesPreprocessor p = defaultPreproc();
		ArrayList<FeaturesGenerator> lst = new ArrayList<FeaturesGenerator>();
		lst.add(new CDKDescriptorsGenerator());
		lst.add(new FuncGroupsCDKGenerator());
		CombinedFeaturesGenerator gen1 = new CombinedFeaturesGenerator(lst.toArray(new FeaturesGenerator[lst.size()]));
		PreprocessedFeaturesGenerator g = new PreprocessedFeaturesGenerator(gen1, p);
		return g;
	}

	public static FeaturesGenerator allDescriptorsNoPreproc() {
		ArrayList<FeaturesGenerator> lst = new ArrayList<FeaturesGenerator>();
		lst.add(new CDKDescriptorsGenerator());
		lst.add(new FuncGroupsCDKGenerator());
		CombinedFeaturesGenerator gen1 = new CombinedFeaturesGenerator(lst.toArray(new FeaturesGenerator[lst.size()]));
		return gen1;
	}	
	
	public static PreprocessedFeaturesGenerator funcGroups() {
		FeaturesPreprocessor p = defaultPreproc();
		ArrayList<FeaturesGenerator> lst = new ArrayList<FeaturesGenerator>();
		lst.add(new FuncGroupsCDKGenerator());
		CombinedFeaturesGenerator gen1 = new CombinedFeaturesGenerator(lst.toArray(new FeaturesGenerator[lst.size()]));
		PreprocessedFeaturesGenerator g = new PreprocessedFeaturesGenerator(gen1, p);
		return g;
	}

	
	public static PreprocessedFeaturesGenerator cdkDescriptors() {
		FeaturesPreprocessor p = defaultPreproc();
		ArrayList<FeaturesGenerator> lst = new ArrayList<FeaturesGenerator>();
		lst.add(new CDKDescriptorsGenerator());
		CombinedFeaturesGenerator gen1 = new CombinedFeaturesGenerator(lst.toArray(new FeaturesGenerator[lst.size()]));
		PreprocessedFeaturesGenerator g = new PreprocessedFeaturesGenerator(gen1, p);
		return g;
	}
	
	public static PreprocessedFeaturesGenerator klekota() {
		FeaturesPreprocessor p = defaultPreproc();
		ArrayList<FeaturesGenerator> lst = new ArrayList<FeaturesGenerator>();
		lst.add(new CDKFingerprintsGenerator(ChemUtils.FingerprintsType.KLEKOTA_ADDITIVE));
		CombinedFeaturesGenerator gen1 = new CombinedFeaturesGenerator(lst.toArray(new FeaturesGenerator[lst.size()]));
		PreprocessedFeaturesGenerator g = new PreprocessedFeaturesGenerator(gen1, p);
		return g;
	}

	public static PreprocessedFeaturesGenerator klekotaAllDescriptors() {
		FeaturesPreprocessor p = defaultPreproc();
		ArrayList<FeaturesGenerator> lst = new ArrayList<FeaturesGenerator>();
		lst.add(new CDKDescriptorsGenerator());
		lst.add(new FuncGroupsCDKGenerator());
		lst.add(new CDKFingerprintsGenerator(ChemUtils.FingerprintsType.KLEKOTA_ADDITIVE));
		CombinedFeaturesGenerator gen1 = new CombinedFeaturesGenerator(lst.toArray(new FeaturesGenerator[lst.size()]));
		PreprocessedFeaturesGenerator g = new PreprocessedFeaturesGenerator(gen1, p);
		return g;
	}

	public static PreprocessedFeaturesGenerator eCFPAndAllDescriptors() {
		FeaturesPreprocessor p = defaultPreproc();
		ArrayList<FeaturesGenerator> lst = new ArrayList<FeaturesGenerator>();
		lst.add(new CDKDescriptorsGenerator());
		lst.add(new FuncGroupsCDKGenerator());
		lst.add(new CDKFingerprintsGenerator(ChemUtils.FingerprintsType.ECFP_6_4096_ADDITIVE));
		CombinedFeaturesGenerator gen1 = new CombinedFeaturesGenerator(lst.toArray(new FeaturesGenerator[lst.size()]));
		PreprocessedFeaturesGenerator g = new PreprocessedFeaturesGenerator(gen1, p);
		return g;
	}	
	
	public static PreprocessedFeaturesGenerator allDescriptorsTrained(String[] train) {
		PreprocessedFeaturesGenerator g = allDescriptors();
		trainPrecomputePreproc(g, train);
		return g;
	}

	public static PreprocessedFeaturesGenerator funcGroupsTrained(String[] train) {
		PreprocessedFeaturesGenerator g = funcGroups();
		trainPrecomputePreproc(g, train);
		return g;
	}

	public static PreprocessedFeaturesGenerator klekotaTrained(String[] train) {
		PreprocessedFeaturesGenerator g = klekota();
		trainPrecomputePreproc(g, train);
		return g;
	}

	public static PreprocessedFeaturesGenerator klekotaAllDescriptorsTrained(String[] train) {
		PreprocessedFeaturesGenerator g = klekotaAllDescriptors();
		trainPrecomputePreproc(g, train);
		return g;
	}

	public static PreprocessedFeaturesGenerator cdkDescriptorsTrained(String[] train) {
		PreprocessedFeaturesGenerator g = cdkDescriptors();
		trainPrecomputePreproc(g, train);
		return g;
	}

	public static PreprocessedFeaturesGenerator eCFPAndAllDescriptorsTrained(String[] train) {
		PreprocessedFeaturesGenerator g = eCFPAndAllDescriptors();
		trainPrecomputePreproc(g, train);
		return g;
	}
	
}