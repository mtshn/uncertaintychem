package ru.ac.phyche.badprediction2.featurepreprocessors;

import ru.ac.phyche.badprediction2.featuregenerators.FeaturesGenerator;

/**
 * 
 * The subclass of the abstract DropFeaturesPreprocessor class. Training is
 * disabled.
 *
 */
public class DropFeaturesPreprocessorNoTrain extends DropFeaturesPreprocessor {

	@Override
	public void train(FeaturesGenerator features, String[] data) {
		throw (new RuntimeException("Instance of DropFeaturesPreprocessorNoTrain cannot be trained!"
				+ " Use one of other sub-classes of DropFeaturesPreprocessor instead!"));
	}

}
