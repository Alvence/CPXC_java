package com.yunzhejia.unimelb.cpexpl;

import java.util.List;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class CPExplainer {

	public List<Explanation> getExplanations(AbstractClassifier cl, Instance instance, int N, double minSupp, double minRatio, int K){
		List<Explanation> ret = null;
		
		//step 1, sample the neighbours from the instance
		Instances samples = sampleNeighbours(instance);
		
		//step 2, label the samples using the classifier cl
		samples = labelSample(samples, cl);
		
		//step 3, mine the contrast patterns from the newly labelled samples.
		
		//step 4, select K patterns and convert them to explanations.
		
		
		
		return ret;
	}

	private Instances labelSample(Instances samples, AbstractClassifier cl) {
		// TODO Auto-generated method stub
		return null;
	}

	private Instances sampleNeighbours(Instance instance) {
		// TODO Auto-generated method stub
		return null;
	}
}
