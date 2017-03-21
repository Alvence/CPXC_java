package com.yunzhejia.unimelb.cpexpl;

import java.util.List;

import org.jfree.data.DataUtilities;

import com.yunzhejia.cpxc.util.DataUtils;
import com.yunzhejia.unimelb.cpexpl.sampler.Sampler;
import com.yunzhejia.unimelb.cpexpl.sampler.SimplePerturbationSampler;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class CPExplainer {

	public List<Explanation> getExplanations(AbstractClassifier cl, Instance instance, Instances headerInfo, int N, double minSupp, double minRatio, int K){
		List<Explanation> ret = null;
		
		//step 1, sample the neighbours from the instance
		Sampler sampler = new SimplePerturbationSampler();
		Instances samples = sampler.samplingFromInstance(headerInfo, instance, N);
		
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
	
	
	public static void main(String[] args){
		CPExplainer app = new CPExplainer();
		try {
			Instances data = DataUtils.load("data/mushroom.arff");
			app.getExplanations(null, null, data, 50, 3, 0.01, 5);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}
}
