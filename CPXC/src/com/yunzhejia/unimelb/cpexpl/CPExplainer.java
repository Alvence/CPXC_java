package com.yunzhejia.unimelb.cpexpl;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;
import com.yunzhejia.cpxc.util.DataUtils;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.PatternSet;
import com.yunzhejia.pattern.patternmining.IPatternMiner;
import com.yunzhejia.pattern.patternmining.RFContrastPatternMiner;
import com.yunzhejia.unimelb.cpexpl.sampler.Sampler;
import com.yunzhejia.unimelb.cpexpl.sampler.SimplePerturbationSampler;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class CPExplainer {
	

	public List<Explanation> getExplanations(AbstractClassifier cl, Instance instance, Instances headerInfo, int N, double minSupp, double minRatio, int K) throws Exception{
		List<Explanation> ret = null;
		System.out.println("instance being tested: " + instance);
		//step 1, sample the neighbours from the instance
		Sampler sampler = new SimplePerturbationSampler();
		Instances samples = sampler.samplingFromInstance(headerInfo, instance, N);
		
//		System.out.println(samples);
		//step 2, label the samples using the classifier cl
		samples = labelSample(samples, cl);
//		System.out.println(samples);
//		ScatterPlotDemo3.render(ScatterPlotDemo3.createChart(samples, 0, 1));;
		//step 3, mine the contrast patterns from the newly labelled samples.
		IPatternMiner patternMiner = new RFContrastPatternMiner();
		PatternSet patternSet = patternMiner.minePattern(samples, minSupp, minRatio, (int)cl.classifyInstance(instance));
		
		for(IPattern p:patternSet){
			System.out.println(p + "  sup=" + p.support());
		}
		//step 4, select K patterns and convert them to explanations.
		
		
		
		return ret;
	}

	private Instances labelSample(Instances samples, AbstractClassifier cl) throws Exception {
		for (Instance ins:samples){
			ins.setClassValue(cl.classifyInstance(ins));
		}
		
		return samples;
	}

	private Instances sampleNeighbours(Instance instance) {
		// TODO Auto-generated method stub
		return null;
	}
	
	
	public static void main(String[] args){
		CPExplainer app = new CPExplainer();
		try {
			Instances data = DataUtils.load("data/synthetic2.arff");
//			Instances data = DataUtils.load("data/mushroom.arff");
			AbstractClassifier cl = ClassifierGenerator.getClassifier(ClassifierType.LOGISTIC);
			cl.buildClassifier(data);
			app.getExplanations(cl, data.get(1), data, 50, 3, 0.01, 10);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}
}
