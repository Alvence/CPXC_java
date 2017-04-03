package com.yunzhejia.unimelb.cpexpl;

import java.util.List;

import com.yunzhejia.cpxc.Discretizer;
import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;
import com.yunzhejia.cpxc.util.DataUtils;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.PatternSet;
import com.yunzhejia.pattern.patternmining.GcGrowthContrastPatternMiner;
import com.yunzhejia.pattern.patternmining.GcGrowthPatternMiner;
import com.yunzhejia.pattern.patternmining.IPatternMiner;
import com.yunzhejia.unimelb.cpexpl.sampler.PatternBasedSampler;
import com.yunzhejia.unimelb.cpexpl.sampler.Sampler;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class CPExplainer {

	public List<Explanation> getExplanations(AbstractClassifier cl, Instance instance, Instances headerInfo, int N, double minSupp, double minRatio, int K) throws Exception{
		List<Explanation> ret = null;
		System.out.println("instance being tested: " + instance+" classification="+  cl.classifyInstance(instance));
		//step 1, sample the neighbours from the instance
		/*
		Sampler sampler = new SimplePerturbationSampler();
		*/
		 Discretizer discretizer0 = new Discretizer();
			discretizer0.initialize(headerInfo);
		IPatternMiner pm = new GcGrowthPatternMiner(discretizer0);
		PatternSet ps = pm.minePattern(headerInfo, minSupp);
		Sampler sampler = new PatternBasedSampler(ps);
		
		
		Instances samples = sampler.samplingFromInstance(headerInfo, instance, N);
		
//		System.out.println(samples);
		//step 2, label the samples using the classifier cl
		samples = labelSample(samples, cl);
		
		/*
		List<Instances> tmp = new ArrayList<>();
		Instances tmp1= new Instances(headerInfo,0);
		tmp1.add(instance);
		tmp.add(tmp1);
		tmp.add(headerInfo);
		tmp.add(samples);
		
		ScatterPlotDemo3.render(ScatterPlotDemo3.createChart(tmp, 0, 1));;
		*/
		
		//step 3, mine the contrast patterns from the newly labelled samples.
		 Discretizer discretizer = new Discretizer();
		discretizer.initialize(headerInfo);
//		IPatternMiner patternMiner = new RFContrastPatternMiner();
		IPatternMiner patternMiner = new GcGrowthContrastPatternMiner(discretizer);
		PatternSet patternSet = patternMiner.minePattern(samples, minSupp, minRatio, (int)cl.classifyInstance(instance), true);
		
		//step 4, select K patterns and convert them to explanations.
		patternSet=sort(patternSet);
		patternSet = patternSet.getMatchingPatterns(instance);
		
		print_pattern(patternSet,K,"positive");
		
		
		
		patternSet = patternMiner.minePattern(samples, minSupp, minRatio, (int)cl.classifyInstance(instance), false);
		patternSet=sort(patternSet);
		print_pattern(patternSet,K,"negative");
		
		
		
		return ret;
	}
	
	public void print_pattern(PatternSet ps, int K, String name){
		System.out.println(name);
		for(int i = 0; i < K && i < ps.size(); i++){
			IPattern p = ps.get(i);
			System.out.println(p + "  sup=" + p.support()+" ratio="+p.ratio());
		}
	}
	
	private PatternSet sort(PatternSet ps){
		PatternSet ret = new PatternSet();
		for(IPattern p:ps){
			int index = 0;
			while(ret.size()>index && ret.get(index).support()>p.support()){
				index++;
			}
			ret.add(index, p);
		}
		return ret;
	}

	private Instances labelSample(Instances samples, AbstractClassifier cl) throws Exception {
		for (Instance ins:samples){
			ins.setClassValue(cl.classifyInstance(ins));
		}
		
		return samples;
	}
	
	public static void main(String[] args){
		CPExplainer app = new CPExplainer();
		try {
//			Instances data = DataUtils.load("data/vote.arff");
			Instances data = DataUtils.load("data/titanic/train.arff");
			AbstractClassifier cl = ClassifierGenerator.getClassifier(ClassifierType.NAIVE_BAYES);
			cl.buildClassifier(data);
			app.getExplanations(cl, data.get(1), data, 1000, 0.01, 5, 10);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}
}
