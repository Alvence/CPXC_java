package com.yunzhejia.unimelb.cpexpl;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import com.yunzhejia.cpxc.Discretizer;
import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;
import com.yunzhejia.cpxc.util.DataUtils;
import com.yunzhejia.pattern.ICondition;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.PatternSet;
import com.yunzhejia.pattern.patternmining.IPatternMiner;
import com.yunzhejia.pattern.patternmining.RFContrastPatternMiner;
import com.yunzhejia.pattern.patternmining.RFPatternMiner;
import com.yunzhejia.unimelb.cpexpl.sampler.AddNoisyFeatureToData;
import com.yunzhejia.unimelb.cpexpl.sampler.Sampler;
import com.yunzhejia.unimelb.cpexpl.sampler.SimplePerturbationSampler;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class CPExplainer {

	public List<IPattern> getExplanations(AbstractClassifier cl, Instance instance, Instances headerInfo, int N, double minSupp, double minRatio, int K) throws Exception{
		List<IPattern> ret = new ArrayList<>();
		System.out.println("instance being tested: " + instance+" classification="+  cl.classifyInstance(instance));
		//step 1, sample the neighbours from the instance
		/*
		Sampler sampler = new SimplePerturbationSampler();
		*/
		 Discretizer discretizer0 = new Discretizer();
			discretizer0.initialize(headerInfo);
//		IPatternMiner pm = new GcGrowthPatternMiner(discretizer0);
		IPatternMiner pm = new RFPatternMiner();
		PatternSet ps = pm.minePattern(headerInfo, minSupp);
//		Sampler sampler = new PatternBasedSampler(ps);
		Sampler sampler = new SimplePerturbationSampler();
		
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
		IPatternMiner patternMiner = new RFContrastPatternMiner();
//		IPatternMiner patternMiner = new GcGrowthContrastPatternMiner(discretizer);
		PatternSet patternSet = patternMiner.minePattern(samples, minSupp, minRatio, (int)cl.classifyInstance(instance), true);
		
		//step 4, select K patterns and convert them to explanations.
		patternSet=sort(patternSet);
//		patternSet = sortByConfidence(cl, instance, patternSet);
		patternSet = patternSet.getMatchingPatterns(instance);
//		print_pattern(patternSet,K,"positive");
		for(int i = 0; i < K && i < patternSet.size(); i++){
			IPattern p = patternSet.get(i);
//			System.out.println(p + "  sup=" + p.support()+" ratio="+p.ratio());
//			System.out.println("With pattern   "+predictionByPattern(cl, instance, p));
//			System.out.println("Original   "+prediction(cl, instance));
//			System.out.println("Without pattern   "+predictionByRemovingPattern(cl, instance, p));
//			System.out.println();
			ret.add(p);
		}
		
		
//		patternSet = patternMiner.minePattern(samples, minSupp, minRatio, (int)cl.classifyInstance(instance), false);
//		patternSet=sort(patternSet);
//		print_pattern(patternSet,K,"negative");
		
		return ret;
	}
	
	public void print_pattern(PatternSet ps, int K, String name){
		System.out.println(name);
		for(int i = 0; i < K && i < ps.size(); i++){
			IPattern p = ps.get(i);
			System.out.println(p + "  sup=" + p.support()+" ratio="+p.ratio());
		}
	}
	
	//Get the prediction using only features appearing in the pattern
	public Prediction predictionByPattern(AbstractClassifier cl, Instance instance, IPattern pattern) throws Exception{
		Prediction pred = new Prediction();
		
		Instance ins = (Instance)instance.copy();

		Set<Integer> attrs = new HashSet<>();
		for (ICondition condition:pattern.getConditions()){
			attrs.add(condition.getAttrIndex());
		}
		
		for (int i = 0 ; i < ins.numAttributes(); i++){
			if (!attrs.contains(i)){
				ins.setMissing(i);
			}
		}
//		System.out.println(ins);
		pred.classIndex = cl.classifyInstance(ins);
		pred.prob = cl.distributionForInstance(ins)[(int)pred.classIndex];
		
		return pred;
	}
	
	
	public Prediction prediction(AbstractClassifier cl, Instance instance) throws Exception{
		Prediction pred = new Prediction();
		pred.classIndex = cl.classifyInstance(instance);
		pred.prob = cl.distributionForInstance(instance)[(int)pred.classIndex];
		return pred;
	}
	
	//Get the prediction without features appearing in the pattern
		public Prediction predictionByRemovingPattern(AbstractClassifier cl, Instance instance, IPattern pattern) throws Exception{
			Prediction pred = new Prediction();
			
			Instance ins = (Instance)instance.copy();

			Set<Integer> attrs = new HashSet<>();
			for (ICondition condition:pattern.getConditions()){
				attrs.add(condition.getAttrIndex());
			}
			
			for (int i = 0 ; i < ins.numAttributes(); i++){
				if (attrs.contains(i)){
					ins.setMissing(i);
				}
			}
//			System.out.println(ins);
			pred.classIndex = cl.classifyInstance(ins);
			pred.prob = cl.distributionForInstance(ins)[(int)cl.classifyInstance(instance)];
			
			return pred;
		}
	
	
	private PatternSet sortByConfidence(AbstractClassifier cl, Instance instance, PatternSet ps) throws Exception{
		PatternSet ret = new PatternSet();
		
//		double[] scores = new double[ps.size()];
		Map<IPattern, Double> scores = new HashMap<>(); 
		for(int i = 0; i < ps.size(); i++){
//			scores.put(ps.get(i), predictionByPattern(cl, instance, ps.get(i)).prob);
			scores.put(ps.get(i), prediction(cl, instance).prob - predictionByRemovingPattern(cl, instance, ps.get(i)).prob);
		}
		
		for(int i = 0; i < ps.size(); i++){
			IPattern p = ps.get(i);
			int index = 0;
			while(ret.size()>index && (scores.get(ret.get(index))+ret.get(index).support()/2) > (scores.get(p)+p.support()/2)){
				index++;
			}
			ret.add(index, p);
		}
		return ret;
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
			Instances data = DataUtils.load("data/balloon.arff");
//			Instances data = DataUtils.load("data/titanic/train.arff");
			int numGoldFeature = data.numAttributes();
			Set<Integer> goldFeatures = new HashSet<>();
			for(int i = 0; i < numGoldFeature-1; i++){
				goldFeatures.add(i);
			}
			
//			Instances data = DataUtils.load("tmp/newData.arff");
			data = AddNoisyFeatureToData.generateNoisyData(data);
			
			Random random = new Random(0);
			//split the data into train and test
			data.randomize(random);
			Instances train = data.trainCV(5, 1);
			Instances test = data.testCV(5, 1);
			
			AbstractClassifier cl = ClassifierGenerator.getClassifier(ClassifierType.LOGISTIC);
			cl.buildClassifier(train);
			double precision = 0;
			for(Instance ins:test){
				List<IPattern> expls = app.getExplanations(cl, ins, data, 2000, 0.01, 10, 1);
				precision += ExplEvaluation.eval(expls, goldFeatures);
			}
			System.out.println("precision = "+precision/test.numInstances());
			//calc gold features
			
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}
	
	class Prediction{
		double classIndex;
		double prob;
		
		@Override
		public String toString(){
			return "ClassIndex = "+classIndex+"  Prob="+prob;
		}
	}
}
