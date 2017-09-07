package com.yunzhejia.unimelb.cpexpl;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;
import com.yunzhejia.cpxc.util.DataUtils;
import com.yunzhejia.pattern.ICondition;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.unimelb.cpexpl.CPExplainer.CPStrategy;
import com.yunzhejia.unimelb.cpexpl.CPExplainer.FPStrategy;
import com.yunzhejia.unimelb.cpexpl.CPExplainer.PatternSortingStrategy;
import com.yunzhejia.unimelb.cpexpl.CPExplainer.SamplingStrategy;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

public class ImprovingClassifier {
	public static void main(String[] args){
//		String[] files = {"balloon.arff","banana.arff", "blood.arff", 
//				"diabetes.arff","haberman.arff","hepatitis.arff","iris.arff","labor.arff",
//				"mushroom.arff","sick.arff","titanic.arff","vote.arff"};
//		String[] files = {"balloon.arff", "blood.arff", "diabetes.arff","haberman.arff","iris.arff","labor.arff"};
//		int[] numsOfExpl = {1,5,10};
//		int[] numsOfSamples={10,200,500,1000};
//		CPStrategy[] miningStrategies = {CPStrategy.APRIORI,CPStrategy.RF};
		SamplingStrategy[] samplingStrategies = {SamplingStrategy.PATTERN_BASED_PERTURBATION};
//		ClassifierGenerator.ClassifierType[] typesOfClassifier = {ClassifierType.LOGISTIC, ClassifierType.DECISION_TREE};
		
		
		
		String[] files = {"synthetic/balloon_synthetic.arff"};
//		String[] files = {"blood.arff"};
//		String[] files = {"iris.arff"};
		int[] numsOfExpl = {1};
		CPStrategy[] miningStrategies = {CPStrategy.APRIORI};
//		SamplingStrategy[] samplingStrategies = {SamplingStrategy.PATTERN_BASED_PERTURBATION};
		ClassifierGenerator.ClassifierType[] typesOfClassifier = {ClassifierType.LOGISTIC};
		int[] numsOfSamples={500};
		CPExplainer app = new CPExplainer();
//		RandomExplainer app = new RandomExplainer();
		try {
			for(String file:files){
//			Instances data = DataUtils.load("data/synthetic2.arff");
			Instances data = DataUtils.load("data/"+file);
			int numGoldFeature = data.numAttributes();
			Set<Integer> goldFeatures = new HashSet<>();
//			for(int i = 0; i < numGoldFeature-1; i++){
//				goldFeatures.add(i);
//			}
			
//			Instances data = DataUtils.load("tmp/newData.arff");
//			data = AddNoisyFeatureToData.generateNoisyData(data);
			DataUtils.save(data,"tmp/newwData.arff");
			
			//split the data into train and test
//			Instances train = DataUtils.load("data/synthetic/balloon_synthetic.arff");
//			Instances test = DataUtils.load("data/synthetic/balloon_synthetic.arff");
			Instances train = DataUtils.load("data/synthetic/DNF3G_train.arff");
			Instances test = DataUtils.load("data/synthetic/DNF3G_test.arff");
			
			for(CPStrategy miningStrategy : miningStrategies){
			for(SamplingStrategy samplingStrategy:samplingStrategies){
			for(int numOfSamples:numsOfSamples){
			
				for(ClassifierType type:typesOfClassifier){
					for(int numOfExpl:numsOfExpl){
						try{

			
			AbstractClassifier cl = ClassifierGenerator.getClassifier(ClassifierType.LOGISTIC);
			cl.buildClassifier(train);
			Instances newTrain = new Instances(train,0);
			for(Instance ins:train){
				goldFeatures = getGoldFeature(ins);
				try{
				List<IPattern> expls = app.getExplanations(FPStrategy.APRIORI, samplingStrategy, 
						miningStrategy, PatternSortingStrategy.PROBDIFF_AND_SUPP,
						cl, ins, train, numOfSamples, 0.15, 3, numOfExpl, false);
				Instance newIns = (Instance)ins.copy();
				if (expls.size()!=0){
					IPattern p = expls.get(0);
					for (ICondition c:p.getConditions()){
						if(!goldFeatures.contains(c.getAttrIndex())){
							ins.setMissing(c.getAttrIndex());
						}
					}
				}else{
//					System.err.println("No explanations!");
				}
				newTrain.add(newIns);
				}catch(Exception e){
					throw e;
//					e.printStackTrace();
				}
			}
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(cl, test);
			
			String output = "   acc="+eval.correct()*1.0/test.numInstances();
			System.out.println(output);
			
			
			AbstractClassifier cl2 = ClassifierGenerator.getClassifier(ClassifierType.LOGISTIC);
			cl2.buildClassifier(train);
			Evaluation eval2 = new Evaluation(train);
			eval2.evaluateModel(cl2, test);
			
			output = "   acc="+eval2.correct()*1.0/test.numInstances();
			System.out.println(output);
			}catch(Exception e){
//				throw e;
				e.printStackTrace();
//				continue;
			}
			
			
					}}}}}}
			
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}
	public static Set<Integer> getGoldFeature(Instance instance){
		Set<Integer> ret = new HashSet<>();
		 
		if (instance.stringValue(0).equals("0") && instance.stringValue(1).equals("0")&& instance.stringValue(2).equals("0")){
			ret.add(0);
			ret.add(1);
			ret.add(2);
			ret.add(3);
			ret.add(4);
			
		}else if (instance.stringValue(0).equals("0") && instance.stringValue(1).equals("0")&& instance.stringValue(2).equals("1")){
			ret.add(0);
			ret.add(1);
			ret.add(2);
			ret.add(5);
			ret.add(6);
		}else if (instance.stringValue(0).equals("0") && instance.stringValue(1).equals("1")&& instance.stringValue(2).equals("0")){
			ret.add(0);
			ret.add(1);
			ret.add(2);
			ret.add(7);
			ret.add(8);
		}else if (instance.stringValue(0).equals("0") && instance.stringValue(1).equals("1")&& instance.stringValue(2).equals("1")){
			ret.add(0);
			ret.add(1);
			ret.add(2);
			ret.add(9);
			ret.add(10);
		} else if (instance.stringValue(0).equals("1") && instance.stringValue(1).equals("0")&& instance.stringValue(2).equals("0")){
			ret.add(0);
			ret.add(1);
			ret.add(2);
			ret.add(11);
			ret.add(12);
		}else if (instance.stringValue(0).equals("1") && instance.stringValue(1).equals("0")&& instance.stringValue(2).equals("1")){
			ret.add(0);
			ret.add(1);
			ret.add(2);
			ret.add(13);
			ret.add(14);
		}else if (instance.stringValue(0).equals("1") && instance.stringValue(1).equals("1")&& instance.stringValue(2).equals("0")){
			ret.add(0);
			ret.add(1);
			ret.add(2);
			ret.add(15);
			ret.add(16);
		}else {
			ret.add(0);
			ret.add(1);
			ret.add(2);
			ret.add(17);
			ret.add(18);
		}
		
		return ret;
	}
//	public static Set<Integer> getGoldFeature(Instance instance){
//		Set<Integer> ret = new HashSet<>();
//		if(instance.stringValue(0).equals("1") && instance.stringValue(1).equals("1")){
//			ret.add(0);
//			ret.add(2);
//			ret.add(3);
//			ret.add(4);
//		}
//		else if(instance.stringValue(0).equals("0") && instance.stringValue(1).equals("1")){
//			ret.add(0);
//			ret.add(5);
//			ret.add(6);
//			ret.add(7);
//		}
//		return ret;
//	}
//	public static Set<Integer> getGoldFeature(Instance instance){
//		Set<Integer> ret = new HashSet<>();
//		if (instance.stringValue(0).equals("1")){ // act == STRETCH, age = ADULT
//			ret.add(0);
//			ret.add(3);
//			ret.add(4);
//		}else if (instance.stringValue(0).equals("2")){
//			ret.add(0);
//			ret.add(1);
//			ret.add(2);
//		}
//		return ret;
//	}
}
