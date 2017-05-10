package com.yunzhejia.unimelb.cpexpl;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.yunzhejia.pattern.ICondition;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.unimelb.cpexpl.patternselection.PatternEvaluation;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class ExplEvaluation {
	
	public class Evaluation{
		double recall;
		double precision;
		double probDiff;
	}
	
	
	public static double evalProbDiffAvg(List<IPattern> explanations, AbstractClassifier cl, Instances headerInfo, Instance x) throws Exception{
		double sum = 0.0;
		PatternEvaluation pEval = new PatternEvaluation();
		double probOld = cl.distributionForInstance(x)[(int)cl.classifyInstance(x)];
		for(IPattern pattern:explanations){
			sum+= (probOld - pEval.predictionByRemovingPattern(cl, x, pattern, headerInfo));
		}
		sum /= explanations.size();
		return sum;
	}
	
	public static double evalProbDiffMax(List<IPattern> explanations, AbstractClassifier cl, Instances headerInfo, Instance x) throws Exception{
		double max = 0.0;
		PatternEvaluation pEval = new PatternEvaluation();
		double probOld = cl.distributionForInstance(x)[(int)cl.classifyInstance(x)];
		for(IPattern pattern:explanations){
			double eval = probOld -pEval.predictionByRemovingPattern(cl, x, pattern, headerInfo);
			if(eval>max){
				max = eval;
			}
		}
		return max;
	}
	
	public static double evalProbDiffMin(List<IPattern> explanations, AbstractClassifier cl, Instances headerInfo, Instance x) throws Exception{
		double min = 1;
		PatternEvaluation pEval = new PatternEvaluation();
		double probOld = cl.distributionForInstance(x)[(int)cl.classifyInstance(x)];
		for(IPattern pattern:explanations){
			double eval = probOld -pEval.predictionByRemovingPattern(cl, x, pattern, headerInfo);
			if(eval<min){
				min = eval;
			}
		}
		return min;
	}
	
	
	public static double evalPrecision(List<IPattern> explanations, Set<Integer> goldFeatures){
		List<Double> precisions = new ArrayList<>();
		double averagePrecision = 0;
		for(IPattern pattern: explanations){
			int tp = 0;
			int fp = 0;
			for(ICondition condition:pattern.getConditions()){
				if(goldFeatures.contains(condition.getAttrIndex())){
					tp++;
				}else{
					fp++;
				}
			}
			double precision = tp*1.0/(tp+fp);
			precisions.add(precision);
			averagePrecision += precision;
		}
		averagePrecision = averagePrecision/precisions.size();
//		System.out.println("averagePrecision = "+averagePrecision);
		return averagePrecision;
	}
	
	public static double evalPrecisionBest(List<IPattern> explanations, Set<Integer> goldFeatures){
		List<Double> precisions = new ArrayList<>();
		double best = 0;
		for(IPattern pattern: explanations){
			int tp = 0;
			int fp = 0;
			for(ICondition condition:pattern.getConditions()){
				if(goldFeatures.contains(condition.getAttrIndex())){
					tp++;
				}else{
					fp++;
				}
			}
			double precision = tp*1.0/(tp+fp);
			if(precision > best){
				best = precision;
			}
		}
//		System.out.println("averagePrecision = "+averagePrecision);
		return best;
	}
	
	public static double evalRecall(List<IPattern> explanations, Set<Integer> goldFeatures){
		Set<Integer> features = new HashSet<>();
		
		List<Double> recalls = new ArrayList<>();
		double average = 0;
		for(IPattern pattern: explanations){
			int tp = 0;
			int fp = 0;
			for(ICondition condition:pattern.getConditions()){
				if(goldFeatures.contains(condition.getAttrIndex())){
					tp++;
				}else{
					fp++;
				}
			}
			double recall = tp*1.0/goldFeatures.size();
			recalls.add(recall);
			average += recall;
		}
		
		
		
		
		return average;
	}
	
	public static double evalRecallBest(List<IPattern> explanations, Set<Integer> goldFeatures){
		double best = 0;
		for(IPattern pattern: explanations){
			int tp = 0;
			int fp = 0;
			for(ICondition condition:pattern.getConditions()){
				if(goldFeatures.contains(condition.getAttrIndex())){
					tp++;
				}else{
					fp++;
				}
			}
			double recall = tp*1.0/goldFeatures.size();
			if(best < recall){
				best = recall;
			}
		}
		
//		System.out.println("averagePrecision = "+averagePrecision);
		return best;
	}
}
