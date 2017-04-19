package com.yunzhejia.unimelb.cpexpl;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.yunzhejia.pattern.ICondition;
import com.yunzhejia.pattern.IPattern;

public class ExplEvaluation {
	public static double eval(List<IPattern> explanations, Set<Integer> goldFeatures){
		List<Double> recalls = new ArrayList<>();
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
	
	public static double evalRecall(List<IPattern> explanations, Set<Integer> goldFeatures){
		List<Double> recalls = new ArrayList<>();
		List<Double> precisions = new ArrayList<>();
		Set<Integer> features = new HashSet<>();
		
		double recall = 0;
		for(IPattern pattern: explanations){
			for(ICondition condition:pattern.getConditions()){
				features.add(condition.getAttrIndex());
			}
		}
		int tp = 0;
		int fp = 0;
		for(int f:features){
			if(goldFeatures.contains(f)){
				fp++;
			}
		}
		recall = fp*1.0/goldFeatures.size();
//		System.out.println("averagePrecision = "+averagePrecision);
		return recall;
	}
}
