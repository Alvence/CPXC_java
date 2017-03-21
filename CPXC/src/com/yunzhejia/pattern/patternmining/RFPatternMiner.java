package com.yunzhejia.pattern.patternmining;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.yunzhejia.cpxc.util.DataUtils;
import com.yunzhejia.pattern.ICondition;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.NominalCondition;
import com.yunzhejia.pattern.NumericCondition;
import com.yunzhejia.pattern.Pattern;
import com.yunzhejia.pattern.PatternSet;

import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.classifiers.trees.RandomTree.Tree;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class RFPatternMiner implements IPatternMiner {
	
	
	@Override
	public PatternSet minePattern(Instances data, double minSupp) {
		RandomForest rf = new RandomForest();
		List<IPattern> patternSet = new ArrayList<>();
		
		try {
			rf.buildClassifier(data);
			Classifier[] trees = rf.getClassifiers();
			for(Classifier cl:trees){
				RandomTree tree = (RandomTree)cl;
				Set<IPattern> patterns = new HashSet<>();
				generatePatterns(tree.m_Tree, tree.m_Info, patterns,null, minSupp );
				for (IPattern pattern:patterns){
					if (!patternSet.contains(pattern)){
						patternSet.add(pattern);
					}
				}
			}
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		
		return new PatternSet(patternSet);
	}

	@Override
	public PatternSet minePattern(Instances data, double minSupp, int featureId) {
		// TODO Auto-generated method stub
		return null;
	}
	
//	public static void outputTree(Tree tree) throws Exception{
//		if (tree.m_Attribute == -1){
//			System.out.println( "classDist: " + Arrays.toString(tree.m_ClassDistribution)+"   count"+c++);
//		}else{
//			System.out.print("("+tree.m_Attribute + "<" + tree.m_SplitPoint+")  AND  ");
//			outputTree(tree.m_Successors[0]);
//			System.out.print("("+tree.m_Attribute + ">=" + tree.m_SplitPoint+")   AND  ");
//			outputTree(tree.m_Successors[1]);
//		}
//	}
//	
	

	public void generatePatterns(Tree tree, Instances m_Info, Set<IPattern> patterns, Set<ICondition> conditions, double minSupport) throws Exception{
		if (conditions == null){
			conditions = new HashSet<>();
		}
		if (tree.m_Attribute == -1){
			if(conditions != null && !conditions.isEmpty() && tree.m_ClassDistribution != null){
				if(tree.m_ClassDistribution[Utils.maxIndex(tree.m_ClassDistribution)] >= minSupport){
					patterns.add(new Pattern(conditions));
				}
			}
		}else if (m_Info.attribute(tree.m_Attribute).isNominal()) {

	          // For nominal attributes
	          for (int i = 0; i < tree.m_Successors.length; i++) {
	        	  ICondition condition = new NominalCondition(m_Info.attribute(tree.m_Attribute).name(), tree.m_Attribute, m_Info.attribute(tree.m_Attribute).value(i));
	        	  Set<ICondition>  newConditions = new HashSet<>();
	        	  if (conditions!=null){
	        		  for (ICondition con:conditions){
	        			  newConditions.add(con);
	        		  }
	        	  }
	        	  newConditions.add(condition);
	        	  generatePatterns(tree.m_Successors[i],m_Info,patterns,newConditions, minSupport);
	          }
		}else{
			for (int i = 0; i < 2; i++) {
	        	ICondition condition;  
				if (i==0){
	        		  condition = new NumericCondition(m_Info.attribute(tree.m_Attribute).name(), tree.m_Attribute, Double.MIN_VALUE, tree.m_SplitPoint);
	        	  }else{
	        		  condition = new NumericCondition(m_Info.attribute(tree.m_Attribute).name(), tree.m_Attribute, tree.m_SplitPoint, Double.MAX_VALUE);
	        	  }
	        	  Set<ICondition>  newConditions = new HashSet<>();
	        	  if (conditions!=null){
	        		  for (ICondition con:conditions){
	        			  newConditions.add(con);
	        		  }
	        	  }
	        	  newConditions.add(condition);
	        	  generatePatterns(tree.m_Successors[i],m_Info,patterns,newConditions, minSupport);
	          }
		}
	}
	public static void main(String[] args){
		try {
			Instances data = DataUtils.load("data/ILPD.arff");
			PatternSet ps = new RFPatternMiner().minePattern(data, data.numInstances()*0.1);
			System.out.println(ps.size());
			for(IPattern p:ps){
				System.out.println(p);
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Override
	public PatternSet minePattern(Instances data, double minSupp, double minRatio, int classIndex) {
		// TODO Auto-generated method stub
		return null;
	}
}
