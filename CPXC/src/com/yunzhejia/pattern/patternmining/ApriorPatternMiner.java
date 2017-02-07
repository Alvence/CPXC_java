package com.yunzhejia.pattern.patternmining;


import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

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

public class ApriorPatternMiner implements IPatternMiner {

	@Override
	public PatternSet minePattern(Instances data, double minSupp) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public PatternSet minePattern(Instances data, double minSupp, int featureId) {
		// TODO Auto-generated method stub
		return null;
	}
	
	public static void main(String[] args){
		try {
			DataSource source;
			Instances data;
			
			DataSource sourceTest;
			Instances dataTest;
//			String[] files = {"data/synthetic2.arff"};
			String file="data/synthetic2.arff";
			String fileTest="data/sick/test.arff";
//			source = new DataSource("data/synthetic2.arff");
			source = new DataSource(file);
			sourceTest = new DataSource(fileTest);
//			source = new DataSource("data/iris.arff");
			data = source.getDataSet();
			dataTest = source.getDataSet();
			
			if (data.classIndex() == -1){
				data.setClassIndex(data.numAttributes() - 1);
			}
			if (dataTest.classIndex() == -1){
				dataTest.setClassIndex(dataTest.numAttributes() - 1);
			}
		
			RandomForest rf = new RandomForest();
			
			rf.buildClassifier(data);
			System.out.println(rf.getClassifiers().length);
			Classifier[] trees = rf.getClassifiers();
			RandomTree tree = (RandomTree)trees[0];
			System.out.println(tree);
			outputTree(tree.m_Tree);
			
			Set<IPattern> patterns = new HashSet<>();
			generatePatterns(tree.m_Tree, tree.m_Info, patterns,null, data.numAttributes()*0.01);
			for(IPattern p : patterns){
				System.out.println(p + "  sup="+ p.support(data));
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	static int c = 0;
	public static void outputTree(Tree tree) throws Exception{
		if (tree.m_Attribute == -1){
			System.out.println( "classDist: " + Arrays.toString(tree.m_ClassDistribution)+"   count"+c++);
		}else{
			System.out.print("("+tree.m_Attribute + "<" + tree.m_SplitPoint+")  AND  ");
			outputTree(tree.m_Successors[0]);
			System.out.print("("+tree.m_Attribute + ">=" + tree.m_SplitPoint+")   AND  ");
			outputTree(tree.m_Successors[1]);
		}
	}
	
	

	public static void generatePatterns(Tree tree, Instances m_Info, Set<IPattern> patterns, Set<ICondition> conditions, double minSupport) throws Exception{
		if (patterns == null){
			patterns = new HashSet<>();
		}
		if (conditions == null){
			conditions = new HashSet<>();
		}
		if (tree.m_Attribute == -1){
			if(conditions != null && !conditions.isEmpty()){
				if(tree.m_ClassDistribution[Utils.maxIndex(tree.m_ClassDistribution)] >= minSupport){
					patterns.add(new Pattern(conditions));
				}
				return;
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
}
