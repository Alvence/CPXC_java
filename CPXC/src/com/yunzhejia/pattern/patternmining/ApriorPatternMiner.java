package com.yunzhejia.pattern.patternmining;


import com.yunzhejia.pattern.PatternSet;

import weka.associations.AbstractAssociator;
import weka.associations.Apriori;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.supervised.attribute.Discretize;

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
			String file="data/adult/train.arff";
			String fileTest="data/adult/test.arff";
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
		
			AbstractAssociator apr = new Apriori();
			
			Discretize discretize =  new weka.filters.supervised.attribute.Discretize();
			
			int[] indices = new int[data.numAttributes()];
			for(int i=0;i< indices.length;i++){
				indices[i]=i;
			}
			discretize.setAttributeIndicesArray(indices);
			discretize.setInputFormat(data);
			discretize.useFilter(data, discretize);
			apr.buildAssociations(data);
			
			
			 
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
