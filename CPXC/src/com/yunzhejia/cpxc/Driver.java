package com.yunzhejia.cpxc;


import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Driver {
	public static void main(String[] args){
		AbstractClassifier cpxc = new CPXC(ClassifierType.NAIVE_BAYES,ClassifierType.NAIVE_BAYES, 0.45, 0.01,1);
		DataSource source;
		Instances data;
		try {
			source = new DataSource("data/ILPD.arff");
			//source = new DataSource("data/diabetes.arff");
			data = source.getDataSet();
			if (data.classIndex() == -1){
				data.setClassIndex(data.numAttributes() - 1);
			}
			cpxc.buildClassifier(data);
			
			 
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
}
