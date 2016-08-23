package com.yunzhejia.cpxc.data;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Data {
	public static void main(String[] args){
		DataSource source;
		Instances data;
		try {
			source = new DataSource("data/ILPD.csv");
			data = source.getDataSet();
			// setting class attribute if the data format does not provide this information
			 // For example, the XRFF format saves the class attribute information as well
			
			for (int i = 0 ; i < data.numInstances(); i++){
			 }
			 if (data.classIndex() == -1)
			   data.setClassIndex(data.numAttributes() - 1);
			 System.out.println(data.classIndex());
			 Classifier cl = new NaiveBayes();
			 cl.buildClassifier(data);
			 for (int i = 0 ; i < data.numInstances(); i++){
				 System.out.println(cl.classifyInstance(data.get(i)));
			 }
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}
}	
