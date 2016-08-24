package com.yunzhejia.cpxc.data;

import com.yunzhejia.cpxc.util.OutputUtils;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Data {
	public static void main(String[] args){
		DataSource source;
		Instances data;
		try {
			source = new DataSource("data/ILPD.arff");
			data = source.getDataSet();
			// setting class attribute if the data format does not provide this information
			 // For example, the XRFF format saves the class attribute information as well
			 if (data.classIndex() == -1)
			   data.setClassIndex(data.numAttributes() - 1);
			 AbstractClassifier cl = new NaiveBayes();
			 cl.buildClassifier(data);
			 
			 System.out.println(data.toSummaryString());
			 
			 int count = 0;
			 for (int i = 0 ; i < data.numInstances(); i++){
				 double[] dist = cl.distributionForInstance(data.get(i));
				 int label = (int)data.get(i).classValue();
				 double prob = dist[label];
				 
				 if (prob < 0.5) {
					 count++;
				 
				 System.out.println(count + " out of "+ data.numInstances() + "  label=" +label);
				 OutputUtils.print(dist);
				 }
			 }
			 
			 
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}
}	
