package com.yunzhejia.unimelb.cpexpl;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class Synthetic4Classifier extends AbstractClassifier {

	@Override
	public void buildClassifier(Instances data) throws Exception {

	}
	
	@Override
	public double[] distributionForInstance(Instance instance)throws Exception{
		double sum = 0.0;
		for(int i =0; i < 8;i++){
			sum+= Integer.parseInt(instance.stringValue(i));
		}
		if ((instance.stringValue(0).equals("1") && instance.stringValue(1).equals("1")&&instance.stringValue(2).equals("1"))||
				(instance.stringValue(3).equals("1") && instance.stringValue(4).equals("1")&&instance.stringValue(5).equals("1"))||
				(instance.stringValue(6).equals("1") && instance.stringValue(7).equals("1")&&instance.stringValue(8).equals("1"))){ // act == STRETCH, age = ADULT
				double[] ret = {0,1};
				return ret;
			}else if(sum>0){
				double[] ret= {1-sum/20,sum/20};
				return ret;
			}else{
				double[] ret = {1,0};
				return ret;
			}
			
	}

}
