package com.yunzhejia.unimelb.cpexpl;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class Synthetic3Classifier extends AbstractClassifier {

	@Override
	public void buildClassifier(Instances data) throws Exception {

	}
	
	@Override
	public double[] distributionForInstance(Instance instance)throws Exception{
		double centerX = 5, centerY = 5;
		double r1 = 4,r2 = 1;
		double x = instance.value(1);
		double y = instance.value(2);
		
		double dis = Math.sqrt((x - centerX)*(x - centerX)+(y-centerY)*(y-centerY));
		if (instance.stringValue(0).equals("1")){ // act == STRETCH, age = ADULT
			if (dis>=r2 && dis<= r1){
				double[] ret = {0,1};
				return ret;
			}else{
				double[] ret = {1,0};
				return ret;
			}
			
		}else{
			if (dis>=r2 && dis<= r1){
				double[] ret = {1,0};
				return ret;
			}else{
				double[] ret = {0,1};
				return ret;
			}
		}
	}

}
