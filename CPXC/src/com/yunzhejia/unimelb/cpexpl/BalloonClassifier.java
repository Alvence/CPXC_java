package com.yunzhejia.unimelb.cpexpl;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class BalloonClassifier extends AbstractClassifier {

	@Override
	public void buildClassifier(Instances data) throws Exception {

	}
	
	@Override
	public double[] distributionForInstance(Instance instance)throws Exception{
		/*if (instance.stringValue(0).equals("1")){ // act == STRETCH, age = ADULT
			if(instance.stringValue(3).equals("STRETCH") && instance.stringValue(4).equals("ADULT")){
				double[] ret = {1,0};
				return ret;
			}else if(instance.stringValue(3).equals("STRETCH") || instance.stringValue(4).equals("ADULT")){
				double[] ret = {0.3,0.7};
				
				return ret;
			}else {
				double[] ret = {0,1};
				return ret;
			}
			
		}else{
			if(instance.stringValue(1).equals("YELLOW") && instance.stringValue(2).equals("SMALL")){
				double[] ret = {1,0};
				return ret;
			}else if(instance.stringValue(1).equals("YELLOW") || instance.stringValue(2).equals("SMALL")){
				double[] ret = {0.3,0.7};
				return ret;
			}else{
				double[] ret = {0,1};
				return ret;
			}
		}*/
			if(instance.stringValue(3).equals("STRETCH") && instance.stringValue(4).equals("ADULT")){
				double[] ret = {1,0};
				return ret;
			
			}
			
		else if(instance.stringValue(1).equals("YELLOW") && instance.stringValue(2).equals("SMALL")){
				double[] ret = {1,0};
				return ret;
			}else{
				double[] ret = {0,1};
				return ret;
			}
		
	}

}
