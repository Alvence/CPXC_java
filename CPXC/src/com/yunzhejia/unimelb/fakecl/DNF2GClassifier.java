package com.yunzhejia.unimelb.fakecl;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class DNF2GClassifier extends AbstractClassifier {

	@Override
	public void buildClassifier(Instances data) throws Exception {

	}
	
	@Override
	public double[] distributionForInstance(Instance instance)throws Exception{
		
		if (instance.stringValue(0).equals("1")&&instance.stringValue(1).equals("1")){ // act == STRETCH, age = ADULT
			if(instance.stringValue(2).equals("1") && instance.stringValue(3).equals("1")&& instance.stringValue(4).equals("1")){
				double[] ret = {0,1};
				return ret;
			}else if(instance.stringValue(2).equals("1") || instance.stringValue(3).equals("1") || instance.stringValue(4).equals("1")){
				double[] ret = {0.7,0.3};
				return ret;
			}else {
				double[] ret = {1,0};
				return ret;
			}
			
		}else if (instance.stringValue(0).equals("0")&&instance.stringValue(1).equals("1")){
			if(instance.stringValue(5).equals("1") && instance.stringValue(6).equals("1")&& instance.stringValue(7).equals("1")){
				double[] ret = {0,1};
				return ret;
			}else if(instance.stringValue(5).equals("1") || instance.stringValue(6).equals("1") || instance.stringValue(7).equals("1")){
				double[] ret = {0.7,0.3};
				return ret;
			}else {
				double[] ret = {1,0};
				return ret;
			}
		}else {
			
				double[] ret = {1,0};
				return ret;
			
		}
	}

}
