package com.yunzhejia.cpxc.util;

import java.util.Set;

import weka.core.Instance;
import weka.core.Instances;

public class OverlapCalculation {
	public static void calcOverlap(Instances data1, Instances data2) throws Exception{
		DataUtils.save(data1, "tmp/d1.arff");
		DataUtils.save(data2, "tmp/d2.arff");
		Instances overlap;
		boolean[] flags = new boolean[data2.size()];
		for(int i = 0; i < flags.length; i++){
			flags[i]= true;
		}
		double c = 0;
		for(Instance ins: data1){
			for(int i = 0; i < data2.size();i++){
				if(equal(data2.get(i),ins) && flags[i]){
					c+=1;
					flags[i] = false;
				}
			}
		}
		System.out.println(Double.toString(c/data1.size()));
	}
	
	public static boolean equal(Instance ins1, Instance ins2){
		if(!ins1.equalHeaders(ins2)){
//			System.out.println("1");
			return false;
		}
		
		for(int i = 0; i < ins1.numAttributes(); i++){
			if((ins1.isMissing(i) && !ins2.isMissing(i))
					||ins2.isMissing(i) && !ins1.isMissing(i)){
//				System.out.println("2");
				return false;
			} else{
				if (ins1.attribute(i).isNumeric()){
					if((int)(ins1.value(i)* 100) != (int) (ins2.value(i)*100)){
//						System.out.println("311");
						return false;
					}
				}else{
					if (!ins1.stringValue(i).equals(ins2.stringValue(i))){
//						System.out.println("4");
						return false;
					}
				}
			}
		}
		
		return true;
	}
	
	public static void calcOverlap(Set<Integer> data1, Set<Integer>  data2) throws Exception{
		double c = 0;
		for(int ins: data1){
			if(data2.contains(ins)){
				c+=1;
			}
		}
		System.out.println(Double.toString(c/data1.size()));
	}
	
	public static void main(String[] args){
		try {
			Instances data = DataUtils.load("data/iris.arff");
			Instances data2 = DataUtils.load("data/iris.arff");
			data2.get(0).setValue(1, 124);
			System.out.println(data.size());
			calcOverlap(data,data2);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
}
