package com.yunzhejia.unimelb.cpexpl.truth;

import java.util.HashSet;
import java.util.Set;

import com.yunzhejia.cpxc.util.DataUtils;

import weka.core.Instance;
import weka.core.Instances;

public class BalloonExpl {

	public static void main(String[] args) {
		try {
			Instances data = DataUtils.load("data/synthetic/DNF3G_test.arff");
			for(Instance ins:data){
				System.out.println(getDNF3GGoldFeature(ins).toString().replace("[", "").replace("]","").replace(" ", ""));
			}
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static Set<Integer> getBalloonGoldFeature(Instance instance){
		Set<Integer> ret = new HashSet<>();
		ret.add(0);
		if (instance.stringValue(0).equals("1")){ // act == STRETCH, age = ADULT
			ret.add(3);
			ret.add(4);
		}else if (instance.stringValue(0).equals("2")){
			ret.add(1);
			ret.add(2);
		}
		return ret;
	}
	
	public static Set<Integer> getDNF9GoldFeature(Instance instance){
		Set<Integer> ret = new HashSet<>();
		ret.add(0);
		if (instance.stringValue(0).equals("1")){ // act == STRETCH, age = ADULT
			ret.add(1);
			ret.add(2);
		}else if (instance.stringValue(0).equals("2")){
			ret.add(3);
			ret.add(4);
		}else if (instance.stringValue(0).equals("3")){
			ret.add(5);
			ret.add(6);
		}else if (instance.stringValue(0).equals("4")){
			ret.add(7);
			ret.add(8);
		}
		
		return ret;
	}
	
	public static Set<Integer> getDNF2GGoldFeature(Instance instance){
		Set<Integer> ret = new HashSet<>();
		ret.add(0);
		ret.add(1);
		if(instance.stringValue(0).equals("1") && instance.stringValue(1).equals("1")){
			ret.add(2);
			ret.add(3);
			ret.add(4);
		}
		else if(instance.stringValue(0).equals("0") && instance.stringValue(1).equals("1")){
			ret.add(5);
			ret.add(6);
			ret.add(7);
		}
		return ret;
	}
	
	public static Set<Integer> getDNF3GGoldFeature(Instance instance){
		Set<Integer> ret = new HashSet<>();
		ret.add(0);
		ret.add(1);
		ret.add(2);
		if (instance.stringValue(0).equals("0") && instance.stringValue(1).equals("0")&& instance.stringValue(2).equals("0")){
			ret.add(3);
			ret.add(4);
			
		}else if (instance.stringValue(0).equals("0") && instance.stringValue(1).equals("0")&& instance.stringValue(2).equals("1")){
			ret.add(5);
			ret.add(6);
		}else if (instance.stringValue(0).equals("0") && instance.stringValue(1).equals("1")&& instance.stringValue(2).equals("0")){
			ret.add(7);
			ret.add(8);
		}else if (instance.stringValue(0).equals("0") && instance.stringValue(1).equals("1")&& instance.stringValue(2).equals("1")){
			ret.add(9);
			ret.add(10);
		} else if (instance.stringValue(0).equals("1") && instance.stringValue(1).equals("0")&& instance.stringValue(2).equals("0")){
			ret.add(11);
			ret.add(12);
		}else if (instance.stringValue(0).equals("1") && instance.stringValue(1).equals("0")&& instance.stringValue(2).equals("1")){
			ret.add(13);
			ret.add(14);
		}else if (instance.stringValue(0).equals("1") && instance.stringValue(1).equals("1")&& instance.stringValue(2).equals("0")){
			ret.add(15);
			ret.add(16);
		}else {
			ret.add(17);
			ret.add(18);
		}
		
		return ret;
	}

}
