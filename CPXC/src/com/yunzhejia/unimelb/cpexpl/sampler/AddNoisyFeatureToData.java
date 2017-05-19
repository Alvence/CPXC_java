package com.yunzhejia.unimelb.cpexpl.sampler;

import java.util.Iterator;
import java.util.Random;

import com.yunzhejia.cpxc.util.DataUtils;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class AddNoisyFeatureToData {
	
	public static Instances generateNoisyData(Instances data){
		Instances ret = new Instances(data);
		try {
			Random rand = new Random(0);
			int numOfNewAttr = ret.numAttributes();
			
			for(int i = 0 ; i < numOfNewAttr; i++){
			
				Attribute newAttr= new Attribute("N"+i);
				ret.insertAttributeAt(newAttr, ret.numAttributes()-1);
			
				for (int j = 0; j < ret.numInstances();j++){
					ret.get(j).setValue(ret.numAttributes()-2, rand.nextInt(100));
				}
			}
			
			DataUtils.save(ret, "tmp/newData.arff");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return ret;
	}
	

	public static void main(String[] args) {
		biasedNoise();
	}
	
	public static void biasedNoise(){
		try {
			Instances data = DataUtils.load("data/synthetic/DNF2G_train.arff");
			Random rand = new Random(10);
			int numOfNewAttr = 10;
			
			for(int i = 0 ; i < numOfNewAttr; i++){
			
				Attribute newAttr= new Attribute("N"+i);
				data.insertAttributeAt(newAttr, data.numAttributes()-1);
			
				for (int j = 0; j < data.numInstances();j++){
					if (i == 2|| i ==4){
						if(data.get(j).classValue()==0){
							data.get(j).setValue(data.numAttributes()-2, rand.nextInt(10));
						}else{
							data.get(j).setValue(data.numAttributes()-2, rand.nextInt(50)+50);
						}
					}else
						data.get(j).setValue(data.numAttributes()-2, rand.nextInt(100));
				}
			}
//			Iterator<Instance> it = data.iterator();
//			while(it.hasNext()){
//				Instance ins = it.next();
//				if(ins.classValue() == 0){
//					it.remove();
//				}
//			}
			DataUtils.save(data, "data/synthetic/DNF2G_biasednoisy_train.arff");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void randomNoise(){
		try {
			Instances data = DataUtils.load("data/synthetic/DNF9_test.arff");
			Random rand = new Random(10);
			int numOfNewAttr = 10;
			
			for(int i = 0 ; i < numOfNewAttr; i++){
			
				Attribute newAttr= new Attribute("N"+i);
				data.insertAttributeAt(newAttr, data.numAttributes()-1);
			
				for (int j = 0; j < data.numInstances();j++){
					data.get(j).setValue(data.numAttributes()-2, rand.nextInt(100));
				}
				data.randomize(new Random(1));
			}
//			Iterator<Instance> it = data.iterator();
//			while(it.hasNext()){
//				Instance ins = it.next();
//				if(ins.classValue() == 0){
//					it.remove();
//				}
//			}
			DataUtils.save(data, "data/synthetic/DNF9_noisy_test.arff");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
