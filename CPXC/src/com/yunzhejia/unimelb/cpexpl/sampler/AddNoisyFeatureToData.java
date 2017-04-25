package com.yunzhejia.unimelb.cpexpl.sampler;

import java.util.Random;

import com.yunzhejia.cpxc.util.DataUtils;

import weka.core.Attribute;
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
		try {
			Instances data = DataUtils.load("data/balloon_synthetic.arff");
			Random rand = new Random(0);
			int numOfNewAttr = data.numAttributes()/2;
			
			for(int i = 0 ; i < numOfNewAttr; i++){
			
				Attribute newAttr= new Attribute("N"+i);
				data.insertAttributeAt(newAttr, data.numAttributes()-1);
			
				for (int j = 0; j < data.numInstances();j++){
					data.get(j).setValue(data.numAttributes()-2, rand.nextInt(10));
				}
				data.randomize(new Random(1));
			}
			
			DataUtils.save(data, "tmp/newData.arff");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
