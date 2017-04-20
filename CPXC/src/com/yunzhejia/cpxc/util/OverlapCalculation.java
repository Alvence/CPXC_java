package com.yunzhejia.cpxc.util;

import weka.core.Instance;
import weka.core.Instances;

public class OverlapCalculation {
	public static void calcOverlap(Instances data1, Instances data2) throws Exception{
		DataUtils.save(data1, "tmp/d1");
		DataUtils.save(data2, "tmp/d2");
		Instances overlap;
		double c = 0;
		for(Instance ins: data1){
			if(data2.contains(ins)){
				c+=1;
			}
		}
		System.out.println(Double.toString(c/data1.size()));
	}
}
