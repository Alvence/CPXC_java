package com.yunzhejia.cpxc.util;

import java.util.ArrayList;
import java.util.List;

public class ArrayUtils {

	private ArrayUtils() {
		// TODO Auto-generated constructor stub
	}
	
	public static List<Double> arrayToList(double[] arr){
		List<Double> ret = new ArrayList<>();
		for (double element:arr){
			ret.add(element);
		}
		return ret;
	}
	
	public static void normalize(List<Double> list){
		double sum = 0;
	    for (double d : list) {
	      sum += d;
	    }
	    
	    normalize(list,sum);
	}
	
	public static void normalize(List<Double> list, double sum) {

	    if (Double.isNaN(sum)) {
	      throw new IllegalArgumentException("Can't normalize array. Sum is NaN.");
	    }
	    if (sum == 0) {
	      // Maybe this should just be a return.
	      throw new IllegalArgumentException("Can't normalize array. Sum is zero.");
	    }
	    for (int i = 0; i < list.size(); i++) {
	    	list.set(i, list.get(i)/sum);
	    }
	  }

}
