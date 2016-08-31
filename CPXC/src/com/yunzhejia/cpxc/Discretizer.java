package com.yunzhejia.cpxc;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class Discretizer implements Serializable{
	private static final long serialVersionUID = 2555190450596422877L;

	public static int SHIFT_SIZE = 10;
	
	private HashMap<Integer, List<Double>> cuttingPoints;
	
	private HashMap<Integer, List<String>> nominals;
	
	public void initialize(Instances data) throws Exception{
		int[] nominalAttributes = nominalAttributeIndex(data);
		int[] nocuttingPointAttributes = noCuttingPointAttributeIndex(data);
		int[] cuttingPointAttributes = cuttingPointAttributeIndex(data);
		
		
//		OutputUtils.print(nominalAttributes);
//		OutputUtils.print(nocuttingPointAttributes);
//		OutputUtils.print(cuttingPointAttributes);
		
		cuttingPoints = new HashMap<Integer,List<Double>>();
		calculateNominal(data, nominalAttributes);
		//System.out.println(nominals);
		calculateCuttingPoints(data, cuttingPointAttributes);
		calculateCuttingPointsEq(data, nocuttingPointAttributes);
		//System.out.println(cuttingPoints);
		
		
	}
	
	public String getDiscretizedInstance(Instance ins){
		String ret = "";
		for (int j = 0; j < ins.numAttributes(); j++){
			if (j == ins.classIndex()){
				continue;
			}
			if (ins.attribute(j).isNumeric()){
				ret += getDiscretizedValue(j, ins.value(j))+(j<<SHIFT_SIZE)+" ";
			}else{
				ret += getDiscretizedValue(j, ins.stringValue(j))+(j<<SHIFT_SIZE)+" ";
			}
		}
		return ret;
	}
	
	public int getShiftedDiscretizedValue(int attrIndex, Object attrVal){
		return getDiscretizedValue(attrIndex, attrVal)+(attrIndex<<SHIFT_SIZE);
	}
	
	public int getDiscretizedValue(int attrIndex, Object attrVal){
		if (cuttingPoints.keySet().contains(attrIndex)){
			return getNumericDiscretizedValue(attrIndex, attrVal);
		} else{
			return getNominalDiscretizedValue(attrIndex, attrVal);
		}
	}
	
	
	private int getNumericDiscretizedValue(int attrIndex, Object attrVal){
		List<Double> points = cuttingPoints.get(attrIndex);
		int index = 0;
		while (index < points.size() && points.get(index) < (Double)attrVal){
			index++;
		}
		return index;
	}
	
	private int getNominalDiscretizedValue(int attrIndex, Object attrVal){
		List<String> points = nominals.get(attrIndex);
		int index = 0;
		while (index < points.size() && (!points.get(index).equals(attrVal))){
			index++;
		}
		return index;
	}
	
	private void calculateCuttingPointsEq(Instances data, int[] cuttingPointAttributes) throws Exception {
		weka.filters.unsupervised.attribute.Discretize discretizer = new weka.filters.unsupervised.attribute.Discretize();
		discretizer.setAttributeIndicesArray(cuttingPointAttributes);
		discretizer.setInputFormat(data);
		discretizer.setBins(3);
		weka.filters.supervised.attribute.Discretize.useFilter(data, discretizer);
		
		for(int index: cuttingPointAttributes){
			List<Double> points = new ArrayList<Double>();
			for (double point: discretizer.getCutPoints(index)){
				points.add(point);
			}
			cuttingPoints.put(index, points);
		}
	}
	
	private void calculateCuttingPoints(Instances data, int[] cuttingPointAttributes) throws Exception {
		weka.filters.supervised.attribute.Discretize discretizer = new weka.filters.supervised.attribute.Discretize();
		
		discretizer.setAttributeIndicesArray(cuttingPointAttributes);
		discretizer.setInputFormat(data);
		weka.filters.supervised.attribute.Discretize.useFilter(data, discretizer);
		
		for(int index: cuttingPointAttributes){
			List<Double> points = new ArrayList<Double>();
			for (double point: discretizer.getCutPoints(index)){
				points.add(point);
			}
			cuttingPoints.put(index, points);
		}
	}

	private void calculateNominal(Instances data, int[] nominalAttributes) {
		nominals = new HashMap<Integer, List<String>>();
		for (int index : nominalAttributes){
			List<String> values = new ArrayList<String>();
			for (int valIndex = 0; valIndex < data.attribute(index).numValues(); valIndex++){
				values.add(data.attribute(index).value(valIndex));
			}
			nominals.put(index, values);
		}
	}

	private int[] nominalAttributeIndex(Instances data){
		List<Integer> list = new ArrayList<Integer>();
		for (int i = 0; i < data.numAttributes(); i++){
			if (data.attribute(i).type() == Attribute.NOMINAL){
				list.add(i);
			}
		}
		int[] ret = new int[list.size()];
		for(int i = 0; i < ret.length; i++){
			ret[i] = list.get(i);
		}
		return ret;
	}
	
	private int[] cuttingPointAttributeIndex(Instances data) throws Exception{
		List<Integer> list = new ArrayList<Integer>();
		weka.filters.supervised.attribute.Discretize discretizer = new weka.filters.supervised.attribute.Discretize();
		
		discretizer.setInputFormat(data);
		weka.filters.supervised.attribute.Discretize.useFilter(data, discretizer);
		
		for (int i = 0; i < data.numAttributes(); i++){
			double [] cuttings = discretizer.getCutPoints(i);
			if(cuttings!=null && data.attribute(i).type() == Attribute.NUMERIC){
				list.add(i);
			}
		}
		int[] ret = new int[list.size()];
		for(int i = 0; i < ret.length; i++){
			ret[i] = list.get(i);
		}
		return ret;
	}
	
	private int[] noCuttingPointAttributeIndex(Instances data) throws Exception{
		List<Integer> list = new ArrayList<Integer>();
		weka.filters.supervised.attribute.Discretize discretizer = new weka.filters.supervised.attribute.Discretize();
		
		discretizer.setInputFormat(data);
		weka.filters.supervised.attribute.Discretize.useFilter(data, discretizer);
		
		for (int i = 0; i < data.numAttributes(); i++){
			double [] cuttings = discretizer.getCutPoints(i);
			if(cuttings==null && data.attribute(i).type() == Attribute.NUMERIC){
				list.add(i);
			}
		}
		int[] ret = new int[list.size()];
		for(int i = 0; i < ret.length; i++){
			ret[i] = list.get(i);
		}
		return ret;
	}
	
}
