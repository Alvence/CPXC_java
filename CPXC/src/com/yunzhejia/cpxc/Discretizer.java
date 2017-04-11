package com.yunzhejia.cpxc;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import com.yunzhejia.cpxc.util.OutputUtils;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class Discretizer implements Serializable{
	private static final long serialVersionUID = 2555190450596422877L;

	public static int SHIFT_SIZE = 10;
	
	private int defaultBin;
	
	private HashMap<Integer, List<Double>> cuttingPoints;
	
	private HashMap<Integer, List<String>> nominals;
	
	public Discretizer(){
		this(10);
	}
	
	public Discretizer(int defaultBin){
		this.defaultBin = defaultBin;
	}
	
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
		
//		OutputUtils.print(nominalAttributes);
//		for(int attr: nominals.keySet()){
//			for(String val: nominals.get(attr)){
//				System.out.println(attr+"  "+val);
//			}
//		}
//		
	}
	
	public boolean isNumeric(int attrIndex){
		return !nominals.containsKey(new Integer(attrIndex));
	}
	
	public int getAttributeIndex(int discretizedValue){
		return (discretizedValue >> SHIFT_SIZE);
	}
	
	public int getValue(int discretizedValue){
		return (discretizedValue & ((1 << SHIFT_SIZE) -1));
	}
	
	public String getNominal(int discretizedValue){
		int attr = getAttributeIndex(discretizedValue);
		int val = getValue(discretizedValue);
		return nominals.get(attr).get(val);
	}
	
	public double getLeft(int discretizedValue){
		int attr = getAttributeIndex(discretizedValue);
		int val = getValue(discretizedValue);
		if(val == 0){
			return Double.MIN_VALUE;
		}
		return cuttingPoints.get(attr).get(val-1);
	}
	
	public double getRight(int discretizedValue){
		int attr = getAttributeIndex(discretizedValue);
		int val = getValue(discretizedValue);
		if(val ==  cuttingPoints.get(attr).size()){
			return Double.MAX_VALUE;
		}
		return cuttingPoints.get(attr).get(val);
	}
	
	
	public String getDiscretizedInstance(Instance ins){
		return getDiscretizedInstance(ins," ");
	}
	
	public String getDiscretizedInstanceKeepMissingValue(Instance ins, String delimeter){
		String ret = "";
		for (int j = 0; j < ins.numAttributes(); j++){
			if (j == ins.classIndex() ){
				continue;
			}
			if(ins.isMissing(j)){
				ret += "?"+ (j==ins.numAttributes()-2?"":delimeter);
				continue;
			}
			if (ins.attribute(j).isNumeric()){
				ret += getDiscretizedValue(j, ins.value(j))+(j<<SHIFT_SIZE)+ (j==ins.numAttributes()-2?"":delimeter);
			}else{
				ret += getDiscretizedValue(j, ins.stringValue(j))+(j<<SHIFT_SIZE)+ (j==ins.numAttributes()-2?"":delimeter);
			}
		}
		return ret;
	}
	
	public String getDiscretizedInstance(Instance ins, String delimeter){
		String ret = "";
		for (int j = 0; j < ins.numAttributes(); j++){
			if (j == ins.classIndex() || ins.isMissing(j)){
				continue;
			}
			if (ins.attribute(j).isNumeric()){
				ret += getDiscretizedValue(j, ins.value(j))+(j<<SHIFT_SIZE)+ (j==ins.numAttributes()-2?"":delimeter);
			}else{
				ret += getDiscretizedValue(j, ins.stringValue(j))+(j<<SHIFT_SIZE)+ (j==ins.numAttributes()-2?"":delimeter);
			}
		}
		return ret;
	}
	
	public int getShiftedDiscretizedValue(int attrIndex, Object attrVal){
		return getDiscretizedValue(attrIndex, attrVal)+(attrIndex<<SHIFT_SIZE);
	}
	
	public int getDiscretizedValue(int attrIndex, Object attrVal){
		if (!nominals.keySet().contains(attrIndex)){
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
		discretizer.setBins(defaultBin);
		weka.filters.supervised.attribute.Discretize.useFilter(data, discretizer);
		
		for(int index: cuttingPointAttributes){
//			System.out.println("attr" + index + "  " + Arrays.toString(discretizer.getCutPoints(index)));
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
