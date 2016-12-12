package com.yunzhejia.pattern.patternmining;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.NominalCondition;
import com.yunzhejia.pattern.NumericCondition;
import com.yunzhejia.pattern.Pattern;
import com.yunzhejia.pattern.PatternSet;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ParallelCoordinatesMiner implements IPatternMiner, Serializable{
	private int NUMERIC_BIN_WIDTH = 10;
	
	public ParallelCoordinatesMiner(){
		this(10);
	}
	
	public ParallelCoordinatesMiner(int num_bin){
		this.NUMERIC_BIN_WIDTH = num_bin;
	}
	
	@Override
	public PatternSet minePattern(Instances data, double minSupp) {
		//1, get all the 
		PatternSet ps = new PatternSet();
		int numAttribute = data.numAttributes();
		calExtreme(data);
		for(int i = 0; i < numAttribute; i++){
			if (i==data.classIndex()){
				continue;
			}
			String attrName = data.attribute(i).name();
			if (data.attribute(i).isNumeric()){
//				double lower = data.attribute(i).getLowerNumericBound();
//				double upper = data.attribute(i).getUpperNumericBound();
				double lower = mins.get(i);
				double upper = maxs.get(i);
				double width = (upper-lower)/NUMERIC_BIN_WIDTH;
				double left = Double.MIN_VALUE;
				double right = lower + width;
				for (int bin = 0; bin < NUMERIC_BIN_WIDTH; bin++){
					if(bin == NUMERIC_BIN_WIDTH-1){
						right = Double.MAX_VALUE;
					}
					IPattern pattern = new Pattern(new NumericCondition(attrName,i, left, right));
					if (pattern.support(data)>= minSupp){
						ps.add(pattern);
					}
					left = right;
					right = right + width;
				}
			}else{
				Enumeration<Object> values = data.attribute(i).enumerateValues();
				while(values.hasMoreElements()){
					String value = values.nextElement().toString();
					IPattern pattern = new Pattern(new NominalCondition(attrName,i, value));
					if (pattern.support(data)>= minSupp){
						ps.add(pattern);
					}
				}
			}
		}
		
		return  ps;
	}
	
	private List<Double> mins;
	private List<Double> maxs;
	private void calExtreme(Instances data){
		mins = new ArrayList<Double>();
		maxs = new ArrayList<Double>();
		for (int i = 0; i < data.numAttributes(); i++){
			if (!data.attribute(i).isNumeric()){
				mins.add(0.0);
				maxs.add(0.0);
			}else{
				double min = Double.MAX_VALUE;
				double max = Double.MIN_VALUE;
				for(Instance in:data){
					if (in.value(i)<min){
						min = in.value(i);
					}
					if (in.value(i)>max){
						max = in.value(i);
					}
				}
				mins.add(min);
				maxs.add(max);
			}
		}
	}
	@Override
	public PatternSet minePattern(Instances data, double minSupp, int featureId) {
		//1, get all the 
				PatternSet ps = new PatternSet();
				int numAttribute = data.numAttributes();
				calExtreme(data);
				if (featureId==data.classIndex()){
					return null;
				}
				String attrName = data.attribute(featureId).name();
				if (data.attribute(featureId).isNumeric()){
//					double lower = data.attribute(i).getLowerNumericBound();
//					double upper = data.attribute(i).getUpperNumericBound();
					double lower = mins.get(featureId);
					double upper = maxs.get(featureId);
					double width = (upper-lower)/NUMERIC_BIN_WIDTH;
					double left = Double.MIN_VALUE;
					double right = lower + width;
					for (int bin = 0; bin < NUMERIC_BIN_WIDTH; bin++){
						if(bin == NUMERIC_BIN_WIDTH-1){
							right = Double.MAX_VALUE;
						}
						IPattern pattern = new Pattern(new NumericCondition(attrName,featureId, left, right));
						if (pattern.support(data)>= minSupp){
							ps.add(pattern);
						}
						left = right;
						right = right + width;
				}
				}else{
					Enumeration<Object> values = data.attribute(featureId).enumerateValues();
					while(values.hasMoreElements()){
						String value = values.nextElement().toString();
						IPattern pattern = new Pattern(new NominalCondition(attrName,featureId, value));
						if (pattern.support(data)>= minSupp){
							ps.add(pattern);
						}
					}
				}
				
				return  ps;
	}
	
	public static void main(String[] args){
		DataSource source;
		Instances data;
		try {
//			source = new DataSource("data/synthetic1.arff");
			source = new DataSource("data/ILPD.arff");
			data = source.getDataSet();
			if (data.classIndex() == -1){
				data.setClassIndex(data.numAttributes() - 1);
			}
			ParallelCoordinatesMiner miner = new ParallelCoordinatesMiner();
			PatternSet ps = miner.minePattern(data, 0.1);
			System.out.println(ps.size());
			for(IPattern p:ps){
				System.out.println(p+"  support="+p.support(data));
			}
			 
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	
}
