package com.yunzhejia.cpxc.pattern;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import com.yunzhejia.cpxc.Discretizer;
import com.yunzhejia.cpxc.LocalClassifier;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class PatternSet extends AbstractList<Pattern> implements Serializable{
	private static final long serialVersionUID = -904353594925075971L;
	private List<Pattern> patterns;
	
	public PatternSet(){
		patterns = new ArrayList<Pattern>();
	}
	
	public PatternSet(PatternSet copy){
		patterns = new ArrayList<Pattern>(copy.patterns);
	}
	
	public PatternSet(List<Pattern> patterns){
		this.patterns = patterns;
	}

	public void readPatterns(String filename){
		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
			String line = br.readLine();
			while(line != null){
				Pattern pattern = Pattern.parsePattern(line);
				if(pattern!=null){
					patterns.add(pattern);
				}
				line = br.readLine();
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void contrast(Instances LE, Instances SE, Discretizer discretizer, double minRatio) {
		Iterator<Pattern> it = patterns.iterator();
		while(it.hasNext()){
			Pattern p = it.next();
			float supLE = p.supportOfData(LE, discretizer);
			float supSE = p.supportOfData(SE, discretizer);
			if (supLE/supSE < 2){
				it.remove();
			}
		}
	}
	
	public double TER(Instances data, AbstractClassifier baseClassifier, Discretizer discretizer) throws Exception{
		double TER = 0;
		double sum = 0;
		double totalErr = 0;
		for (Instance ins:data){
			double errB = getBaseError(baseClassifier, ins);
			double errM = error(ins,discretizer);
			if(errM > 0){
				sum += Math.abs(errB - errM);
//				System.out.println("ErrM="+errM+"  ErrB="+errB);
			}
			totalErr += errB;
		}
		if(totalErr > 0){
			TER = sum/totalErr;
		}
		return TER;
	}
	
	public boolean match(Instance ins, Discretizer discretizer){
		for (Pattern p:patterns){
			if (p.match(ins, discretizer) > 0){
				return true;
			}
		}
		return false;
	}
	
	public PatternSet filter(PatternFilter filter) throws Exception{
		return filter.filter(this);
	}
	
	private double error(Instance instance, Discretizer discretizer) throws Exception{
		boolean noMatch = true;
		double[] probs = new double[instance.numClasses()];
		int label = (int)instance.classValue();
		for (Pattern pattern: patterns){
			LocalClassifier ensemble = pattern.getLocalClassifier();
			if(pattern.match(instance, discretizer) > 0){
				int response = (int)ensemble.predict(instance);
				probs[response] += ensemble.getWeight()* pattern.match(instance, discretizer);
				noMatch = false;
			}
		}
		if(noMatch){
			return -1;
		}
		Utils.normalize(probs);
		return 1 - probs[label];
	}
	
	private double getBaseError(AbstractClassifier baseClassifier, Instance ins) throws Exception{
		int response = (int)ins.classValue();
		return 1-baseClassifier.distributionForInstance(ins)[response];
	}

	
	public List<Pattern> getPatterns(){
		return new ArrayList<>(patterns);
	}
	
	@Override
	public Pattern get(int index) {
		return patterns.get(index);
	}

	@Override
	public int size() {
		return patterns.size();
	}
	
	@Override
	public void add(int index, Pattern pattern){
		patterns.add(index, pattern);
	}
	
	@Override
	public Pattern set(int index, Pattern pattern){
		Pattern oldPattern = patterns.get(index);
		patterns.set(index, pattern);
		return oldPattern;
	}
	
	@Override
	public Pattern remove(int index){
		Pattern oldPattern = patterns.get(index);
		patterns.remove(index);
		return oldPattern;
	}
	
}
