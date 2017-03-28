package com.yunzhejia.pattern;

import java.io.Serializable;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.List;


import weka.core.Instance;
import weka.core.Instances;

public class PatternSet extends AbstractList<IPattern> implements Serializable{
	private static final long serialVersionUID = -904353594925075971L;
	private List<IPattern> patterns;
	
	public PatternSet(){
		patterns = new ArrayList<IPattern>();
	}
	
	public PatternSet(PatternSet copy){
		patterns = new ArrayList<IPattern>(copy.patterns);
	}
	
	public PatternSet(List<IPattern> patterns){
		this.patterns = patterns;
	}

	
	
	public boolean match(Instance ins){
		for (IPattern p:patterns){
			if (p.match(ins)){
				return true;
			}
		}
		return false;
	}
	/*
	public PatternSet filter(PatternFilter filter) throws Exception{
		return filter.filter(this);
	}*/
	
	public PatternSet getMatchingPatterns(Instance instance){
		PatternSet ret = new PatternSet();
		for (IPattern pattern : this.patterns){
			if (!pattern.match(instance)){
				ret.add(pattern);
			}
		}
		return ret;
	}
	
	public Instances getNoMatchingData(Instances data){
		Instances ret = new Instances(data,0);
		for (Instance ins : data){
			if (!this.match(ins)){
				ret.add(ins);
			}
		}
		return ret;
	}
	
	public Instances getMatchingData(Instances data){
		Instances ret = new Instances(data,0);
		for (Instance ins : data){
			if (this.match(ins)){
				ret.add(ins);
			}
		}
		return ret;
	}

	
	public List<IPattern> getPatterns(){
		return new ArrayList<>(patterns);
	}
	
	@Override
	public IPattern get(int index) {
		return patterns.get(index);
	}

	@Override
	public int size() {
		return patterns.size();
	}
	
	@Override
	public void add(int index, IPattern pattern){
		patterns.add(index, pattern);
	}
	
	@Override
	public IPattern set(int index, IPattern pattern){
		IPattern oldPattern = patterns.get(index);
		patterns.set(index, pattern);
		return oldPattern;
	}
	
	@Override
	public IPattern remove(int index){
		IPattern oldPattern = patterns.get(index);
		patterns.remove(index);
		return oldPattern;
	}
	
}
