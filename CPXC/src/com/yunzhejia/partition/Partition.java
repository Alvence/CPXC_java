package com.yunzhejia.partition;

import java.util.List;
import java.util.Set;
import java.util.UUID;

import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.pattern.IPattern;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class Partition implements IPartition{
	private List<Set<IPattern>> patternSetList;
	private Instances data;
	private AbstractClassifier classifier;
	private double weight;
	private boolean active;
	private String id;
	
	public Partition(){
		id = UUID.randomUUID().toString();
		active = true;
	}
	
	public boolean match(Instance ins){
		for (Set<IPattern> patternSet:patternSetList){
			if (match(ins, patternSet)){
				return true;
			}
		}
		return false;
	}

	private boolean match(Instance ins, Set<IPattern> patterns){
		for (IPattern p:patterns){
			if(!p.match(ins)){
				return false;
			}
		}
		return true;
	}
	public boolean isActive(){
		return active;
	}
	
	
	
	public List<Set<IPattern>> getPatternSetList() {
		return patternSetList;
	}

	public void setPatternSetList(List<Set<IPattern>> patternSetList) {
		this.patternSetList = patternSetList;
	}

	public Instances getData() {
		return data;
	}

	public void setData(Instances data) {
		this.data = data;
	}

	public AbstractClassifier getClassifier() {
		return classifier;
	}

	public void setClassifier(AbstractClassifier classifier) {
		this.classifier = classifier;
	}

	public double getWeight() {
		return weight;
	}

	public void setWeight(double weight) {
		this.weight = weight;
	}

	public void setActive(boolean active) {
		this.active = active;
	}

	
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((id == null) ? 0 : id.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Partition other = (Partition) obj;
		if (id == null) {
			if (other.id != null)
				return false;
		} else if (!id.equals(other.id))
			return false;
		return true;
	}

	@Override
	public String toString(){
		String ret = "";
		for (Set<IPattern> patternSet:patternSetList){
			ret+="{";
			for (IPattern p:patternSet){
				ret+= p.toString()+" ";
			}
			ret+="}";
		}
		ret+=" weight ="+ weight;
		ret+=" data size="+data.size();
		return ret;
	}

}
