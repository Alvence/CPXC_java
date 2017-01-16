package com.yunzhejia.partition;

import java.util.List;
import java.util.Set;
import java.util.UUID;

import com.yunzhejia.pattern.IPattern;

import weka.classifiers.AbstractClassifier;
import weka.clusterers.AbstractClusterer;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

public class ClusterPartition implements IPartition {
	private String id;
	
	private boolean active;
	private double weight;
	
	private Instances data;
	private AbstractClusterer clusterer;
	private int label;
	private AbstractClassifier classifier;

	public ClusterPartition(AbstractClusterer clusterer, int label) {
		id = UUID.randomUUID().toString();
		this.clusterer = clusterer;
		this.label = label;
	}

	@Override
	public boolean match(Instance ins) {
		try {
			if(ins.classIndex()!=-1){
				Instances mdata = new Instances(data,0);
				mdata.add(ins);
				weka.filters.unsupervised.attribute.Remove filter = new weka.filters.unsupervised.attribute.Remove();
				filter.setAttributeIndices("" + (mdata.classIndex() + 1));
				filter.setInputFormat(mdata);
				Instances dataClusterer = Filter.useFilter(mdata, filter);
				ins = dataClusterer.get(0);
			}
			return (clusterer.clusterInstance(ins)==label);
		} catch (Exception e) {
			e.printStackTrace();
			return false;
		}
	}

	@Override
	public boolean isActive() {
		return active;
	}

	@Override
	public List<Set<IPattern>> getPatternSetList() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setPatternSetList(List<Set<IPattern>> patternSetList) {
		// TODO Auto-generated method stub

	}

	@Override
	public Instances getData() {
		return data;
	}

	@Override
	public void setData(Instances data) {
		this.data = data;
	}

	@Override
	public AbstractClassifier getClassifier() {
		return classifier;
	}

	@Override
	public void setClassifier(AbstractClassifier classifier) {
		this.classifier = classifier;
	}

	@Override
	public double getWeight() {
		return weight;
	}

	@Override
	public void setWeight(double weight) {
		this.weight=weight;

	}

	@Override
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
		ClusterPartition other = (ClusterPartition) obj;
		if (id == null) {
			if (other.id != null)
				return false;
		} else if (!id.equals(other.id))
			return false;
		return true;
	}
	
	@Override
	public String toString(){
		return "label = "+label+" weight="+weight+" active="+active+"  data size="+data.size();
	}
	

}
