package com.yunzhejia.cpxc.pattern;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import com.yunzhejia.cpxc.Discretizer;
import com.yunzhejia.cpxc.LocalClassifier;

import weka.core.Instance;
import weka.core.Instances;

public class Pattern implements Serializable{
	private static final long serialVersionUID = -6857238961341042594L;
	private String id;
	private List<Integer> items;
	private int support;
	private LocalClassifier localClassifier = null;
	
	public Pattern(){
		id = UUID.randomUUID().toString();
		items = new ArrayList<Integer>();
	}
	
	public double match(Instance ins, Discretizer discretizer){
		List<Integer> insItems = new ArrayList<Integer>();
		for (int i = 0; i < ins.numAttributes(); i++){
			if (i == ins.classIndex()){
				continue;
			}
			int item = -1;
			if (ins.attribute(i).isNumeric()){
				item = discretizer.getShiftedDiscretizedValue(i, ins.value(i));
			}else{
				item = discretizer.getShiftedDiscretizedValue(i, ins.stringValue(i));
			}
			insItems.add(item);
		}
		//System.out.println(items+" ins="+insItems+"  "+insItems.containsAll(items));
		return insItems.containsAll(items)?1:0;
	}
	
	
	
	public void setLocalClassifier(LocalClassifier localClassifier) {
		this.localClassifier = localClassifier;
	}

	public LocalClassifier getLocalClassifier(){
		return localClassifier;
	}
	
	public int getSupport(){
		return support;
	}
	
	public Instances getMatches(Instances data, Discretizer discretizer){
		Instances mds = new Instances(data,0);
		for (Instance ins:data){
			if (this.match(ins, discretizer) > 0){
				mds.add(ins);
			}
		}
		return mds;
		
	}
	
	public int supportOfData(Instances data, Discretizer discretizer){
		int count = 0;
		for (Instance ins:data){
			if (this.match(ins, discretizer) > 0){
				count++;
			}
		}
		return count;
	}
	
	public static Pattern parsePattern(String str){
		Pattern pattern = new Pattern();
		String[] tokens = str.split(" ");
		if(tokens.length==2){
			return null;
		}
		for (int i = 0; i < tokens.length; i++){
			//ignore first
			if ( 0 == i){
				continue;
			}else if(tokens.length - 1 == i){
				pattern.support = Integer.parseInt(tokens[i]);
			}else{
				pattern.items.add(Integer.parseInt(tokens[i]));
			}
		}
		return pattern;
	}
	
	@Override
	public String toString(){
		StringBuilder str = new StringBuilder();
		for (Integer item:items){
			str.append(item+" ");
		}
		str.append("support = ");
		str.append(support);
		str.append("weight = ");
		str.append(localClassifier==null?null:localClassifier.getWeight());
		return str.toString();
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
		Pattern other = (Pattern) obj;
		if (id == null) {
			if (other.id != null)
				return false;
		} else if (!id.equals(other.id))
			return false;
		return true;
	}
	
	
}
