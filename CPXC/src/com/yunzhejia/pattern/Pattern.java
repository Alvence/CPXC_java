/**
 * 
 */
package com.yunzhejia.pattern;

import java.util.Set;

import weka.core.Instance;
import weka.core.Instances;

/**
 * 
 * @author Yunzhe(Alvin) Jia, The University of Melbourne
 * @version 1.0
 *
 */
public class Pattern implements IPattern {
	private Set<ICondition> conditions;
	private Instances associatedData;
	private Instances mds;
	
	
	
	public Pattern(Set<ICondition> conditions) {
		super();
		this.conditions = conditions;
	}

	/**
	 * @param ins instance for testing
	 * @return return true if all conditions are true for the instance
	 */
	public boolean match(Instance ins){
		for(ICondition condition: conditions){
			if(!condition.match(ins)){
				return false;
			}
		}
		return true;
	}

	@Override
	public Instances matchingDataSet(Instances data) {
		if(data == associatedData){
			return mds;
		}else{
			findMds(data);
			return mds;
		}
	}

	@Override
	public double support(Instances data) {
		return matchingDataSet(data).size()*1.0/data.size();
	}
	
	private void findMds(Instances data){
		associatedData = data;
		mds = new Instances(data,0);
		for(Instance ins:data){
			if(this.match(ins)){
				mds.add(ins);
			}
		}
	}
}
