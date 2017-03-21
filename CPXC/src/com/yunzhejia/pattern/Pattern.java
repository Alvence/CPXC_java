/**
 * 
 */
package com.yunzhejia.pattern;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.UUID;

import weka.core.Instance;
import weka.core.Instances;

/**
 * 
 * @author Yunzhe(Alvin) Jia, The University of Melbourne
 * @version 1.0
 *
 */
public class Pattern implements IPattern {
	private String id;
	private Set<ICondition> conditions;
	private Instances associatedData;
	private Instances mds;
	
	private Pattern(){}
	public Pattern(ICondition condition) {
		id = UUID.randomUUID().toString();
		this.conditions = new HashSet<>();
		this.conditions.add(condition);
	}
	static int c = 1;
	public Pattern(Set<ICondition> conditions) {
		for(ICondition con:conditions){
			addCondition(con);
		}
//		this.conditions = conditions;
//		Pattern p = new Pattern();
//		
//		p.conditions = conditions;
//		System.out.println((c)+":old pattern: " + p);
//		System.out.println((c++)+":new pattern: " + this+"\n");
		
	}
	

	private void addCondition(ICondition condition){
		if (this.conditions==null){
			this.conditions = new HashSet<>();
		}
		if (condition instanceof NumericCondition){
			NumericCondition numCon = (NumericCondition)condition;
			boolean flag = false;
			Iterator<ICondition> it = conditions.iterator();
			ICondition newCondition = null;
			while(it.hasNext()){
				ICondition existedCon = (ICondition)it.next();
				if (existedCon instanceof NumericCondition){
					NumericCondition numExistedCon = (NumericCondition)existedCon;
					if(numExistedCon.attrIndex == numCon.attrIndex){
						if(numCon.left<=numExistedCon.right && numCon.right >= numExistedCon.left){
							double newleft = numCon.left>numExistedCon.left?numCon.left:numExistedCon.left;
							double newright = numCon.right<numExistedCon.right?numCon.right:numExistedCon.right;
							newCondition = new NumericCondition(numExistedCon.attrName,numExistedCon.attrIndex,newleft,newright);
							flag = true;
							it.remove();
							break;
						}
					}
				}
			}
			if(!flag){
				newCondition = condition;
			}
			this.conditions.add(newCondition);
		}else{
			this.conditions.add(condition);
		}
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
	
	@Override
	public String toString(){
		String ret = "";
		for (ICondition condition:conditions){
			ret += condition +"   ";
		}
		return ret;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((conditions == null) ? 0 : conditions.hashCode());
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
		if (conditions == null) {
			if (other.conditions != null)
				return false;
		} else if (!conditions.equals(other.conditions))
			return false;
		return true;
	}
	@Override
	public double support() {
		return mds.size()*1.0/associatedData.size();
	}

	
	
}
