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
	private double ratio;
	
	private Pattern(){}
	public Pattern(ICondition condition) {
		id = UUID.randomUUID().toString();
		this.conditions = new HashSet<>();
		this.conditions.add(condition);
	}
	static int c = 1;
	public Pattern(Set<ICondition> conditions) {
		id = UUID.randomUUID().toString();
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
			boolean flag = true;
			Iterator<ICondition> it = conditions.iterator();
			ICondition newCondition = null;
			while(it.hasNext()){
				ICondition existedCon = (ICondition)it.next();
				if (existedCon.getAttrIndex() == numCon.getAttrIndex() && existedCon instanceof NumericCondition){
					flag = false;
					NumericCondition numExistedCon = (NumericCondition)existedCon;
						if(numCon.left<=numExistedCon.right && numCon.right >= numExistedCon.left){
							double newleft = numCon.left>numExistedCon.left?numCon.left:numExistedCon.left;
							double newright = numCon.right<numExistedCon.right?numCon.right:numExistedCon.right;
							newCondition = new NumericCondition(numExistedCon.attrName,numExistedCon.attrIndex,newleft,newright);
							it.remove();
							break;
						}else{
							it.remove();
							break;
						}
				}
			}
			if(flag){
				this.conditions.add(condition);
			}else if(newCondition!=null){
				this.conditions.add(newCondition);
			}
			
		}else{
			NominalCondition nomCon = (NominalCondition)condition;
			for(ICondition c:conditions){
				if (c.getAttrIndex() == nomCon.getAttrIndex() && c instanceof NominalCondition){
					if (((NominalCondition)c).getValue().equals(((NominalCondition)nomCon).getValue())){
						return;
					}else{
						this.conditions.remove(c);
						return;
					}
				}
			}
			if(!this.conditions.contains(nomCon)){
				this.conditions.add(condition);
			}
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
	@Override
	public double ratio() {
		return ratio;
	}
	@Override
	public void setRatio(double r) {
		ratio = r;
	}
	@Override
	public Set<ICondition> getConditions() {
		return conditions;
	}
	@Override
	public IPattern conjuction(IPattern p) {
//		Set<ICondition> cond = new HashSet<>();
//		cond.addAll(conditions);
//		cond.addAll(p.getConditions());
//		return new Pattern(cond);
		Set<ICondition> cond = new HashSet<>();
		Set<ICondition> newC = new HashSet<>();
		for(ICondition c:conditions){
			cond.add(c.copy());
		}
		Iterator<ICondition> it = p.getConditions().iterator();
		while(it.hasNext()){
			ICondition c = it.next();
			boolean flag = true;
			for(ICondition con:cond){
				if (con instanceof NumericCondition){
				double left = ((NumericCondition)con).getLeft();
				double right = ((NumericCondition)con).getRight();
				
					if (c.getAttrIndex() == con.getAttrIndex() && c instanceof NumericCondition){
						if (((NumericCondition)c).getRight() >= left && ((NumericCondition)c).getLeft() <= right){
							double newleft = ((NumericCondition)con).left< ((NumericCondition)c).left? ((NumericCondition)c).left:((NumericCondition)con).left;
							double newright = ((NumericCondition)con).right> ((NumericCondition)c).right? ((NumericCondition)c).right:((NumericCondition)con).right;
							flag = false;
//							it.remove();
//							newC.add(newCondition);
							((NumericCondition) con).left =newleft;
							((NumericCondition) con).right = newright;
							newC.add(con);
						}
					
				}
				
				}else{
					if(!cond.contains(c)){
						flag = false;
					}
				}
			}
			if(flag){
				newC.add(c);
			}
		}
		Pattern pattern = new Pattern();
		pattern.conditions = newC;
		return pattern;
	}
	@Override
	public IPattern disjuction(IPattern p) {
		Set<ICondition> cond = new HashSet<>();
		Set<ICondition> newC = new HashSet<>();
		for(ICondition c:conditions){
			cond.add(c.copy());
		}
		for(ICondition c:p.getConditions()){
			boolean flag = true;
			for(ICondition con:cond){
				if (con instanceof NumericCondition){
				double left = ((NumericCondition)con).getLeft();
				double right = ((NumericCondition)con).getRight();
				
					if (c.getAttrIndex() == con.getAttrIndex() && c instanceof NumericCondition){
						if (((NumericCondition)c).getRight() >= left && ((NumericCondition)c).getLeft() <= right){
							double newleft = ((NumericCondition)con).left> ((NumericCondition)c).left? ((NumericCondition)c).left:((NumericCondition)con).left;
							double newright = ((NumericCondition)con).right< ((NumericCondition)c).right? ((NumericCondition)c).right:((NumericCondition)con).right;
							flag = false;
//							it.remove();
//							newC.add(newCondition);
							((NumericCondition) con).left =newleft;
							((NumericCondition) con).right = newright;
						}
					
				}
				
				}else{
					if(cond.contains(c)){
						flag = false;
					}
				}
			}
			if(flag){
				newC.add(c);
			}
		}
		cond.addAll(newC);
		Pattern pattern = new Pattern();
		pattern.conditions = cond;
		return pattern;
	}
	@Override
	public boolean subset(IPattern p) {
		return this.conditions.containsAll(p.getConditions());
	}
	
}
