/**
 * 
 */
package com.yunzhejia.pattern;

import weka.core.Instance;

/**
 * @author Yunzhe(Alvin) Jia, The University of Melbourne
 * @version 1.0
 */
public class NominalCondition implements ICondition {

	private int attrIndex;
	private String value;
	private String attrName;
	/**
	 * 
	 */
	
	
	public NominalCondition(int attrIndex, String value) {
		this(null, attrIndex, value);
	}
	
	public String getValue() {
		return value;
	}

	public void setValue(String value) {
		this.value = value;
	}

	public String getAttrName() {
		return attrName;
	}

	public void setAttrName(String attrName) {
		this.attrName = attrName;
	}

	public void setAttrIndex(int attrIndex) {
		this.attrIndex = attrIndex;
	}

	public NominalCondition(String attrName, int attrIndex, String value) {
		this.attrName = attrName;
		this.attrIndex = attrIndex;
		this.value = value;
	}

	public NominalCondition(NominalCondition nominalCondition) {
		this(nominalCondition.attrName,nominalCondition.attrIndex,nominalCondition.value);
	}

	/* (non-Javadoc)
	 * @see com.yunzhejia.pattern.ICondition#match(weka.core.Instance)
	 */
	@Override
	public boolean match(Instance ins) {
		if (ins.stringValue(attrIndex).equals(value)){
			return true;
		}
		return false;
	}

	@Override
	public String toString(){
		return attrName + "=" + value;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + attrIndex;
		result = prime * result + ((attrName == null) ? 0 : attrName.hashCode());
		result = prime * result + ((value == null) ? 0 : value.hashCode());
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
		NominalCondition other = (NominalCondition) obj;
		if (attrIndex != other.attrIndex)
			return false;
		if (attrName == null) {
			if (other.attrName != null)
				return false;
		} else if (!attrName.equals(other.attrName))
			return false;
		if (value == null) {
			if (other.value != null)
				return false;
		} else if (!value.equals(other.value))
			return false;
		return true;
	}

	@Override
	public int getAttrIndex() {
		return attrIndex;
	}

	@Override
	public ICondition copy() {
		return new NominalCondition(this);
	}
	
	
}
