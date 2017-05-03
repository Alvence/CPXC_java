package com.yunzhejia.pattern;

import java.util.HashSet;
import java.util.Set;

import weka.core.Instance;
import weka.core.Instances;

public class OverlapCalculator {
	public static double overlapMDS(IPattern p1, IPattern p2, Instances samples){
		int countCup = 0;
		int countCap = 0;
		for(Instance ins:samples){
			if(p1.match(ins)||p2.match(ins)){
				countCup++;
			}
			if(p1.match(ins)&&p2.match(ins)){
				countCap++;
			}
		}
		if(countCup==0){
			return 0;
		}else{
			return countCap*1.0/countCup;
		}
	}
	
	public static double overlap(IPattern p1, IPattern p2, Instances headerInfo){
//		System.out.println(p1);
//		System.out.println(p2);
		IPattern union = p1.disjuction(p2);
		IPattern intersec = p1.conjuction(p2);
//		System.out.println("union: " +union);
//		System.out.println("inter: "+intersec);
		double delimeter = union.getConditions().size();
		double nominiter = 0;
		
		for(ICondition cond : intersec.getConditions()){
			if (cond instanceof NominalCondition){
				if(union.getConditions().contains(cond)){
					nominiter += 1;
				}
			}else{
				NumericCondition numCond = (NumericCondition) cond;
				double r = (numCond.right==Double.MAX_VALUE?headerInfo.attributeStats(numCond.attrIndex).numericStats.max:numCond.right)
						- (numCond.left == Double.MIN_VALUE?headerInfo.attributeStats(numCond.attrIndex).numericStats.min:numCond.left);
				for (ICondition c:union.getConditions()){
					if (c.getAttrIndex() == numCond.getAttrIndex()){
						NumericCondition numC = (NumericCondition)c;
						if(numC.left < numCond.right && numC.right > numCond.left){
							double newR =  (numC.right==Double.MAX_VALUE?headerInfo.attributeStats(numC.attrIndex).numericStats.max:numC.right)
									- (numC.left == Double.MIN_VALUE?headerInfo.attributeStats(numC.attrIndex).numericStats.min:numC.left);
							nominiter+= (r/newR);
							break;
						}
					}
				}
			}
		}
		
//		System.out.println(nominiter/delimeter);
		return nominiter/delimeter;
	}
	
	public static void main(String[] args){
		ICondition c11 = new NominalCondition("color", 1, "PURPLE");
		ICondition c12 = new NominalCondition("region", 0, "2");
		ICondition c13 = new NumericCondition("C", 2, 4, 7);
		ICondition c21 = new NominalCondition("region", 0, "1");
		ICondition c22 = new NominalCondition("color", 1, "PURPLE");
		ICondition c23 = new NominalCondition("size", 3, "SMALL");
		ICondition c24 = new NumericCondition("C", 2, 8, 9);
		Set<ICondition> conds = new HashSet<>();
		conds.add(c11);
		conds.add(c12);
		conds.add(c13);
		
		Set<ICondition> conds2 = new HashSet<>();
		conds2.add(c21);
		conds2.add(c22);
		conds2.add(c23);
		conds2.add(c24);
		
		Pattern p1 = new Pattern(conds);
		Pattern p2 = new Pattern(conds2);
		System.out.println(overlap(p1, p2,null));
	}
}
