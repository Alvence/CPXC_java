package com.yunzhejia.unimelb.cpexpl.patternselection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.yunzhejia.pattern.ICondition;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.NominalCondition;
import com.yunzhejia.pattern.NumericCondition;
import com.yunzhejia.pattern.OverlapCalculator;
import com.yunzhejia.pattern.PatternSet;

import lpsolve.LpSolve;
import lpsolve.LpSolveException;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class ProbDiffPatternSelectionLP implements IPatternSelection {

	private Random rand = new Random(0);
	
	
	public ProbDiffPatternSelectionLP(){
	}
	
	class Pair{
		Pair(int i, int j){
			this.i = i;
			this.j = j;
		}
		int i;
		int j;
	}
	
	private String str(double[] arr){
		String ret = "";
		for(double e:arr){
			ret = ret+e+" ";
		}
		return ret;
	}
	
	@Override
	public PatternSet select(Instance x, PatternSet ps, AbstractClassifier cl,int K, Instances samples, Instances headerInfo) throws Exception {
		PatternSet ret = new PatternSet();;
		List<Double> scores = new ArrayList<>();
		List<Double> overlaps = new ArrayList<>();
		for(IPattern p: ps){
			scores.add(eval(cl, p, x, samples, headerInfo));
		}
		
		Map<Integer, Pair> pairs = new HashMap<>();
		int index = 0;
		for(int i = 0; i < ps.size();i++){
			for(int j = i+1; j < ps.size(); j++){
				pairs.put(index++, new Pair(i,j));
				double overlap = OverlapCalculator.overlapMDS(ps.get(i), ps.get(j), samples);
				overlaps.add(0.6*overlap);
			}
		}
		
		
		try {
		      // Create a problem with 4 variables and 0 constraints
			  int numVar = scores.size()+overlaps.size();
		      LpSolve solver = LpSolve.makeLp(0, numVar);
		      // add constraints
		      double[] cons = new double[scores.size()+overlaps.size()];
		      for(int i = 0; i < scores.size();i++){
		    	  cons[i] = 1;
		      }
		      for(int i = 0; i < overlaps.size();i++){
		    	  cons[i+scores.size()] = 0;
		      }
		      
//		      System.out.println(scores.size()+" "+overlaps.size());
		      solver.strAddConstraint(str(cons), LpSolve.LE, K);
		      //binary constraints
		      for(int i = 0; i < numVar; i++){
		    	  double[] bicon = new double[scores.size()+overlaps.size()];
		    	  bicon[i] = 1;
		    	  solver.strAddConstraint(str(bicon), LpSolve.LE, 1);
		      }
		      
		      //overlaps constraints
		      for(int i = 0; i < overlaps.size(); i++){
		    	  double[] bicon = new double[scores.size()+overlaps.size()];
		    	  bicon[pairs.get(i).i] = 1;
		    	  bicon[pairs.get(i).j] = 1;
		    	  bicon[i+scores.size()] = -2;
		    	  solver.strAddConstraint(str(bicon), LpSolve.LE, 1);
		      }
		     
		      
		      // set objective function
		      
		      double[] obj = new double[scores.size()+overlaps.size()];
		      for(int i = 0; i < scores.size();i++){
		    	  obj[i] = -1*scores.get(i);
		      }
		      for(int i = scores.size(); i < obj.length;i++){
		    	  obj[i] = overlaps.get(i-scores.size());
		      }
//		      System.out.println(Arrays.toString(obj));
		      solver.strSetObjFn(str(obj));
		      for(int i = 0; i < numVar;i++){
		    	  solver.setInt(i+1, true);
		      }
//		      solver.setInt(1, true);

		      // solve the problem
		      solver.solve();
		      solver.setOutputfile(null);
//		      solver.printLp();
		      // print solution
		      double[] var = solver.getPtrVariables();
		      System.out.println("solution:"+Arrays.toString(var));
		      for (int i = 0; i < scores.size(); i++) {
		        if( var[i] == 1.0){
		        	ret.add(ps.get(i));
		        }
		      }

		      // delete the problem and free memory
		      solver.deleteLp();
		    }
		    catch (LpSolveException e) {
		       e.printStackTrace();
		    }
		
		return ret;
	}

	private double prediction(AbstractClassifier cl, Instance instance, double classIndex) throws Exception{
		return  cl.distributionForInstance(instance)[(int)classIndex];
	}
	
	private double getRand(double lower, double upper){
		return lower + rand.nextDouble()*(upper-lower);
	}
	
	//Get the prediction without features appearing in the pattern
	public double predictionByRemovingPattern(AbstractClassifier cl, Instance instance, IPattern pattern, Instances data) throws Exception{
				
		Instance ins = (Instance)instance.copy();
		
		List<List<String>> values = new ArrayList<>();
		for(int i = 0; i < instance.numAttributes();i++){
			values.add(new ArrayList<String>());
		}
		
		int numNumericAttr = 5;
		
		for (ICondition condition:pattern.getConditions()){
			if (condition instanceof NominalCondition){
				String val = ((NominalCondition) condition).getValue();
				Enumeration<Object> enums = data.attribute(condition.getAttrIndex()).enumerateValues();
				while(enums.hasMoreElements()){
					String o = (String)enums.nextElement();
					if(!o.equals(val)){
						values.get(condition.getAttrIndex()).add(o);
					}
				}
			}else{
				double left = ((NumericCondition)condition).getLeft();
				double right = ((NumericCondition)condition).getRight();
				if(left!=Double.MIN_VALUE){
					double upper = left;
					double lower = data.attributeStats(condition.getAttrIndex()).numericStats.min;
					for (int i = 0; i < numNumericAttr; i++){
						values.get(condition.getAttrIndex()).add(Double.toString(getRand(lower,upper)));
					}
				}
				if(right!=Double.MAX_VALUE){
					double upper = data.attributeStats(condition.getAttrIndex()).numericStats.max;
					double lower = right;
					for (int i = 0; i < numNumericAttr; i++){
						values.get(condition.getAttrIndex()).add(Double.toString(getRand(lower,upper)));
					}
				}
			}
		}
		for(int i = 0; i < values.size();i++){
			if(values.get(i).size()>0){
				String val = values.get(i).get(rand.nextInt(values.get(i).size()));
				if(ins.attribute(i).isNumeric()){
					ins.setValue(i, Double.parseDouble(val));
				}else{
					ins.setValue(i, val);
				}
			}
		}
		/*
		Instances tmp = new Instances(data,0);
		int[] caps = new int[values.size()];
		int[] curs = new int[values.size()];
		for(int i =0; i < values.size();i++){
			caps[i] = (values.get(i).size());
			curs[i] = 0;
		}
		
		int pos = 0;
		while(true){
			if (pos == values.size()){
				break;
			}
			if(curs[pos] == caps[pos]){
				curs[pos] = 0;
				pos++;
			}
		}*/
		
		
//		System.out.println(ins);
		int classIndex = (int)cl.classifyInstance(instance);
		
		return prediction(cl,ins,classIndex);
	}	
	
	
	

	
	private double eval(AbstractClassifier cl, IPattern pattern, Instance instance, Instances samples, Instances headerInfo) throws Exception {
		double L = 0.0;
		double classIndex = cl.classifyInstance(instance);
		double probOriginal = prediction(cl,instance,classIndex);
		double probDiff = predictionByRemovingPattern(cl, instance, pattern, headerInfo);
//			L += p.support(data)*(probOriginal - probDiff);
		L += (probOriginal - probDiff);
		
		
//		if(tmp.size()>0)
//			L=L/tmp.size();
		
		
//		System.out.println("L=  "+L+"   Omega="+omega+"  tmp="+tmp.size()+" tmp:"+tmp +"  obj="+( L - 0.1*omega));
		return L*pattern.support(samples);
	}

	
//	public static void main(String[] args){
//
//		try {
//		      // Create a problem with 4 variables and 0 constraints
//		      LpSolve solver = LpSolve.makeLp(0, 2);
//
//		      // add constraints
//		      solver.strAddConstraint("3 1 ", LpSolve.GE, 8);
//		      solver.strAddConstraint("0 4 ", LpSolve.GE, 4);
//		      solver.strAddConstraint("2 0 ", LpSolve.LE, 2);
//		      // set objective function
//		      solver.strSetObjFn("5 10");
//
//		      // solve the problem
//		      solver.solve();
//
//		      // print solution
//		      System.out.println("Value of objective function: " + solver.getObjective());
//		      double[] var = solver.getPtrVariables();
//		      for (int i = 0; i < var.length; i++) {
//		        System.out.println("Value of var[" + i + "] = " + var[i]);
//		      }
//		      System.out.println("aaa");
//
//		      // delete the problem and free memory
//		      solver.deleteLp();
//		    }
//		    catch (LpSolveException e) {
//		       e.printStackTrace();
//		    }
////
////        Assert.assertEquals(0.0, solution.getPoint()[0], .0000001);
////        Assert.assertEquals(1.0, solution.getPoint()[1], .0000001);
////        Assert.assertEquals(1.0, solution.getPoint()[2], .0000001);
////        Assert.assertEquals(3.0, solution.getValue(), .0000001);
//		
////		double[] sol = solver.solve(lp);
//	}
	
}
