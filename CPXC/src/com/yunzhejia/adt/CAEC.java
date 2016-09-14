package com.yunzhejia.adt;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.yunzhejia.cpxc.Discretizer;
import com.yunzhejia.cpxc.LocalClassifier;
import com.yunzhejia.cpxc.pattern.Pattern;
import com.yunzhejia.cpxc.pattern.PatternSet;
import com.yunzhejia.cpxc.pattern.SupportPatternFilter;
import com.yunzhejia.cpxc.pattern.TERPatternFilter;
import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class CAEC extends AbstractClassifier{
	private static final long serialVersionUID = 3445125166100986742L;
	/** type of base classifier*/
	protected ClassifierType baseType; 
	/** type of local classifiers*/
	protected ClassifierType ensembleType;
	/** ratio to divide dataset to LargeErrSet and SmallErrSet*/
	protected double rho; 
	/** min support for contrast patterns*/
	protected double minSup;
	/** min growth ratio for contrast patterns*/
	protected double minRatio;
	
	protected transient PatternSet patternSet;
	protected transient AbstractClassifier baseClassifier;
	protected transient AbstractClassifier LEClassifier;
	protected transient AbstractClassifier SEClassifier;
	//protected transient HashMap<Pattern, LocalClassifier> ensembles;
	protected transient Discretizer discretizer;
	
	public CAEC() {
	}


	@Override
	public void buildClassifier(Instances data) throws Exception {
		Instances LE = new Instances(data,0);
		Instances SE = new Instances(data,0);
		
		for(Instance ins: data){
			if(ins.value(0)<7){
				LE.add(ins);
			}else{
				SE.add(ins);
			}
		}
		
		LEClassifier = ClassifierGenerator.getClassifier(ClassifierType.LOGISTIC);
		SEClassifier = ClassifierGenerator.getClassifier(ClassifierType.LOGISTIC);
		
		LEClassifier.buildClassifier(LE);
		SEClassifier.buildClassifier(SE);
		
		//this.distributionsForInstances(data);
		
	}
	
	private void testData(Instances data, Instances test) throws Exception{
		AbstractClassifier cl = ClassifierGenerator.getClassifier(baseType);
		cl.buildClassifier(data);
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(cl, test);
		System.out.println("acc="+eval.pctCorrect());
	}

	@Override
	public double[] distributionForInstance(Instance instance)throws Exception{
		if(instance.value(0)<7){
			return LEClassifier.distributionForInstance(instance);
		}else{
			return SEClassifier.distributionForInstance(instance);
		}
	}
	
	@Override
	  public double[][] distributionsForInstances(Instances batch)
	    throws Exception {
	    double[][] batchPreds = new double[batch.numInstances()][];
	    Instances data = patternSet.getMatchingData(batch, discretizer);
	    System.out.println(data.numInstances() +" out of "+batch.numInstances()+"are covered");
	    int count = 0;
	    for (Instance ins : data){
	    	if (ins.classValue() == 0){
	    		count ++;
	    	}
	    }
	    System.out.println(count+" instances of class 0 are covered");
	    System.out.println(data.numInstances()-count+" instances of class 1 are covered");
	    for (int i = 0; i < batch.numInstances(); i++) {
	      batchPreds[i] = distributionForInstance(batch.instance(i));
	    }

	    return batchPreds;
	}
	
	private void buildLocalClassifiers(Instances data) throws Exception {
		for (Pattern p:patternSet){
			LocalClassifier localClassifier = new LocalClassifier(p, this.ensembleType);
			Instances mds = p.getMatches(data, discretizer);
			localClassifier.train(mds);
			localClassifier.setWeight(localClassifier.AER(mds, baseClassifier));
			p.setLocalClassifier(localClassifier);
			/*
			Evaluation eval = new Evaluation(trainingData.get(p));
			eval.evaluateModel(localClassifier.getClassifier(), trainingData.get(p));
			System.out.println("accuracy on pattern: " + eval.pctCorrect() + "%");
			*/
		}
	}


	private PatternSet minePatterns(Instances data, Discretizer discretizer){
		PatternSet ps = null;
		String tmpFile = "tmp/dataForPattern.txt";
		String patternFile = "tmp/output.key";
		File file = new File(tmpFile);
		try {
			PrintWriter writer = new PrintWriter(file);
			for(int i = 0; i < data.numInstances(); i++){
				Instance ins = data.get(i);
				writer.println(discretizer.getDiscretizedInstance(ins));
			}
			writer.close();
			
			String[] cmd = {"program\\GcGrowth.exe", tmpFile,(int)(minSup*data.numInstances())+"","tmp\\output" };
			Process process = new ProcessBuilder(cmd).start();
			//wait until the program terminates
			while(isRunning(process)){}
			ps = new PatternSet();
			ps.readPatterns(patternFile);
			
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		/*verify support for patterns
		for (Pattern p:ps.patterns){
			if(p.getSupport() != p.supportOfData(data, discretizer)){
				System.out.println("!!  "+p);
			}
		}
		*/
		return ps;
	}
	
	private boolean isRunning(Process process) {
	    try {
	        process.exitValue();
	        return false;
	    } catch (Exception e) {
	        return true;
	    }
	}
	
	private void divideData(Instances data, Instances LE, Instances SE) throws Exception{
		List<Double> errs = new ArrayList<Double>();
		double[][] dist = baseClassifier.distributionsForInstances(data);
		for (int i = 0; i < data.numInstances(); i++){
			Instance ins = data.get(i);
			int label = (int)ins.classValue();
			errs.add(1-dist[i][label]);
		}
		//get cutting point
		double k = cuttingPoint(errs);
		//initialize two data sets
		for (int i = 0; i < data.numInstances(); i++){
			Instance ins = data.get(i);
			if (errs.get(i) > k){
				LE.add(ins);
			}else{
				SE.add(ins);
			}
		}
//		System.out.println("cutting error = " + k);
		
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(baseClassifier, data);
		System.out.println("accuracy on whole data: " + eval.pctCorrect() + "%");
		Evaluation eval1 = new Evaluation(data);
		eval1.evaluateModel(baseClassifier, LE);
		System.out.println("accuracy on LE: " + eval1.pctCorrect() + "%   size="+LE.numInstances());
		Evaluation eval2 = new Evaluation(data);
		eval2.evaluateModel(baseClassifier, SE);
		System.out.println("accuracy on SE: " + eval2.pctCorrect() + "%   size="+SE.numInstances());/**/
	}
	/*
	private double cuttingPoint(List<Double> errs){
		List<Double> list = new ArrayList<Double>(errs);
		Collections.sort(list);
		//OutputUtils.print(errs);
		double sum = 0f;
		for (double err:list){
			sum += err;
		}
		double threshold = sum * rho;
		
		double calc = 0f;
		int index = list.size()-1;
		while (calc < threshold){
			calc += list.get(index);
			index--;
		}
		return list.get(index);
	}
	*/
	
	private double cuttingPoint(List<Double> errs){
		double sum = 0f;
		for (double err:errs){
			sum += err;
		}
		double threshold = sum/errs.size() * rho;
		
		return threshold;
	}
	
	public static void main(String[] args){
		CAEC adt = new CAEC();
		DataSource source;
		Instances data;
		try {
			source = new DataSource("data/synthetic1.arff");
//			source = new DataSource("data/vote.arff");
			data = source.getDataSet();
			if (data.classIndex() == -1){
				data.setClassIndex(data.numAttributes() - 1);
			}
			
//			weka.filters.supervised.attribute.Discretize discretizer = new weka.filters.supervised.attribute.Discretize();
//			discretizer.setInputFormat(data);
//			data = weka.filters.supervised.attribute.Discretize.useFilter(data, discretizer);
//			System.out.println(data);
			
			Evaluation eval = new Evaluation(data);
//			adt.buildClassifier(data);
//			eval.evaluateModel(adt, data);
			eval.crossValidateModel(adt, data, 7, new Random(1));
			
			System.out.println("accuracy of "+": " + eval.pctCorrect() + "%");
			System.out.println("AUC of "+": " + eval.weightedAreaUnderROC());
			System.out.println(eval.toSummaryString());
			
			/*
			AbstractClassifier cl = new NaiveBayes();
			cl.buildClassifier(data);
			Evaluation eval1 = new Evaluation(data);
			eval1.evaluateModel(cl, data);
//			eval1.crossValidateModel(cl, data, 7, new Random(1));
			System.out.println("accuracy of NBC: " + eval1.pctCorrect() + "%");
			System.out.println("AUC of NBC: " + eval1.weightedAreaUnderROC());
			*/
			 
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
