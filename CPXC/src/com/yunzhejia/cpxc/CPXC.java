package com.yunzhejia.cpxc;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import com.yunzhejia.cpxc.pattern.AERPatternFilter;
import com.yunzhejia.cpxc.pattern.Pattern;
import com.yunzhejia.cpxc.pattern.PatternFilter;
import com.yunzhejia.cpxc.pattern.PatternSet;
import com.yunzhejia.cpxc.pattern.SupportPatternFilter;
import com.yunzhejia.cpxc.pattern.TERPatternFilter;
import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.OutputUtils;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class CPXC extends AbstractClassifier{
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
	protected transient AbstractClassifier defaultClassifier;
	//protected transient HashMap<Pattern, LocalClassifier> ensembles;
	protected transient Discretizer discretizer;
	
	public CPXC(ClassifierType baseType, ClassifierType ensembleType, double rho, double minSup, double minRatio) {
		super();
		this.baseType = baseType;
		this.ensembleType = ensembleType;
		this.rho = rho;
		this.minSup = minSup;
		this.minRatio = minRatio;
	}


	@Override
	public void buildClassifier(Instances data) throws Exception {
		Instances LE = new Instances(data,0);
		Instances SE = new Instances(data,0);
		
		baseClassifier = ClassifierGenerator.getClassifier(baseType);
		//step 1 learn a base classifier 
		baseClassifier.buildClassifier(data);
		
		//step 2 divide D into LE and SE
		divideData(data,LE,SE);
		
		//step 3 perform binning
		discretizer = new Discretizer();
		discretizer.initialize(data);
		
		//step 4 contrast pattern mining
		patternSet = minePatterns(LE,discretizer);
//		System.out.println("Pattern number = "+patternSet.size());
		//contrasting
		patternSet.contrast(LE,SE,discretizer,minRatio);
//		System.out.println("Pattern number after contrasting = "+patternSet.size());
		
		//step 5 reduce the set of mined contrast pattern
		patternSet = patternSet.filter(new SupportPatternFilter(data.numAttributes()));
//		System.out.println("Pattern number after filtering = "+patternSet.size());
		//step 6 build local classifiers
		buildLocalClassifiers(data);
		
		
		//step 7 remove contrast pattern of low utility
//		System.out.println("Initial TER="+patternSet.TER(data, baseClassifier, discretizer));
//		patternSet = patternSet.filter(new AERPatternFilter());
		
		//step 8 select an optimal set of patterns
		patternSet = patternSet.filter(new TERPatternFilter(data, baseClassifier, discretizer));
//		
//		System.out.println("Final TER="+patternSet.TER(data, baseClassifier, discretizer));
//		System.out.println(patternSet.size());
//		System.out.println(patternSet.getNoMatchingData(LE, discretizer).numInstances()+" out of "+ LE.numInstances() +" are not covered");
//		System.out.println(patternSet.getNoMatchingData(SE, discretizer).numInstances()+" out of "+ SE.numInstances() +" are not covered");
		
		
		//step 9 train the default classifier
		Instances noMatchingData = patternSet.getNoMatchingData(data, discretizer);
		System.out.println(noMatchingData.numInstances()+" out of "+ data.numInstances() +" are not covered");
		
		if(noMatchingData.numInstances() < data.numInstances() * 0.05){
			defaultClassifier = baseClassifier;
		}else{
			defaultClassifier = ClassifierGenerator.getClassifier(ensembleType);
			defaultClassifier.buildClassifier(noMatchingData);
		}
		
	}

	@Override
	public double[] distributionForInstance(Instance instance)throws Exception{
		double[] probs = new double[instance.numClasses()];
		for(int i = 0; i < probs.length; i++){
			probs[i] = 0;
		}
		boolean noMatch = true;
		for (Pattern pattern: patternSet){
			LocalClassifier ensemble = pattern.getLocalClassifier();
			if(pattern.match(instance, discretizer) > 0){
				double[] dist = ensemble.distributionForInstance(instance);
				for(int i = 0; i < dist.length; i++){
					probs[i] += dist[i]*ensemble.getWeight()* pattern.match(instance, discretizer);
				}
				noMatch = false;
			}
		}
		if(noMatch){
			return defaultClassifier.distributionForInstance(instance);
		}
		
		Utils.normalize(probs);
		return probs;
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
		/*
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(baseClassifier, data);
		System.out.println("accuracy on whole data: " + eval.pctCorrect() + "%");
		Evaluation eval1 = new Evaluation(data);
		eval1.evaluateModel(baseClassifier, LE);
		System.out.println("accuracy on LE: " + eval1.pctCorrect() + "%   size="+LE.numInstances());
		Evaluation eval2 = new Evaluation(data);
		eval2.evaluateModel(baseClassifier, SE);
		System.out.println("accuracy on SE: " + eval2.pctCorrect() + "%   size="+SE.numInstances());*/
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
}
