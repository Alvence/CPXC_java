package com.yunzhejia.cpxc;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import com.yunzhejia.cpxc.pattern.Pattern;
import com.yunzhejia.cpxc.pattern.PatternFilter;
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
		
		//step 3 perform binning & contrast pattern mining
		discretizer = new Discretizer();
		discretizer.initialize(data);
		//pattern mining
		patternSet = minePatterns(data,LE,SE,discretizer);
		System.out.println("Patterns number = "+patternSet.size());
		//contrasting
		patternSet.contrast(LE,SE,discretizer,minRatio);
		System.out.println("Patterns number after contrasting= "+patternSet.size());
		
		//step 4 reduce the set of mined contrast pattern
		patternSet = patternSet.filter(new SupportPatternFilter(data.numAttributes()));
		System.out.println("Patterns number after filtering= "+patternSet.size());
		//step 5 build local classifiers
		buildLocalClassifiers(data);
		
		
		//step 6 remove contrast pattern of low utility
		System.out.println("Initial TER="+patternSet.TER(data, baseClassifier, discretizer));
		patternSet = patternSet.filter(new TERPatternFilter(data, baseClassifier, discretizer));
		System.out.println("Final TER="+patternSet.TER(data, baseClassifier, discretizer));
		System.out.println(patternSet.size());
		
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
				int response = (int)ensemble.predict(instance);
				probs[response] += ensemble.getWeight()* pattern.match(instance, discretizer);
				noMatch = false;
			}
		}
		if(noMatch){
			return baseClassifier.distributionForInstance(instance);
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


	private PatternSet minePatterns(Instances data, Instances LE, Instances SE, Discretizer discretizer){
		PatternSet ps = null;
		String tmpFile = "tmp/dataForPattern.txt";
		String patternFile = "tmp/output.closed";
		File file = new File(tmpFile);
		try {
			PrintWriter writer = new PrintWriter(file);
			for(int i = 0; i < LE.numInstances(); i++){
				Instance ins = LE.get(i);
				writer.println(discretizer.getDiscretizedInstance(ins));
			}
			for(int i = 0; i < SE.numInstances(); i++){
				Instance ins = SE.get(i);
				writer.println(discretizer.getDiscretizedInstance(ins));
			}
			writer.close();
			
			String[] cmd = {"program\\GcGrowth.exe", tmpFile,(int)(minSup*data.numInstances())+"","tmp\\output" };
			Process process = new ProcessBuilder(cmd).start();
			//wait until the program is complete
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
		System.out.println("cutting error = " + k);
		
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(baseClassifier, data);
		System.out.println("accuracy on whole data: " + eval.pctCorrect() + "%");
		Evaluation eval1 = new Evaluation(data);
		eval1.evaluateModel(baseClassifier, LE);
		System.out.println("accuracy on LE: " + eval1.pctCorrect() + "%");
		Evaluation eval2 = new Evaluation(data);
		eval2.evaluateModel(baseClassifier, SE);
		System.out.println("accuracy on SE: " + eval2.pctCorrect() + "%");
	}
/*
	private double cuttingPoint(List<Double> errs){
		
		Collections.sort(errs);
		OutputUtils.print(errs);
		double sum = 0f;
		for (double err:errs){
			sum += err;
		}
		double threshold = sum * rho;
		sum = 0f;
		int index = 0;
		while (sum < threshold){
			sum += errs.get(index);
			index++;
		}
		return errs.get(index-1);
	}
	*/
private double cuttingPoint(List<Double> errs){
		List<Double> list = new ArrayList<Double>(errs);
		Collections.sort(list);
		int index = (int)(list.size() * (1-rho));
		return list.get(index);
	}
}
