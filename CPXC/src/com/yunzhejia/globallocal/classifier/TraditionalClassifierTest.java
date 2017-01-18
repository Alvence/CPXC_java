package com.yunzhejia.globallocal.classifier;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Random;

import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class TraditionalClassifierTest {

	public static void main(String[] args) {
		try {

			Writer writer = new BufferedWriter(new OutputStreamWriter(
			          new FileOutputStream("tmp/result_t_SVMR"), "UTF-8"));
			DataSource source;
			Instances data;
			String[] files = {"data/synthetic2.arff","data/banana.arff","data/anneal.arff","data/blood.arff","data/diabetes.arff",
					"data/hepatitis.arff","data/ILPD.arff","data/iris.arff","data/labor.arff","data/planning.arff","data/sick.arff"};
//			String[] files = {"data/sick.arff"};
			for(String file:files){
				try{
//			source = new DataSource("data/synthetic2.arff");
			source = new DataSource(file);
//			source = new DataSource("data/iris.arff");
			data = source.getDataSet();
		
			
			AbstractClassifier adt = ClassifierGenerator.getClassifier(ClassifierType.SVM);
			
			if (data.classIndex() == -1){
				data.setClassIndex(data.numAttributes() - 1);
			}
			
			Evaluation eval = new Evaluation(data);
//			adt.buildClassifier(data);
//			adt.testDecisionClassifier(data);
//			eval.evaluateModel(adt, data);
//			System.out.println("accuracy of "+": " + eval.pctCorrect() + "%");
			eval.crossValidateModel(adt, data, 10, new Random(1));
			System.out.println(eval.toSummaryString());
			writer.write(file+"\n");
			writer.write("ACC="+eval.pctCorrect()+"\n");
			writer.write("AUC="+eval.weightedAreaUnderROC()+"\n\n");
				}catch(Exception e){
					//do nothing
//					e.printStackTrace();
				}
			
			}
			writer.close();
			/*AbstractClassifier cl = ClassifierGenerator.getClassifier(GreedyGlobalLocalClassifier_Cluster.globalType);
//			cl.buildClassifier(data);
			Evaluation eval1 = new Evaluation(data);
//			eval1.evaluateModel(cl, data);
			eval1.crossValidateModel(cl, data, 10, new Random(1));
			System.out.println("accuracy of "+": " + bestAcc + "%");
			System.out.println("AUC of "+": " + bestAUC);
			System.out.println("accuracy of global: " + eval1.pctCorrect() + "%");
			System.out.println("AUC of global: " + eval1.weightedAreaUnderROC()+"  bin="+bestNumBin);*/
			/*cl.buildClassifier(data);
			
			    Writer writer = new BufferedWriter(new OutputStreamWriter(
			              new FileOutputStream("tmp/res"), "UTF-8"));
			    Writer writer2 = new BufferedWriter(new OutputStreamWriter(
			              new FileOutputStream("tmp/res2"), "UTF-8"));
			    for (double x = 0; x < 20; x+=0.1){
			    	for(double y = -10; y < 45; y+=0.1){
			    		Instance newIns = data.firstInstance();
			    		newIns.setValue(0, x);
			    		newIns.setValue(1, y);
			    		if (cl.classifyInstance(newIns)==1){
			    			writer.write(x+","+y+"\n");
			    		}
			    		if (cl.classifyInstance(newIns)==0){
			    			writer2.write(x+","+y+"\n");
			    		}
			    	}
			    }
			    writer.close();
			    writer2.close();
		
			
			
			*/
			 
		} catch (Exception e) {
			// TODO Auto-generated catch block
			//e.printStackTrace();
		}

	}

}
