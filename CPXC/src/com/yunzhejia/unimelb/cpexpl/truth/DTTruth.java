package com.yunzhejia.unimelb.cpexpl.truth;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.DataUtils;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.C45Split;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.core.Instance;
import weka.core.Instances;

public class DTTruth {

	public static void main(String[] args) {
		
		try {
			Instances train = DataUtils.load("data/titanic/train.arff");
			Instances test = DataUtils.load("data/titanic/test.arff");
			
			AbstractClassifier cl = ClassifierGenerator.getClassifier(ClassifierGenerator.ClassifierType.DECISION_TREE);
			cl.buildClassifier(train);
			System.out.println(cl);
			
			Instance instance = test.get(1);
			System.out.println("Ins: "+instance);
			
			getExplanation(cl,instance);
			
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		

	}

	private static void getExplanation(AbstractClassifier cl, Instance instance) throws Exception {
		if (!(cl instanceof J48)){
			System.err.println("not J48 tree");
			return;
		}
		J48 dt = (J48)cl;
		ClassifierTree root = dt.m_root;
		int pred = (int)cl.classifyInstance(instance);
		Set<Integer> expl = new HashSet<>();
		while(!root.m_isLeaf){
//			System.out.println(instance.attribute(((C45Split)root.m_localModel).m_attIndex).name());
			int which = root.m_localModel.whichSubset(instance);
			for(int i = 0; i < root.m_sons.length;i++){
				if(i == which)
					continue;
				System.out.println("tree "+i+" dis="+Arrays.toString(root.m_sons[i].distributionForInstance(instance, false)));
				if(root.m_sons[i].classifyInstance(instance)!=pred){
					expl.add(((C45Split)root.m_localModel).m_attIndex);
				}
			}
			
			root = root.m_sons[which];
		}
		System.out.println(expl);
	}
}
