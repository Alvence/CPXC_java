package com.yunzhejia.cpxc.util;

import java.io.Serializable;

import weka.clusterers.AbstractClusterer;
import weka.clusterers.Canopy;
import weka.clusterers.EM;
import weka.clusterers.SimpleKMeans;

public class ClustererGenerator implements Serializable{
	private static final long serialVersionUID = 5173567202950660952L;

	public enum ClustererType {EM,KMEANS,CANOPY};
	
	public static AbstractClusterer getClusterer(ClustererType type){
		AbstractClusterer clusterer = null;
		switch(type){
			case EM:
				clusterer = new EM();
				break;
			case KMEANS:
				clusterer = new SimpleKMeans();
				break;
			case CANOPY:
				clusterer = new Canopy();
				break;
			default:
				clusterer = new EM();
				break;
				
		}
		return clusterer;
	}
}
