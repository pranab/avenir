/*
 * avenir: Predictive analytic based on Hadoop Map Reduce
 * Author: Pranab Ghosh
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you
 * may not use this file except in compliance with the License. You may
 * obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0 
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */


package org.avenir.tree;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.log4j.Logger;
import org.chombo.util.BaseAttribute;
import org.chombo.util.BasicUtils;
import org.chombo.util.FeatureField;
import org.chombo.util.FeatureSchema;
import org.chombo.util.Utility;
import org.hoidla.query.Predicate;

/**
 * @author pranab
 *
 */
public class SplitManager {
	private Map<Integer, String> dataTypes = new HashMap<Integer, String>();
	private List<List<AttributePredicate>> decisionPaths = new ArrayList<List<AttributePredicate>>();
	private FeatureSchema schema;
	private static final String OPERATOR_IN = "in";
	private boolean treeAvailable;
	private Set<Integer> usedAttributes = new HashSet<Integer>();
	private static String predDelim;
	private List<Integer> allAttributes;
	private List<Integer> randomAllAttributes;
	private Map<String, List<Integer>> remainingAttributes = new HashMap<String, List<Integer>>();
	private Map<String, List<Integer>> randomRemainingAttributes = new HashMap<String, List<Integer>>();
	private static final Logger LOG = Logger.getLogger(SplitManager.class);
	private static final String SPACE = " ";
	private int[] customBaseAttributeOrdinals;
	private Map<Integer, List<List<AttributePredicate>>> allSplitPredicates = new HashMap<Integer, List<List<AttributePredicate>>>();
	private boolean debugOn;
	 
	
	/**
	 * @param schema
	 * @throws IOException
	 */
	public SplitManager(FeatureSchema schema,  String predDelim){
		super();
		this.schema = schema;
		SplitManager.predDelim = predDelim;
	}
	
	/**
	 * @param config
	 * @param statFilePathParam
	 * @param delim
	 * @param schema
	 * @throws IOException
	 */
	public SplitManager(Configuration config, String decPathFilePathParam,   String delim, FeatureSchema schema) 
			throws IOException {
		super();
		this.schema = schema;
		predDelim = delim;
		List<String> lines = Utility.getFileLines(config, decPathFilePathParam);
		treeAvailable = !lines.isEmpty();
		for (String line : lines) {
			//each line is decision path
			List<AttributePredicate> decisionPath = new ArrayList<AttributePredicate>(); 
			decisionPaths.add(decisionPath);
			
			String[] splits = line.split(delim);
			for (String split : splits) {
				String[] splitItems = split.split("\\s+");
				int attr = Integer.parseInt(splitItems[0]);
				String operator = splitItems[1];
				String dataType = dataTypes.get(attr);
				if (null == dataType) {
					dataType = schema.findFieldByOrdinal(attr).getDataType();
					dataTypes.put(attr, dataType);
				}
				
				AttributePredicate pred = null;
				if (dataType.equals(BaseAttribute.DATA_TYPE_INT)) {
					pred = new  IntPredicate(attr, operator, Integer.parseInt(splitItems[2]));
				} else if (dataType.equals(BaseAttribute.DATA_TYPE_DOUBLE)) { 
					pred = new  DoublePredicate(attr, operator, Double.parseDouble(splitItems[2]));
				} else if (dataType.equals(BaseAttribute.DATA_TYPE_CATEGORICAL)) { 
					pred = new  CategoricalPredicate(attr, operator, splitItems[2]);
				}
				decisionPath.add(pred);
				usedAttributes.add(attr);
			}
		}
	}

	/**
	 * @param customBaseAttributeOrdinals
	 * @return
	 */
	public SplitManager withCustomBaseAttributeOrdinals(int[] customBaseAttributeOrdinals) {
		this.customBaseAttributeOrdinals = customBaseAttributeOrdinals;
		return this;
	}
	
	/**
	 * 
	 */
	public void initialize() {
		randomAllAttributes = null;
		remainingAttributes.clear();
		randomRemainingAttributes.clear();
	}
	
	public void setDebugOn(boolean debugOn) {
		this.debugOn = debugOn;
	}

	public static String getPredDelim() {
		return predDelim;
	}

	public static void setPredDelim(String predDelim) {
		SplitManager.predDelim = predDelim;
	}

	/**
	 * @return
	 */
	private int[] getBaseAttributeOrdinals() {
		return customBaseAttributeOrdinals != null ? customBaseAttributeOrdinals : 
			schema.getFeatureFieldOrdinals();
	}
	
	/**
	 * @return
	 */
	public List<Integer> getAllAttributes() {
		if (null == allAttributes) {
			allAttributes = Utility.fromIntArrayToList(getBaseAttributeOrdinals());
		}
		return allAttributes;
	}
	
	/**
	 * @param count
	 * @return
	 */
	public List<Integer> getRandomAllAttributes(int count) {
		if (null == randomAllAttributes) {
			List<Integer> allAttrs =  getAllAttributes();
			if (count > allAttrs.size()) {
				count = allAttrs.size();
			}
			randomAllAttributes =  count == allAttrs.size() ?   allAttrs : Utility.selectRandomFromList(allAttrs, count);
		}
		return randomAllAttributes;
	}
	
	/**
	 * @return
	 */
	private List<Integer> getCurrentAttributes() {
		List<Integer> attributes = new ArrayList<Integer>();
		
		if (!decisionPaths.isEmpty()) {
			List<AttributePredicate> decisionPath = decisionPaths.get(0);
			for (AttributePredicate pred :  decisionPath) {
				attributes.add(pred.attribute);
			}
		}
		return attributes;
	}
	
	private List<Integer> getCurrentAttributes(String currentDecPath) {
		List<Integer> attributes = new ArrayList<Integer>();
		if (null != currentDecPath) {
			String[] splits = currentDecPath.split(predDelim);
			for (String split : splits) {
				if (!split.equals(DecisionTreeBuilder.ROOT_PATH)) {
					String[] splitItems = split.split("\\s+");
					int attr = Integer.parseInt(splitItems[0]);
					attributes.add(attr);
				}
			}			
		}
		return attributes;
	}

	/**
	 * @return
	 */
	public List<Integer> getRemainingAttributes(String currentDecPath) {
		List<Integer> candidateAttrs = null;
		if (null != currentDecPath) {
			candidateAttrs = remainingAttributes.get(currentDecPath);
			if (null == candidateAttrs) {
				List<Integer> currentAttrs =  currentDecPath.equals(DecisionTreeBuilder.ROOT_PATH) ? 
					new ArrayList<Integer>() : getCurrentAttributes(currentDecPath);
				candidateAttrs = new ArrayList<Integer>();
				for (int fieldOrd :  getBaseAttributeOrdinals()) {
					if (!currentAttrs.contains(fieldOrd)) {
						candidateAttrs.add(fieldOrd);
					}
				}
				remainingAttributes.put(currentDecPath, candidateAttrs);
			}
		} else {
			candidateAttrs = getAllAttributes();
		}
		return candidateAttrs;
	}

	/**
	 * @return
	 */
	public List<Integer> getRandomRemainingAttributes(String currentDecPath, int count) {
		List<Integer> candidateAttrs  = null;
		String effectCurrentDecPath = null != currentDecPath ? currentDecPath : "$";
		candidateAttrs = randomRemainingAttributes.get(effectCurrentDecPath);
		if (null == candidateAttrs) {
			List<Integer> remainingAttrs = getRemainingAttributes(currentDecPath);
			if (count > remainingAttrs.size()) {
				count = remainingAttrs.size();
			}
			candidateAttrs =  count == remainingAttrs.size()? remainingAttrs :  Utility.selectRandomFromList(remainingAttrs, count);
			randomRemainingAttributes.put(effectCurrentDecPath, candidateAttrs);
		}
		return candidateAttrs;
	}
	
	/**
	 * Returns list of predicates list. The top level corresponds to different split segment levels (2,3 etc)
	 * @param attr
	 * @return
	 */
	public List<List<AttributePredicate>> createIntAttrSplitPredicates(int attr) {
		List<List<AttributePredicate>> splitAttrPredicates = allSplitPredicates.get(attr);
		if (null == splitAttrPredicates) {
			FeatureField field = schema.findFieldByOrdinal(attr);
			List<int[]> splitList = new ArrayList<int[]>();
			createIntPartitions(null, field, splitList);
			
			if (debugOn) {
				System.out.println("int attr:  " + attr);
				for (int[] split  :  splitList) {
					StringBuilder stBld = new StringBuilder();
					stBld.append("split: ");
					for (int part : split) {
						stBld.append("  " + part +  ":");
					}
					System.out.println(stBld.substring(0, stBld.length()-1));
				}
			}
			
			//converts to predicates
			splitAttrPredicates = new ArrayList<List<AttributePredicate>>();
			for (int[] splits :  splitList) {
				splitAttrPredicates.add(createIntAttrPredicates( attr, splits));
			}
			
			allSplitPredicates.put(attr,  splitAttrPredicates);
		}
		return splitAttrPredicates;
	}
	
    /**
     * Create all possible splits within the max number of splits allowed
     * @param splits previous split
     * @param featFld
     * @param newSplitList all possible splits
     */
    public void createIntPartitions(int[] splits, FeatureField field, List<int[]> newSplitList) {
		double min = field.getMin();
		double max = field.getMax();
		double splitScanInterval = field.getSplitScanInterval();

		//adjust number of splits
		int numSplits =(int)( (max - min) / splitScanInterval);
		if (0 == numSplits) {
			splitScanInterval = (max - min) / 2;
		}

    	if (null == splits) {
    		//first time
    		for (int split =(int)( min + splitScanInterval) ; split < max; split += splitScanInterval) {
    			int[] newSplits = new int[1];
    			newSplits[0] = split;
    			newSplitList.add(newSplits);
    			createIntPartitions(newSplits, field, newSplitList);
    		}
    	} else {
    		//create split based off last split that will contain one additional split point
    		int len = splits.length;
    		if (len < field.getMaxSplit() -1) {
    			//only split the last segment of current splits
        		for (int split = (int)(splits[len -1] + splitScanInterval);  split < max; split += splitScanInterval) {
        			int[] newSplits = new int[len + 1];
        			int i = 0;
        			for (; i < len; ++i) {
        				newSplits[i] = splits[i];
        			}
        			newSplits[i] = split;
        			newSplitList.add(newSplits);
        			
        			//recurse to generate additional splits
        			createIntPartitions(newSplits, field,newSplitList);
        		}
    		}
    	}
    }

	/**
	 * Returns list of predicates list. The top level corresponds to different split segment levels (2,3 etc)
	 * @param attr
	 * @return
	 */
	public List<List<AttributePredicate>> createDoubleAttrSplitPredicates(int attr) {
		List<List<AttributePredicate>> splitAttrPredicates = allSplitPredicates.get(attr);
		if (null == splitAttrPredicates) {
			FeatureField field = schema.findFieldByOrdinal(attr);
			List<double[]> splitList = new ArrayList<double[]>();
			createDoublePartitions(null, field, splitList);
			
			//converts to predicates
			splitAttrPredicates = new ArrayList<List<AttributePredicate>>();
			for (double[] splits :  splitList) {
				splitAttrPredicates.add( createDoubleAttrPredicates( attr, splits));
			}
			allSplitPredicates.put(attr,  splitAttrPredicates);
		}
		return splitAttrPredicates;
	}

	
    /**
     * Create all possible splits within the max number of splits allowed
     * @param splits previous split
     * @param featFld
     * @param newSplitList all possible splits
     */
    public void createDoublePartitions(double[] splits, FeatureField field, List<double[]> newSplitList) {
		double min = field.getMin();
		double max = field.getMax();
		double splitScanInterval = field.getSplitScanInterval();
		
		//adjust number of splits
		int numSplits =(int)( (max - min) / splitScanInterval);
		if (0 == numSplits) {
			splitScanInterval = (max - min) / 2;
		}
		
    	if (null == splits) {
    		//first time
    		for (double split = min + splitScanInterval ; split < max; split += splitScanInterval) {
    			double[] newSplits = new double[1];
    			newSplits[0] = split;
    			newSplitList.add(newSplits);
    			createDoublePartitions(newSplits, field, newSplitList);
    		}
    	} else {
    		//create split based off last split that will contain one additional split point
    		int len = splits.length;
    		if (len < field.getMaxSplit() -1) {
    			//only split the last segment of current splits
        		for (double split = splits[len -1] + splitScanInterval;  split < max; split += splitScanInterval) {
        			double[] newSplits = new double[len + 1];
        			int i = 0;
        			for (; i < len; ++i) {
        				newSplits[i] = splits[i];
        			}
        			newSplits[i] = split;
        			newSplitList.add(newSplits);
        			
        			//recurse to generate additional splits
        			createDoublePartitions(newSplits, field,newSplitList);
        		}
    		}
    	}
    }
    
	/**
	 * @param attr
	 * @return
	 */
	public List<List<AttributePredicate>> createCategoricalAttrSplitPredicates(int attr) {
		List<List<AttributePredicate>> splitAttrPredicates = allSplitPredicates.get(attr);
		if (null == splitAttrPredicates) {
			splitAttrPredicates = new ArrayList<List<AttributePredicate>>();
			FeatureField field = schema.findFieldByOrdinal(attr);
			int numGroups = field.getMaxSplit();
			List<List<List<String>>> totalSplitList = new ArrayList<List<List<String>>>();
	
			//all group levels
			for (int gr = 2; gr <= numGroups; ++gr) {
				LOG.debug("num of split sets:" + gr);
				List<List<List<String>>> splitList = new ArrayList<List<List<String>>>();
				createCategoricalPartitions(splitList,  field.getCardinality(), 0, gr);
				totalSplitList.addAll(splitList);
			}
			
			if (debugOn) {
				System.out.println("categorical attr:  " + attr);
				for (List<List<String>> split  :  totalSplitList) {
					StringBuilder stBld = new StringBuilder();
					stBld.append("split: ");
					for (List<String> part : split) {
						stBld.append("  " + BasicUtils.join(part, ":"));
					}
					System.out.println(stBld.toString());
				}
			}
			
			//convert to predicates
			for (List<List<String>> splits : totalSplitList) {
				List<AttributePredicate> predicates = new ArrayList<AttributePredicate>();
				for (List<String> split : splits) {
					predicates.add(new CategoricalPredicate(attr, OPERATOR_IN, split));
				}
				splitAttrPredicates.add(predicates);
			}
			allSplitPredicates.put(attr,  splitAttrPredicates);
		}
		return splitAttrPredicates;
	}
    
    /**
     * @param splitList
     * @param cardinality
     * @param cardinalityIndex
     * @param numGroups
     */
    private void createCategoricalPartitions(List<List<List<String>>>  splitList, List<String> cardinality,
    		int cardinalityIndex, int numGroups) {
    	LOG.debug("next round cardinalityIndex:" + cardinalityIndex);
		//first time
    	if (0 == cardinalityIndex) {
			//initial full splits
			List<List<String>> fullSp = createInitialCategoricalSplit(cardinality, numGroups);
			splitList.add(fullSp);
			
			if (cardinality.size() > numGroups) {
				//partial split shorter in length by one 
				List<List<List<String>>> partialSpList = createPartialCategoricalSplit(cardinality,numGroups-1, numGroups);
				splitList.addAll(partialSpList);
				
				//recurse
				cardinalityIndex += numGroups;
				createCategoricalPartitions(splitList, cardinality,cardinalityIndex, numGroups);
			}
    	} else if (cardinalityIndex < cardinality.size()){
    		//more elements to consume	
    		List<List<List<String>>>  newSplitList = new ArrayList<List<List<String>>>(); 
    		String newElement = cardinality.get(cardinalityIndex);
    		for (List<List<String>> sp : splitList) {
    			if (sp.size() == numGroups) {
    				//if full split, append new element to each group within split to create new splits
    				LOG.debug("creating new split from full split");
    				for (int i = 0; i < numGroups; ++i) {
        				List<List<String>> newSp = new ArrayList<List<String>>();
    					for (int j = 0; j < sp.size(); ++j) {
    						List<String> gr = Utility.cloneList(sp.get(j));
    						
    						//one of the groups in the split is expanded
    						if (j == i) {
    							//add new element to exiting group
    							gr.add(newElement);
    						}
    						newSp.add(gr);
    					}
    					newSplitList.add(newSp);
    				}
    			} else {
    				//if partial split i.e split size less than group size, create new group with new element and add to split
    				LOG.debug("creating new split from partial split");
    				List<List<String>> newSp = new ArrayList<List<String>>();
					for (int i = 0; i < sp.size(); ++i) {
						List<String> gr = Utility.cloneList(sp.get(i));
						newSp.add(gr);
					}
					List<String> newGr = new ArrayList<String>();
					newGr.add(newElement);
					newSp.add(newGr);
					newSplitList.add(newSp);
    			}
    			LOG.debug("newSplitList:" + newSplitList);
    		}
    		
    		//generate partial splits
    		if (cardinalityIndex < cardinality.size() -1){        		
    			List<List<List<String>>> partialSpList = createPartialCategoricalSplit(cardinality,cardinalityIndex, numGroups);
				newSplitList.addAll(partialSpList);
    		}
    		
    		//replace old splits with new
			splitList.clear();
			splitList.addAll(newSplitList);
			
			//recurse
			++cardinalityIndex;
			createCategoricalPartitions(splitList, cardinality,cardinalityIndex, numGroups);
    	}
    }	
    
    /**
     * @param cardinality
     * @param cardinalityIndex
     * @param numGroups
     * @return
     */
    private List<List<List<String>>> createPartialCategoricalSplit(List<String> cardinality,
    		int cardinalityIndex, int numGroups) {
		List<List<List<String>>> partialSplitList = new ArrayList<List<List<String>>>();
    	if (numGroups == 2) {
        	List<List<String>> newSp = new ArrayList<List<String>>();
    		List<String> gr = new ArrayList<String>();
    		for (int i = 0;i <= cardinalityIndex; ++i) {
    			gr.add(cardinality.get(i));
    		}
    		newSp.add(gr);
    		partialSplitList.add(newSp);
    	} else {
    		//create split list with splits shorter in length by 1
    		List<String> partialCardinality = new ArrayList<String>();
    		for (int i = 0; i <= cardinalityIndex; ++i) {
    			partialCardinality.add(cardinality.get(i));
    		}
    		createCategoricalPartitions(partialSplitList,  partialCardinality, 0, numGroups-1);
    	}
    	
		LOG.debug("partial split:" + partialSplitList);
    	return partialSplitList;
    }
   
    /**
     * @param cardinality
     * @param numGroups
     * @return
     */
    private List<List<String>> createInitialCategoricalSplit(List<String> cardinality, int numGroups) {
    	List<List<String>> newSp = new ArrayList<List<String>>();
		for (int i = 0; i < numGroups; ++i) {
			//only one member per group
			List<String> gr = new ArrayList<String>();
			gr.add(cardinality.get(i));
			newSp.add(gr);
		}
		LOG.debug("initial split:" + newSp);
    	return newSp;
    }
   
    /**
	 * @param attr
	 * @param scanInterVal
	 * @return
	 */
	@Deprecated
	public List<AttributePredicate> createIntAttrPredicates(int attr) {
		FeatureField field = schema.findFieldByOrdinal(attr);
		double min = field.getMin();
		double max = field.getMax();
		double splitScanInterval = field.getSplitScanInterval();
		
		//number of splits
		int numSplits =(int)( (max - min) / splitScanInterval);
		if (0 == numSplits) {
			numSplits = 1;
			splitScanInterval = (max - min) / 2;
		}
		int[] splitPoints = new int[numSplits];
		
		//split locations
		int splitPoint = (int)(min + splitScanInterval);
		for (int i = 0; i < numSplits; ++i, splitPoint += splitScanInterval) {
			splitPoints[i] = splitPoint;
		}
		
		return createIntAttrPredicates(attr,  splitPoints);
	}
	
	/**
	 * @param attr
	 * @param scanInterVal
	 * @return
	 */
	@Deprecated
	public List<AttributePredicate> createDoubleAttrPredicates(int attr) {
		FeatureField field = schema.findFieldByOrdinal(attr);
		double min = field.getMin();
		double max = field.getMax();
		double splitScanInterval = field.getSplitScanInterval();
		
		//number of splits
		int numSplits =(int)( (max - min) / splitScanInterval);
		if (0 == numSplits) {
			numSplits = 1;
			splitScanInterval = (max - min) / 2;
		}
		double[] splitPoints = new double[numSplits];
		
		//split locations
		double splitPoint = min + splitScanInterval;
		for (int i = 0; i < numSplits; ++i, splitPoint += splitScanInterval) {
			splitPoints[i] = splitPoint;
		}
		
		return createDoubleAttrPredicates(attr,  splitPoints);
	}

	/**
	 * Creates predicate for each split segment
	 * @param attr
	 * @param splitPoints
	 * @return
	 */
	private List<AttributePredicate> createIntAttrPredicates(int attr, int[] splitPoints) {
		List<AttributePredicate> predicates = new ArrayList<AttributePredicate>();
		AttributePredicate pred = null;

		if (splitPoints.length == 1) {
			pred = new IntPredicate( attr, Predicate.OPERATOR_LE, splitPoints[0]);
			predicates.add(pred);
			pred = new IntPredicate( attr, Predicate.OPERATOR_GT, splitPoints[0]);
			predicates.add(pred);
		} else {
			for (int i = 0;  i  < splitPoints.length; ++i) {
				if (i == splitPoints.length - 1) {
					pred = new IntPredicate( attr, Predicate.OPERATOR_LE, splitPoints[i]);
					predicates.add(pred);
					pred = new IntPredicate( attr, Predicate.OPERATOR_GT, splitPoints[i]);
					predicates.add(pred);
				} else if (i == 0) {
					pred = new IntPredicate( attr, Predicate.OPERATOR_LE, splitPoints[i]);
					predicates.add(pred);
				} else {
					pred = new IntPredicate( attr, Predicate.OPERATOR_LE, splitPoints[i], splitPoints[i-1]);
					predicates.add(pred);
				}
			}
		}
		return predicates;
	}
	
	/**
	 * @param attr
	 * @param splitPoints
	 * @return
	 */
	private List<AttributePredicate> createDoubleAttrPredicates(int attr,  double[] splitPoints) {
		List<AttributePredicate> predicates = new ArrayList<AttributePredicate>();
		AttributePredicate pred = null;
		if (splitPoints.length == 1) {
			pred = new DoublePredicate( attr, Predicate.OPERATOR_LE, splitPoints[0]);
			predicates.add(pred);
			pred = new DoublePredicate( attr, Predicate.OPERATOR_GT, splitPoints[0]);
			predicates.add(pred);
		} else {
			for (int i = 0;  i  < splitPoints.length; ++i) {
				if (i == splitPoints.length - 1) {
					pred = new DoublePredicate( attr, Predicate.OPERATOR_LE, splitPoints[i]);
					predicates.add(pred);
					pred = new DoublePredicate( attr, Predicate.OPERATOR_GT, splitPoints[i]);
					predicates.add(pred);
				} else if (i == 0) {
					pred = new DoublePredicate( attr, Predicate.OPERATOR_LE, splitPoints[i]);
					predicates.add(pred);
				} else {
					pred = new DoublePredicate( attr, Predicate.OPERATOR_LE, splitPoints[i], splitPoints[i-1]);
					predicates.add(pred);
				}
			}
		}
		
		return predicates;
	}
	
	/**
	 * @param attr
	 * @param groups
	 * @return
	 */
	public List<AttributePredicate> createCategoricalAttrPredicates(int attr,  List<List<String>> groups) {
		List<AttributePredicate> predicates = new ArrayList<AttributePredicate>();
		for (List<String> group :  groups) {
			AttributePredicate pred = new CategoricalPredicate( attr, Predicate.OPERATOR_LE,  group);
			predicates.add(pred);
		}
		return predicates;
	}
	
	public boolean isTreeAvailable() {
		return treeAvailable;
	}

	/**
	 * @author pranab
	 *
	 */
	public static abstract class AttributePredicate {
		protected int attribute;
		protected String operator;
		protected String prStr;
		
		public AttributePredicate(int attribute, String operator) {
			super();
			this.attribute = attribute;
			this.operator = operator;
		}
		
		/**
		 * @param operand
		 * @return
		 */
		public abstract boolean evaluate(Object operand) ;
		
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static  class IntPredicate extends AttributePredicate {
		private int value; 
		private Integer otherBound;
		
		/**
		 * @param attribute
		 * @param operator
		 * @param value
		 */
		public IntPredicate(int attribute, String operator, int value) {
			super(attribute, operator);
			this.value = value;
		}
		
		/**
		 * @param attribute
		 * @param operator
		 * @param value
		 * @param otherBound
		 */
		public IntPredicate(int attribute, String operator, int value, Integer otherBound) {
			super(attribute, operator);
			this.value = value;
			this.otherBound = otherBound;
		}
		
		/* (non-Javadoc)
		 * @see org.avenir.tree.SplitManager.AttributePredicate#evaluate(java.lang.Object)
		 */
		public boolean evaluate(Object operandObj) {
			boolean result = false;
			int operand = (Integer)operandObj;
			if (operator.equals(Predicate.OPERATOR_GE)) {
				result = operand >= value;
				if (null != otherBound) {
					result = result && operand < otherBound;
				}
			} else if (operator.equals(Predicate.OPERATOR_GT)) {
				result = operand > value;
				if (null != otherBound) {
					result = result && operand <= otherBound;
				}
			} else if (operator.equals(Predicate.OPERATOR_LE)) {
				result = operand <= value;
				if (null != otherBound) {
					result = result && operand > otherBound;
				}
			} else if (operator.equals(Predicate.OPERATOR_LT)) {
				result = operand < value;
				if (null != otherBound) {
					result = result && operand >= otherBound;
				}
			} else {
				throw new IllegalArgumentException("Illegal int  attribute operator");
			}
					
			return result;
		}
		
		/* (non-Javadoc)
		 * @see java.lang.Object#toString()
		 */
		public String toString() {
			if (null == prStr) {
				StringBuilder stBld = new StringBuilder();
				stBld.append(attribute).append(SPACE).append(operator).append(SPACE).append(value);
				if (null != otherBound) {
					stBld.append(SPACE).append(otherBound);
				}
				prStr = stBld.toString();
			}
			return prStr;
		}
	}

	/**
	 * @author pranab
	 *
	 */
	public static  class DoublePredicate extends AttributePredicate {
		private double value; 
		private Double otherBound;
		
		/**
		 * @param attribute
		 * @param operator
		 * @param value
		 */
		public DoublePredicate(int attribute, String operator, double value) {
			super(attribute, operator);
			this.value = value;
		}

		/**
		 * @param attribute
		 * @param operator
		 * @param value
		 * @param otherBound
		 */
		public DoublePredicate(int attribute, String operator, double value, Double otherBound) {
			super(attribute, operator);
			this.value = value;
		}

		/* (non-Javadoc)
		 * @see org.avenir.tree.SplitManager.AttributePredicate#evaluate(java.lang.Object)
		 */
		public boolean evaluate(Object operandObj) {
			boolean result = false;
			double operand = (Double)operandObj;
			if (operator.equals(Predicate.OPERATOR_GE)) {
				result = operand >= value;
				if (null != otherBound) {
					result = result && operand < otherBound;
				}
			} else if (operator.equals(Predicate.OPERATOR_GT)) {
				result = operand > value;
				if (null != otherBound) {
					result = result && operand <= otherBound;
				}
			} else if (operator.equals(Predicate.OPERATOR_LE)) {
				result = operand <= value;
				if (null != otherBound) {
					result = result && operand > otherBound;
				}
			} else if (operator.equals(Predicate.OPERATOR_LT)) {
				result = operand < value;
				if (null != otherBound) {
					result = result && operand >= otherBound;
				}
			} else {
				throw new IllegalArgumentException("Illegal double attribute operator");
			}
			return result;
		}

		/* (non-Javadoc)
		 * @see java.lang.Object#toString()
		 */
		public String toString() {
			if (null == prStr) {
				StringBuilder stBld = new StringBuilder();
				stBld.append(attribute).append(SPACE).append(operator).append(SPACE).append(value);
				if (null != otherBound) {
					stBld.append(SPACE).append(otherBound);
				}
				prStr = stBld.toString();
			}
			return prStr;
		}
	}

	/**
	 * @author pranab
	 *
	 */
	public static  class CategoricalPredicate extends AttributePredicate {
		private List<String> values; 
		
		/**
		 * @param attribute
		 * @param operator
		 * @param values
		 */
		public CategoricalPredicate(int attribute, String operator, List<String> values) {
			super(attribute, operator);
			this.values = values;
		}

		/**
		 * @param attribute
		 * @param operator
		 * @param values
		 */
		public CategoricalPredicate(int attribute, String operator, String values) {
			super(attribute, operator);
			String[] valueItems = values.split(",");
			this.values = Arrays.asList(valueItems);
		}
		
		/* (non-Javadoc)
		 * @see org.avenir.tree.SplitManager.AttributePredicate#evaluate(java.lang.Object)
		 */
		public boolean evaluate(Object operandObj) {
			boolean result = false;
			String operand = (String)operandObj;
			if (operator.equals(OPERATOR_IN)) {
				result = values.contains(operand);
			} else {
				throw new IllegalArgumentException("Illegal categorical attribute operator");
			}
			return result;
		}
		
		/* (non-Javadoc)
		 * @see java.lang.Object#toString()
		 */
		public String toString() {
			if (null == prStr) {
				StringBuilder stBld = new StringBuilder();
				stBld.append(attribute).append(SPACE).append(operator).append(SPACE);
				for (String value : values) {
					stBld.append(value).append(":");
				}
				prStr = stBld.substring(0, stBld.length() -1);
			}
			return prStr;
		}
		
	}
}
