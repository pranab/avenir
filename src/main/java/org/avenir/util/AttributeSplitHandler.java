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


package org.avenir.util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.chombo.util.Utility;

/**
 * Handles splits for an attribute
 * @author pranab
 *
 */
public class AttributeSplitHandler {
	private Map<Integer, List<Split>> attributeSplits = new HashMap<Integer, List<Split>>();
	private List<Split> curSplitList;
	private int cursor;
	
	/**
	 * Adds numerical split
	 * @param attrOrd
	 * @param splitPoints
	 */
	public void addIntSplits(int attrOrd, Integer[] splitPoints) {
		String key = Utility.join(splitPoints,";");
		IntegerSplit intSplit = new IntegerSplit(key, splitPoints);
		List<Split> splitList = getSplitList(attrOrd);
		splitList.add(intSplit);
	}

	private List<Split> getSplitList(int attrOrd) {
		List<Split> splitList = attributeSplits.get(attrOrd);
		if (null == splitList) {
			splitList = new ArrayList<Split>();
			attributeSplits.put(attrOrd, splitList);
		}
		return splitList;
	}
	
	/**
	 * Adds categorical split
	 * @param attrOrd
	 * @param splitSets
	 */
	public void addCategoricalSplits(int attrOrd, List<List<String>> splitSets) {
		CategoricalSplit catSplit = new CategoricalSplit(splitSets);
		List<Split> splitList = getSplitList(attrOrd);
		splitList.add(catSplit);
	}
	
	/**
	 * selects an attribute
	 * @param attrOrd
	 */
	public void selectAttribute(int attrOrd) {
		curSplitList = attributeSplits.get(attrOrd);
		cursor = 0;
	}
	
	/**
	 * returns key for next split
	 * @return
	 */
	public String next() {
		String key = null;
		if (cursor < curSplitList.size()) {
			key = curSplitList.get(cursor).getKey();
		}
		return key;
	}
	
	/**
	 * Returns segment index for numerical split
	 * @param value
	 * @return
	 */
	public int getSegmentIndex(String value) {
		int index = curSplitList.get(cursor).getSegmentIndex(value);
		++cursor;
		return index;
	}
	
	/**
	 * Base class for splits
	 * @author pranab
	 *
	 */
	public  static abstract class Split {
		protected String key;
		protected static final String SPLIT_ELEMENT_SEPRATOR = ":";

		public Split() {
		}

		public Split(String key) {
			this.key = key;
		}
		public String getKey() {
			return key;
		}

		public void setKey(String key) {
			this.key = key;
		}
		
		public abstract int getSegmentIndex(String value);
		
		public abstract void fromString();
		
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static class IntegerSplit extends  Split {
		private Integer[] splitPoints;

		public IntegerSplit(String key, Integer[] splitPoints) {
			super(key);
			this.splitPoints = splitPoints;
		}
		
		public IntegerSplit(String key) {
			super(key);
		}
		
		@Override
		public int getSegmentIndex(String value) {
			int i = 0;
			int iValue = Integer.parseInt(value);
			for ( ; i < splitPoints.length && iValue > splitPoints[i]; ++i) {
			}
			
			return i;
		}
		
		public String toString() {
			return Utility.join(splitPoints, SPLIT_ELEMENT_SEPRATOR);
		}
		
		public void fromString() {
			int[] intArray = Utility.intArrayFromString(key, SPLIT_ELEMENT_SEPRATOR);
			splitPoints = new Integer[intArray.length];
			for (int i = 0; i <  intArray.length; ++i) {
				splitPoints[i] = intArray[i];
			}
		}
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static class CategoricalSplit extends Split {
		private List<List<String>> splitSets;

		public CategoricalSplit(List<List<String>> splitSets) {
			this.splitSets = splitSets;
			key = toString();
		}

		public CategoricalSplit(String key) {
			super(key);
		}
		
		public CategoricalSplit(String key, List<List<String>> splitSets) {
			super(key);
			this.splitSets = splitSets;
		}
		
		@Override
		public int getSegmentIndex(String value) {
			int indx = 0;
			boolean found = false;
			for (List<String> gr : splitSets) {
				if (gr.contains(value)) {
					found = true;
					break;
				}
				++indx;
			}		
			if (!found) {
				throw new IllegalArgumentException("split segment not found for " + value);
			}
			return indx;
		}
		
		public String toString() {
			StringBuilder stBld = new StringBuilder();
			for (List<String> gr : splitSets) {
				stBld.append(gr.toString()).append(SPLIT_ELEMENT_SEPRATOR);
			}
			stBld.deleteCharAt(stBld.length()-1);
			return stBld.toString();
		}
		
		/**
		 * 
		 */
		public void fromString() {
			splitSets = new ArrayList<List<String>>(); 
			String[] splitSetsSt = key.split(SPLIT_ELEMENT_SEPRATOR);
			for (String splitSetSt : splitSetsSt) {
				splitSetSt = splitSetSt.substring(1, splitSetSt.length() -1);
				String[] items = splitSetSt.split(",");
				List<String>  splitSet = new ArrayList<String>();
				for (int i = 0; i < items.length; ++i) {
					splitSet.add(items[i].trim());
				}
				splitSets.add(splitSet);
			}
		}
		
	}

}
