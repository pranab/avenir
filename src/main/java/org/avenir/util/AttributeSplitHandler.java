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
	
	public void addIntSplits(int attrOrd, Integer[] splitPoints) {
		String key = Utility.join(splitPoints);
		IntegerSplit intSplit = new IntegerSplit(key, splitPoints);
		List<Split> splitList = attributeSplits.get(attrOrd);
		if (null == splitList) {
			splitList = new ArrayList<Split>();
			attributeSplits.put(attrOrd, splitList);
		}
		splitList.add(intSplit);
	}
	
	public void selectAttribute(int attrOrd) {
		curSplitList = attributeSplits.get(attrOrd);
		cursor = 0;
	}
	
	public String next() {
		String key = null;
		if (cursor < curSplitList.size()) {
			key = curSplitList.get(cursor).getKey();
		}
		return key;
	}
	
	public int getSegmentIndex(String value) {
		int index = curSplitList.get(cursor).getSegmentIndex(value);
		++cursor;
		return index;
	}
	
	private static abstract class Split {
		protected String key;

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
		
	}
	
	private static class IntegerSplit extends  Split {
		private Integer[] splitPoints;

		public IntegerSplit(String key, Integer[] splitPoints) {
			super(key);
			this.splitPoints = splitPoints;
		}
		@Override
		public int getSegmentIndex(String value) {
			int i = 0;
			int iValue = Integer.parseInt(value);
			for ( ; i < splitPoints.length && iValue > splitPoints[i]; ++i) {
				
			}
			return i;
		}
	}
	
	private static class CategoricalSplit extends Split {
		private List<List<String>> splitSets;
		
		public CategoricalSplit(String key, List<List<String>> splitSets) {
			super(key);
			this.splitSets = splitSets;
		}
		
		@Override
		public int getSegmentIndex(String value) {
			// TODO Auto-generated method stub
			return 0;
		}
		
	}

}
