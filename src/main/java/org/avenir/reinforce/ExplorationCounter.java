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

package org.avenir.reinforce;

import java.util.ArrayList;
import java.util.List;

/**
 * @author pranab
 *
 */
public class ExplorationCounter {
	private String groupID;
	private int count;
	private int explorationCount;
	private int batchSize;
	private List<int[]> selections = new ArrayList<int[]>();
	
	/**
	 * @param groupID
	 * @param count
	 * @param explorationCount
	 * @param batchSize
	 */
	public ExplorationCounter(String groupID, int count, int explorationCount,
			int batchSize) {
		super();
		this.groupID = groupID;
		this.count = count;
		this.explorationCount = explorationCount;
		this.batchSize = batchSize;
	}
	
	/**
	 * @param roundNum
	 */
	public void selectNextRound(int roundNum) {
        int remainingExplorationRounds = explorationCount - (roundNum - 1) * batchSize;
        selections.clear();
        if (remainingExplorationRounds > 0) {
            int itemIndexBeg = remainingExplorationRounds %  count;
            int itemIndexEnd = itemIndexBeg + batchSize - 1;
            if (itemIndexEnd >= count) {
            	//batch across boundary of all items in group
            	int[] range = new int[2];
            	range[0] = itemIndexBeg;
            	range[1] = count -1;
            	selections.add(range);
            	
            	range = new int[2];
            	range[0] = 0;
            	range[1] = itemIndexEnd - count;
            	selections.add(range);
            } else {
            	//batch within item set
            	int[] range = new int[2];
            	range[0] = itemIndexBeg;
            	range[1] = itemIndexEnd;
            	selections.add(range);
            }
        }        
	}
	
	/**
	 * @return
	 */
	public boolean isInExploration() {
		return !selections.isEmpty();
	}
	
	/**
	 * @param itemIndex
	 * @return
	 */
	public boolean shouldExplore(int itemIndex) {
		boolean toExplore = false;
		for (int[] range : selections) {
			if (itemIndex >= range[0] && itemIndex <= range[1]) {
				toExplore = true;
				break;
			}
		}
		
		return toExplore;
	}

	public String getGroupID() {
		return groupID;
	}

	public int getCount() {
		return count;
	}

	public int getExplorationCount() {
		return explorationCount;
	}

	public int getBatchSize() {
		return batchSize;
	}
	
}
