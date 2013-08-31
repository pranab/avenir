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
import java.util.ListIterator;

import org.chombo.util.DynamicBean;

/**
 * Manages a group of items with reward data
 * @author pranab
 *
 */
public class GroupedItems {
	private List<DynamicBean> groupItems = new ArrayList<DynamicBean>();
	public static final String ITEM_ID = "itemID";
	public static final String ITEM_COUNT = "count";
	public  static final String ITEM_REWARD = "reward";

	/**
	 * 
	 */
	public void initialize() {
		groupItems.clear();
	}
	
    /**
     * @param itemID
     * @param count
     * @param reward
     */
    public void createtem(String itemID, int count, int reward) {
		DynamicBean item = new DynamicBean();
		item.set(ITEM_ID, itemID);
		item.set(ITEM_COUNT, count);
		item.set(ITEM_REWARD, reward);
		groupItems.add(item);
    }
	
	/**
	 * @return
	 */
	public List<DynamicBean> getGroupItems() {
		return groupItems;
	}

	/**
	 * @param groupItems
	 */
	public void setGroupItems(List<DynamicBean> groupItems) {
		this.groupItems = groupItems;
	}
	
	/**
	 * @param item
	 */
	public void add(DynamicBean item) {
		groupItems.add(item);
	}

	/**
	 * @param item
	 */
	public void remove(DynamicBean item) {
		groupItems.remove(item);
	}

	public int size() {
		return groupItems.size();
	}
	
    /**
     * @param items
     * @param batchSize
     * @return
     */
    public List<DynamicBean> collectItemsNotTried( int batchSize) {
		//collect items not tried before
    	int thisCount = 0;
    	List<DynamicBean> collectedItems = new ArrayList<DynamicBean>();
		ListIterator<DynamicBean> iter = groupItems.listIterator();
		while (iter.hasNext()) {
			DynamicBean groupItem = iter.next();
			thisCount = groupItem.getInt(ITEM_COUNT);
			if (thisCount == 0) {
				if (collectedItems.size() < batchSize) {
					collectedItems.add(groupItem);
					iter.remove();
				} else if (collectedItems.size() == batchSize) {
					break;
				}
			} 
		}
    	
    	return collectedItems;
    }
    
    /**
     * @return
     */
    public DynamicBean selectRandom() {
		int select = (int)Math.round( Math.random() * groupItems.size());
		select = select < groupItems.size() ? select : groupItems.size() -1; 
		DynamicBean item = groupItems.get(select);
		return item;
    }

    /**
     * @return
     */
    public DynamicBean getMaxRewardItem() {
    	int reward = 0;
    	int maxReward = 0;
    	DynamicBean maxRewardItem = null;
		//max reward in this group
		for (DynamicBean groupItem : groupItems) {
				reward = groupItem.getInt(ITEM_REWARD);
			if (reward > maxReward) {
				maxReward = reward;
				maxRewardItem = groupItem;
			}
		}
    	return maxRewardItem;
    }
    
}
