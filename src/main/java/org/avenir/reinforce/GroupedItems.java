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
	public static final String ITEM_USE_COUNT = "useCount";
	private static final int ZERO_COUNT = 0;
	private static final int ONE_COUNT = 1;
	
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
		item.set(ITEM_USE_COUNT, ZERO_COUNT);
		
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
    public List<DynamicBean> collectItemsNotTried(int batchSize) {
		//collect items not tried before
    	int thisCount = 0;
    	int thisUseCount = 0;
    	List<DynamicBean> collectedItems = new ArrayList<DynamicBean>();
		ListIterator<DynamicBean> iter = groupItems.listIterator();
		while (iter.hasNext()) {
			DynamicBean groupItem = iter.next();
			thisCount = groupItem.getInt(ITEM_COUNT);
			thisUseCount = groupItem.getInt(ITEM_USE_COUNT);
			
			//only items with 0 usage count from past and present
			if (thisCount == ZERO_COUNT && thisUseCount == ZERO_COUNT) {
				if (collectedItems.size() < batchSize) {
					collectedItems.add(groupItem);
				} else if (collectedItems.size() == batchSize) {
					break;
				}
			} 
		}
    	
    	return collectedItems;
    }
    
    /**
     * select item randomly
     * @return
     */
    public DynamicBean selectRandom() {
		int select = (int)Math.round( Math.random() * groupItems.size());
		select = select < groupItems.size() ? select : groupItems.size() -1; 
		DynamicBean item = groupItems.get(select);
		setUseCount(item);
		return item;
    }

    /**
     * select item with maximum reward
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
    
    /**
     * select item with maximum reward
     * @return
     */
    public DynamicBean selectMaxRewardItem() {
    	DynamicBean maxRewardItem = getMaxRewardItem();
    	if (null != maxRewardItem) {
    		setUseCount(maxRewardItem);
    	}
    	return maxRewardItem;
    }    
    
    /**
     * @param selectedItem
     */
    public DynamicBean select(DynamicBean selectedItem) {
    	incrUseCount(selectedItem);
    	return selectedItem;
    }    

    /**
     * @param selectedItem
     * @param reward
     * @return
     */
    public DynamicBean select(DynamicBean selectedItem, int reward) {
    	updateReward(selectedItem, reward);
    	incrUseCount(selectedItem);
    	return selectedItem;
    }    

    /**
     * @return
     */
    public boolean anyItemTried() {
    	boolean anyTried = false;
		for (DynamicBean groupItem : groupItems) {
			if (groupItem.getInt(ITEM_COUNT) > 0) {
				anyTried = true;
				break;
			}
		}    	
		
		return anyTried;
    }
    
    /**
     * @param groupItem
     * @return
     */
    public int getUsageCount(DynamicBean groupItem) {
    	return groupItem.getInt(ITEM_USE_COUNT);
    }
    
    /**
     * @param groupItem
     */
    public int getTrialCount(DynamicBean groupItem) {
    	return groupItem.getInt(ITEM_COUNT);
    }
    
    /**
     * @param groupItem
     */
    public int getReward(DynamicBean groupItem) {
    	return groupItem.getInt(ITEM_REWARD);
    }

    /**
     * @param groupItem
     */
    public void setReward(DynamicBean groupItem, int reward) {
    	groupItem.set(ITEM_REWARD, reward);
    }

    /**
     * @param groupItem
     */
    public void setUseCount(DynamicBean groupItem) {
    	groupItem.set(ITEM_USE_COUNT, ONE_COUNT);
    }
    
    /**
     * @param groupItem
     */
    public void incrUseCount(DynamicBean groupItem) {
    	groupItem.set(ITEM_USE_COUNT, groupItem.getInt(ITEM_USE_COUNT) + ONE_COUNT);
    }
    
    /**
     * @param groupItem
     */
    public void clearUseCount(DynamicBean groupItem) {
    	groupItem.set(ITEM_USE_COUNT, ZERO_COUNT);
    }

    /**
     * 
     */
    public void clearAllUseCount() {
    	for (DynamicBean groupItem : groupItems) {
    		groupItem.set(ITEM_USE_COUNT, ZERO_COUNT);
		}    	
    }
    
    /**
     * @param groupItem
     * @return
     */
    public int getTotalCount(DynamicBean groupItem) {
    	return groupItem.getInt(ITEM_COUNT) + groupItem.getInt(ITEM_USE_COUNT);
    }
    
    /**
     * @param groupItem
     * @return
     */
    public boolean isAlreadySelected(DynamicBean groupItem) {
    	return groupItem.getInt(ITEM_USE_COUNT) > ZERO_COUNT;
    }
    
    /**
     * @param groupItem
     * @return
     */
    public void updateReward(DynamicBean groupItem, int reward) {
    	int count = getTotalCount(groupItem);
    	int newReward = (getReward(groupItem)  * count  + reward) / (count + 1);
    	setReward(groupItem, newReward);
    }
    
}
