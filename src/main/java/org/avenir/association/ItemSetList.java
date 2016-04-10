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

package org.avenir.association;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.chombo.util.Utility;

/**
 * @author pranab
 *
 */
public class ItemSetList {
	private List<ItemSet> itemSetList = new ArrayList<ItemSet>();
	
	/**
	 * @param config
	 * @param statsFilePath
	 * @param itemSetLength
	 * @param containsTransIds
	 * @param delim
	 * @throws IOException
	 */
	public ItemSetList(Configuration config, String statsFilePath, int itemSetLength, boolean containsTransIds, String delim) throws IOException {
    	InputStream fs = Utility.getFileStream(config, statsFilePath);
    	BufferedReader reader = new BufferedReader(new InputStreamReader(fs));
    	String line = null; 
    	
		//item set, transaction Ids, support 
    	while((line = reader.readLine()) != null) {
    		String[] tokens = line.split(delim);
    		itemSetList.add(new ItemSet(tokens, itemSetLength, containsTransIds));
    	}		
	}
	
	public List<ItemSet> getItemSetList() {
		return itemSetList;
	}

	/**
	 * @author pranab
	 *
	 */
	public  static class ItemSet {
		private List<String> items  = new ArrayList<String>();
		private List<String> transactionIds = new ArrayList<String>();
		
		/**
		 * @param tokens
		 * @param itemSetLength
		 */
		public ItemSet(String[] tokens, int itemSetLength, boolean containsTransIds) {
			int i = 0;
			for ( ; i < itemSetLength; ++i ) {
				items.add(tokens[i]);
			}
			
			if (containsTransIds) {
				for ( ; i < tokens.length - 1; ++i) {
					transactionIds.add(tokens[i]);
				}
			}
		}
		
		public List<String> getItems() {
			return items;
		}

		public List<String> getTransactionIds() {
			return transactionIds;
		}

		public boolean containsTrans(String transId) {
			return transactionIds.contains(transId);
		}
		
		public boolean containsItem(String item) {
			return items.contains(item);
		}
	}
}
