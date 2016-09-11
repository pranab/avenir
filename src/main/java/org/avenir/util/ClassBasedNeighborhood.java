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
import java.util.List;

import org.chombo.util.Tuple;

/**
 * @author pranab
 *
 */
public class ClassBasedNeighborhood {
	private String srcEntityID;
    private String srcClassAttr;
    private String trgClassAttr;
	private List<String> trgEntityIDs = new ArrayList<String>();
	
	public ClassBasedNeighborhood(String[] record) {
		srcEntityID = record[0];
		srcClassAttr = record[1];
		trgClassAttr = record[2];
		for (int i = 3; i < record.length; ++i) {
			trgEntityIDs.add(record[i]);
		}
	}
	
	public boolean isSrcEntity(String srcEntityID) {
		return this.srcEntityID.equals(srcEntityID);
	}
	
	public boolean isTrgEntity(String trgEntityID) {
		return trgEntityIDs.contains(trgEntityID);
	}	
	
	public void generateKey(Tuple key, int subKey) {
		key.initialize();
		key.add(srcEntityID, srcClassAttr, trgClassAttr, subKey);
	}
	
}
