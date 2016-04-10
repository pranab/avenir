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

package org.avenir.markov;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.chombo.util.Utility;

/**
 * @author pranab
 *
 */
public class SuffixTreeBuilder {
	private SuffixTreeNode suffixTree = new SuffixTreeNode() ;
	private Map<String, SuffixTreeNode> partitionedSuffixTree = new HashMap<String, SuffixTreeNode>();
	private List<String> tokens = new ArrayList<String>();
	
	/**
	 * @param config
	 * @param suffixTreeFilePathParam
	 * @param delim
	 * @param idOrdinals
	 * @throws IOException
	 */
	public SuffixTreeBuilder(Configuration config, String suffixTreeFilePathParam,  String delim, int[] idOrdinals) throws IOException {
		List<String> lines = Utility.getFileLines(config, suffixTreeFilePathParam);
		for (String line : lines) {
			String[] items = line.split(delim);
			if (null != idOrdinals) {
				String compId = Utility.join(items, 0, idOrdinals.length, delim);
				SuffixTreeNode tree = partitionedSuffixTree.get(compId);
				if (null == tree) {
					tree = new  SuffixTreeNode();
					partitionedSuffixTree.put(compId, tree);
				}
				tokens.clear();
				for (int i = idOrdinals.length; i < items.length; ++i ) {
					tokens.add(items[i]);
				}
				tree.add(tokens);
			} else {
				tokens.clear();
				for (int i = 0; i < items.length; ++i ) {
					tokens.add(items[i]);
				}
				suffixTree.add(tokens);
			}
			
		} 
	}

	/**
	 * @return
	 */
	public SuffixTreeNode getSuffixTree() {
		return suffixTree;
	}

	/**
	 * @param partId
	 * @return
	 */
	public SuffixTreeNode getSuffixTree(String partId) {
		return partitionedSuffixTree.get(partId);
	}
}
