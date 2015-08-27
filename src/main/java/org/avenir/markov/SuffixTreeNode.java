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

import java.util.ArrayList;
import java.util.List;

/**
 * Builds suffix tree
 * @author pranab
 *
 */
public class SuffixTreeNode {
	private int count;
	private String token;
	private SuffixTreeNode parent;
	private List<SuffixTreeNode> children = new ArrayList<SuffixTreeNode>();

	
	/**
	 * 
	 */
	public SuffixTreeNode() {
	}
	
	/**
	 * @param token
	 */
	public SuffixTreeNode(String token) {
		this.token = token;
	}

	/**
	 * Adds a sequence of tokens. 
	 * @param tokens
	 */
	public void add(List<String> tokens) {
		add(tokens, 0);
	}
	
	/**
	 * Adds a sequence of tokens. 
	 * @param tokens
	 * @param offset
	 */
	private void add(List<String> tokens, int offset) {
		boolean done = false;
		for (SuffixTreeNode node : children) {
			if (node.token.equals(tokens.get(offset))) {
				if (offset  ==  tokens.size() - 1) {
					incrementCounters(node);
					done = true;
				} else {
					node.add(tokens,  offset + 1);
				}
			} 
		}
		
		//did not match with any child
		if (!done) {
			//create new child and navigate to new node
			SuffixTreeNode newChild = new SuffixTreeNode(tokens.get(offset));
			newChild.parent = this;
			children.add(newChild);
			if (offset  ==  tokens.size() - 1) {
				incrementCounters(newChild);
				done = true;
			}	else {
				newChild.add(tokens,  offset + 1);
			}
		}
	}
	
	/**
	 * @param node
	 */
	private void incrementCounters(SuffixTreeNode node) {
		SuffixTreeNode nextNode = node;
		while(null != nextNode) {
			++nextNode.count;
			nextNode = nextNode.parent;
		}
	}
	
	/**
	 * @param tokens
	 */
	public SuffixTreeNode  find(List<String> tokens) {
		return find(tokens, 0);
	}
	
	/**
	 * @param tokens
	 * @param offset
	 */
	public SuffixTreeNode find(List<String> tokens, int offset) {
		boolean done = false;
		SuffixTreeNode foundNode = null;
		for (SuffixTreeNode node : children) {
			if (node.token.equals(tokens.get(offset))) {
				if (offset  ==  tokens.size() - 1 )  {
					if (node.isLeaf()) {
						done = true;
						foundNode = node;
					}
					break;
				} else {
					node.find(tokens,  offset + 1);
				}
			} 
		}
		
		return foundNode;
	}
	
	/**
	 * @return
	 */
	public boolean isRoot() {
		return parent == null;
	}
	
	/**
	 * @return
	 */
	public boolean isLeaf() {
		return  children.isEmpty();
	}

	/**
	 * @return
	 */
	public int getCount() {
		return count;
	}

	/**
	 * @return
	 */
	public String getToken() {
		return token;
	}
}
