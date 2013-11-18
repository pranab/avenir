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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.util.ToolRunner;
import org.avenir.explore.ClassPartitionGenerator;
import org.chombo.util.Utility;

/**
 * Generates candidate splits. It's a very thin wrapper around ClassPartitionGenerator
 * @author pranab
 *
 */
public class SplitGenerator extends ClassPartitionGenerator {

	/**
	 * Determines input output paths based on base path and split path
	 * @param args
	 * @param job
	 * @return
	 */
	protected String[] getPaths(String[] args, Job job) {
		String[] paths = new String[2];

		Configuration conf =  job.getConfiguration();
		String basePath = conf.get("project.base.path");
		if (Utility.isBlank(basePath)) {
			throw new IllegalStateException("base path not defined");
		}
		String splitPath = conf.get("split.path");
		String inPath = Utility.isBlank(splitPath) ? basePath + "/split=root/data" : 
			basePath + "/split=root/data/" + splitPath;
		String outPath = Utility.getSiblingPath(inPath, "splits");
		paths[0] = inPath;
		paths[1] = outPath;
		return paths;
	}
	

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new SplitGenerator(), args);
        System.exit(exitCode);
	}    

}
