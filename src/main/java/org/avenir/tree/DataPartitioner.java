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

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.avenir.util.AttributeSplitHandler;
import org.chombo.util.FeatureField;
import org.chombo.util.FeatureSchema;
import org.chombo.util.SecondarySort;
import org.chombo.util.Utility;
import org.codehaus.jackson.map.ObjectMapper;

/**
 * Partitions data based on a split selected among some candidate splits
 * generated from the parent node and corresponding data
 * @author pranab
 *
 */
public class DataPartitioner extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(DataPartitioner.class);
    private boolean debugOn;
    
	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Partitions data by some split";
        job.setJobName(jobName);
        
        job.setJarByClass(DataPartitioner.class);

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        debugOn = job.getConfiguration().getBoolean("debug.on", false);
        if (debugOn) {
        	LOG.setLevel(Level.DEBUG);
        }

        job.setMapperClass(DataPartitioner.PartitionerMapper.class);
        job.setReducerClass(DataPartitioner.PartitionerReducer.class);

        //find best split and create output path
        String inPath = getNodePath(job);
        if (debugOn)
        	System.out.println("inPath:" + inPath);
        Split split = findBestSplitKey(job, inPath);
        String outPath = inPath + "/" + "split=" + split.getIndex();
        if (debugOn)
        	System.out.println("outPath:" + outPath);
        		
        FileInputFormat.addInputPath(job, new Path(inPath));
        FileOutputFormat.setOutputPath(job, new Path(outPath));
        
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);
        
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);
        
        job.setPartitionerClass(SecondarySort.RawIntKeyTextPartitioner.class);
        int numReducers = split.getSegmentCount();
        if (debugOn)
        	System.out.println("numReducers:" + numReducers);
        job.setNumReduceTasks(numReducers);

        int status =  job.waitForCompletion(true) ? 0 : 1;
        //move output to segment directories
        if (status == 0) {
        	moveOutputToSegmentDir( outPath,  split.getSegmentCount(), job.getConfiguration());
        }
        return status;
	}
	
	/**
	 * @param outPath
	 * @param segmentCount
	 * @param conf
	 * @throws IOException
	 */
	private void moveOutputToSegmentDir(String outPath,  int segmentCount, Configuration conf) throws IOException {
		 FileSystem fileSystem = FileSystem.get(conf);
		 for (int i = 0; i < segmentCount; ++i) {
			 //create segment dir
			String dir = outPath + "/segment=" + i + "/data";
			Path segmentPath = new Path(dir);
			fileSystem.mkdirs(segmentPath);
			
			//move output to segment dir
			Path srcFile = new Path(outPath + "/part-r-0000" + i);
			Path dstFile = new Path(outPath + "/segment=" + i + "/data/partition.txt");
			fileSystem.rename(srcFile, dstFile);
		}
		
		fileSystem.close();
	}
	
	/**
	 * @param job
	 * @return
	 */
	private String getNodePath(Job job) {
		String nodePath = null;
		Configuration conf =  job.getConfiguration();
		String basePath = conf.get("dap.project.base.path");
		if (Utility.isBlank(basePath)) {
			throw new IllegalStateException("base path not defined");
		}
		String splitPath = conf.get("dap.split.path");
		if (debugOn)
			System.out.println("basePath:" + basePath + " splitPath:" + splitPath);
		nodePath = Utility.isBlank(splitPath) ? basePath + "/split=root/data" : 
			basePath + "/split=root/data/" + splitPath;
		return nodePath;
	}
	
	/**
	 * Finds best split according to chosen strategy
	 * @param job
	 * @param iplutPath
	 * @return
	 * @throws IOException
	 */
	private Split findBestSplitKey(Job job, String inputPath) throws IOException {
		String splitKey = null;
		Configuration conf =  job.getConfiguration();
		String splitSelectionStrategy = conf.get("dap.split.selection.strategy", "best");
		
		String candidateSplitsPath = Utility.getSiblingPath(inputPath, "splits/part-r-00000");
        if (debugOn)
        	System.out.println("candidateSplitsPath:" + candidateSplitsPath);
		conf.set("dap.candidate.splits.path", candidateSplitsPath);
		List<String> lines = Utility.getFileLines(conf, "dap.candidate.splits.path");
		
		//create split objects and sort
		Split[] splits = new Split[lines.size()];
		int i = 0;
		for (String line : lines) {
			splits[i] = new Split(line,i);
			++i;
		}
		
		//sort splits
		Arrays.sort(splits);

		//find split
		int splitIndex = 0;
		if (splitSelectionStrategy.equals("best")) {
		} else if (splitSelectionStrategy.equals("randomFromTop")) {
			int numSplits = conf.getInt("dap.num.top.splits", 5);
			splitIndex = (int)(Math.random() * numSplits);
		}
		Split split = splits[splitIndex];
		
		
		
		//set asplit attribute ordinal and split key
		int splitAttribute = split.getAttributeOrdinal();
		conf.setInt("dap.split.attribute", splitAttribute);
        if (debugOn)
        	System.out.println("splitAttribute:" + splitAttribute);
		splitKey = split.getSplitKey();
        if (debugOn)
        	System.out.println("splitKey:" + splitKey);
		conf.set("dap.split.key", splitKey);
		
        return split;
	}
	
	/**
	 * Sortable split
	 * @author pranab
	 *
	 */
	private static class Split implements  Comparable<Split> {
		private String line;
		private int index;
		private String[] items;
		
		public Split(String line, int index) {
			this.line = line;
			this.index = index;
			items = line.split(";");
		}
		
		@Override
		public int compareTo(Split that) {
			double thisVal = Double.parseDouble(items[2]);
			double thatVal = Double.parseDouble(that.items[2]);
			
			//descending order
			return thisVal > thatVal ? -1 : (thisVal < thatVal ? 1 : 0);
		}

		/**
		 * Split segment
		 * @return
		 */
		public String getSplitKey() {
			return items[1];
		}

		/**
		 * Split segment
		 * @return
		 */
		public String getNormalizedSplitKey() {
			String key = items[1].replaceAll("\\s+", "");
			key = key.replaceAll("\\[", "");
			key = key.replaceAll("\\]", "");
			key = key.replaceAll(":", "-");
			return key;
		}
		
		/**
		 * Split attribute ordinal
		 * @return
		 */
		public int getAttributeOrdinal() {
			return Integer.parseInt(items[0]);
		}
		
		/**
		 * Number of segments in the split
		 * @return
		 */
		public int getSegmentCount() {
			String[] segments = items[1].split(":");
			return segments.length;
		}

		public String getLine() {
			return line;
		}

		public int getIndex() {
			return index;
		}
	}
	
 	
	/**
	 * @author pranab
	 *
	 */
	public static class PartitionerMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
		private String fieldDelimRegex;
		private String[] items;
		private IntWritable outKey = new IntWritable();
        private Text outVal = new Text();
        private FeatureSchema schema;
		private int splitAttrOrd;
		private FeatureField featureField;
		private AttributeSplitHandler.Split split;
		private int splitSegment;
		private String attrVal;
		
        private static final Logger LOG = Logger.getLogger(PartitionerMapper.class);
		
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
            if (conf.getBoolean("debug.on", false)) {
            	LOG.setLevel(Level.DEBUG);
            }
        	fieldDelimRegex = conf.get("field.delim.regex", ",");
        	
        	splitAttrOrd = conf.getInt("dap.split.attribute", -1);
        	if (splitAttrOrd == -1) {
        		throw new IOException("split attribute not found");
        	}
        	LOG.debug("splitAttrOrd:" + splitAttrOrd);
        	String splitKey = conf.get("dap.split.key");
        	LOG.debug("splitKey:" + splitKey);
        	
        	InputStream fs = Utility.getFileStream(context.getConfiguration(), "dap.feature.schema.file.path");
            ObjectMapper mapper = new ObjectMapper();
            schema = mapper.readValue(fs, FeatureSchema.class);
            featureField = schema.findFieldByOrdinal(splitAttrOrd);
            if (featureField.isInteger()) {
            	split = new  AttributeSplitHandler.IntegerSplit(splitKey);
            } else if (featureField.isCategorical()) {
            	split = new AttributeSplitHandler.CategoricalSplit(splitKey);
            }
        	split.fromString();
        	
        }
        
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex);
            
            //key is split segment
        	attrVal = items[splitAttrOrd];
        	splitSegment = split.getSegmentIndex(attrVal);
        	LOG.debug("splitSegment:" + splitSegment);
        	
        	outKey.set(splitSegment);

            context.write(outKey,value);
        }        
	}
	
	
	/**
	 * @author pranab
	 *
	 */
	public static class PartitionerReducer extends Reducer<IntWritable, Text, NullWritable, Text> {
		
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void reduce(IntWritable  key, Iterable<Text> values, Context context)
        		throws IOException, InterruptedException {
        	for (Text value : values) {
        		context.write(NullWritable.get(), value);
        	}
        }
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new DataPartitioner(), args);
        System.exit(exitCode);
	}    
	
}
