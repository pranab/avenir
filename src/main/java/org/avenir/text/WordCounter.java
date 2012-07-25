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


package org.avenir.text;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configured;
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
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.avenir.bayesian.BayesianDistribution;
import org.chombo.util.Utility;

/**
 * Word counter MR
 * @author pranab
 *
 */
public class WordCounter extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "word counter   MR";
        job.setJobName(jobName);
        
        job.setJarByClass(WordCounter.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        job.setMapperClass(WordCounter.CounterMapper.class);
        job.setReducerClass(WordCounter.CounterReducer.class);
        
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        job.setNumReduceTasks(job.getConfiguration().getInt("num.reducer", 1));

        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}
	
	public static class CounterMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
		private String fieldDelimRegex;
		private int textFieldOrdinal;
		private String[] items;
		private static final IntWritable ONE = new IntWritable(1);
		private Analyzer analyzer;
		private Text outKey  = new Text();
		 
        protected void setup(Context context) throws IOException, InterruptedException {
        	fieldDelimRegex = context.getConfiguration().get("field.delim.regex", ",");
        	textFieldOrdinal = Integer.parseInt(context.getConfiguration().get("text.field.ordinal"));
            analyzer = new StandardAnalyzer(Version.LUCENE_35);
            System.out.println("textFieldOrdinal:" + textFieldOrdinal);
        }
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            
        	String text = null;
            if (textFieldOrdinal > 0) {
                String[] items  =  value.toString().split(fieldDelimRegex);
	            text = items[textFieldOrdinal];
            } else {
				text =  value.toString();
            }
           List<String> tokens =  tokenize( text) ;
            //String[] tokens = text.split("\\s+");
            for (String token : tokens) {
            	outKey.set(token);
    			context.write(outKey, ONE);
            }
            
        }        
        
        private List<String> tokenize(String text) throws IOException {
        	//stemming
            TokenStream stream = analyzer.tokenStream("contents", new StringReader(text));
            List<String> tokens = new ArrayList<String>();

            CharTermAttribute termAttribute = (CharTermAttribute)stream.getAttribute(CharTermAttribute.class);
            while (stream.incrementToken()) {
        		String token = termAttribute.toString();
    			tokens.add(token);
        	} 
        	return tokens;
        }
	}

	public static class CounterReducer extends Reducer<Text, IntWritable, NullWritable, Text> {
		private String fieldDelimOut;
		private int count;
		private Text outVal = new Text();

		protected void setup(Context context) throws IOException, InterruptedException {
			fieldDelimOut = context.getConfiguration().get("field.delim.out", ",");
       }

    	protected void reduce(Text key, Iterable<IntWritable> values, Context context)
            	throws IOException, InterruptedException {
    		count = 0;
    		for (IntWritable val : values) {
    			++count;
    		}
    		outVal.set(key.toString() +fieldDelimOut + count );
			context.write(NullWritable.get(), outVal);   		
    	}
	}
	
	
    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new WordCounter(), args);
        System.exit(exitCode);
    }
	
}
