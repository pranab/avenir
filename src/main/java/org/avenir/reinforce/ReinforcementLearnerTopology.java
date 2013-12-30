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

import java.io.FileInputStream;
import java.util.Properties;



import backtype.storm.Config;
import backtype.storm.StormSubmitter;
import backtype.storm.topology.TopologyBuilder;

/**
 * @author pranab
 *
 */
public class ReinforcementLearnerTopology {

    public static void main(String[] args) throws Exception {
    	String topologyName = args[0];
    	String configFilePath = args[1];
    	if (args.length != 2) {
    		throw new IllegalArgumentException("Nedd two arguments: topology name and config file path");
    	}
    	
        FileInputStream fis = new FileInputStream(configFilePath);
        Properties configProps = new Properties();
        configProps.load(fis);

        //intialize config
        Config conf = new Config();
        conf.setDebug(true);
        for (Object key : configProps.keySet()){
            String keySt = key.toString();
            String val = configProps.getProperty(keySt);
            conf.put(keySt, val);
        }
        
        //spout
        TopologyBuilder builder = new TopologyBuilder();
        int spoutThreads = Integer.parseInt(configProps.getProperty("spout.threads"));
        RedisSpout spout  = new RedisSpout();
        spout.withStreamTupleFields(RedisSpout.EVENT_STREAM, ReinforcementLearnerBolt.EVENT_ID,  
        		ReinforcementLearnerBolt.ROUND_NUM).withStreamTupleFields(RedisSpout.REWARD_STREAM, 
        				ReinforcementLearnerBolt.ACTION_ID,  ReinforcementLearnerBolt.REWARD);
        builder.setSpout("reinforcementLearnerRedisSpout", spout, spoutThreads);
        
        //bolt
        ReinforcementLearnerBolt  bolt = new  ReinforcementLearnerBolt();
        int boltThreads = Integer.parseInt(configProps.getProperty("bolt.threads"));
        builder.setBolt("reinforcementLearnerRedisBolt", bolt, boltThreads).shuffleGrouping("reinforcementLearnerRedisSpout",
        		RedisSpout.EVENT_STREAM).allGrouping("reinforcementLearnerRedisSpout", RedisSpout.REWARD_STREAM);

        //submit topology
        int numWorkers = Integer.parseInt(configProps.getProperty("num.workers"));
        conf.setNumWorkers(numWorkers);
        StormSubmitter.submitTopology(topologyName, conf, builder.createTopology());

    }	
}
