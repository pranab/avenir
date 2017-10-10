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

package org.avenir.storm;

import java.io.FileInputStream;
import java.util.Properties;

import org.chombo.util.ConfigUtility;






import backtype.storm.Config;
import backtype.storm.StormSubmitter;
import backtype.storm.topology.TopologyBuilder;

/**
 * Builds and submits storm topology for reinforcement learning
 * @author pranab
 *
 */
public class ReinforcementLearnerTopology {

    /**
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
    	String topologyName = args[0];
    	String configFilePath = args[1];
    	if (args.length != 2) {
    		throw new IllegalArgumentException("Need two arguments: topology name and config file path");
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
        int spoutThreads = ConfigUtility.getInt(configProps, "spout.threads", 1);
        RedisSpout spout  = new RedisSpout();
        spout.withTupleFields( ReinforcementLearnerBolt.EVENT_ID,ReinforcementLearnerBolt.ROUND_NUM);
        builder.setSpout("reinforcementLearnerRedisSpout", spout, spoutThreads);
        
        //bolt
        ReinforcementLearnerBolt  bolt = new  ReinforcementLearnerBolt();
        int boltThreads = ConfigUtility.getInt(configProps, "bolt.threads", 1);
        builder.
        	setBolt("reinforcementLearnerRedisBolt", bolt, boltThreads).
        	shuffleGrouping("reinforcementLearnerRedisSpout");

        //submit topology
        int numWorkers = ConfigUtility.getInt(configProps, "num.workers", 1);
        int maxSpoutPending = ConfigUtility.getInt(configProps, "max.spout.pending", 1000);
        int maxTaskParalleism = ConfigUtility.getInt(configProps, "max.task.parallelism", 100);
        conf.setNumWorkers(numWorkers);
        conf.setMaxSpoutPending(maxSpoutPending);
        conf.setMaxTaskParallelism(maxTaskParalleism);
        StormSubmitter.submitTopology(topologyName, conf, builder.createTopology());

    }	
}
