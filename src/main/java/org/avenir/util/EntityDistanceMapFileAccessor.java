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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.MapFile;
import org.apache.hadoop.io.Text;
import org.chombo.util.Utility;

/**
 * Writes entity distance text fle as map file. The key is source entity ID. Values is list of target entity and distance
 * pair
 * @author pranab
 *
 */
public class EntityDistanceMapFileAccessor {
	private FileSystem fileSys;
	private Configuration conf;
	private MapFile.Reader reader;
	private String delim;
	
	/**
	 * @throws IOException
	 */
	public EntityDistanceMapFileAccessor() throws IOException {
		conf = new Configuration();;
		fileSys = FileSystem.get(conf);
	}
	
	/**
	 * @param conf
	 * @throws IOException
	 */
	public EntityDistanceMapFileAccessor(Configuration conf) throws IOException {
		this.conf = conf;
		fileSys = FileSystem.get(conf);
	}

	/**
	 * @param filePathParam
	 * @param delim
	 * @throws IOException
	 */
	public void write(String inPutfilePathParam, String outPutfilePathParam, String delim) throws IOException {
    	InputStream fs = Utility.getFileStream(conf, inPutfilePathParam);
    	if (null != fs) {
    		BufferedReader reader = new BufferedReader(new InputStreamReader(fs));
    		String line = null; 
    		
    		Path outputFile = new Path(conf.get(outPutfilePathParam));
    		Text txtKey = new Text();
            Text txtValue = new Text();
            MapFile.Writer  writer = new MapFile.Writer(conf, fileSys, outputFile.toString(),  txtKey.getClass(), txtKey.getClass());
            
    		while((line = reader.readLine()) != null) {
    			int pos = line.indexOf(delim);
    			String key = line.substring(0, pos);
    			String value = line.substring(pos+1);
    			txtKey.set(key);
                txtValue.set(value);
                writer.append(txtKey, txtValue);
    		}
    		IOUtils.closeStream(writer);
    		this.delim = delim;
    	}
	}

	
	/**
	 * @param mapFileDirPathParam
	 * @throws IOException
	 */
	public void initReader(String mapFileDirPathParam) throws IOException {
		String dirPath = Utility.assertStringConfigParam(conf, mapFileDirPathParam, "missing distance map file directory");
		Path mapFiles = new Path(dirPath);
        MapFile.Reader reader = new MapFile.Reader(fileSys, mapFiles.toString(), conf);
	}

	/**
	 * @param key
	 * @param mapFileDirPathParam
	 * @return
	 * @throws IOException
	 */
	public Map<String, Double> read(String key) throws IOException {
		Map<String, Double> distanceMap = new HashMap<String, Double>();
        Text txtKey = new Text(key);
        Text txtValue = new Text();
        reader.get(txtKey, txtValue);
        String distances =  txtValue.toString();
        String[] entities = distances.split(delim);
        for (String entity : entities) {
        	String[] entityDist = entity.split(delim);
        	distanceMap.put(entityDist[0], Double.parseDouble(entityDist[1]));
        }
        return distanceMap;
	}

	/**
	 * 
	 */
	public void closeReader() {
		IOUtils.closeStream(reader);
	}
	
}
