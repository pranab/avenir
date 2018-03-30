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

package org.avenir.examples;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.avenir.optimize.BasicSearchDomain;
import org.chombo.stats.UniformFloatSampler;
import org.chombo.stats.UniformIntSampler;
import org.chombo.stats.UniformSampler;
import org.chombo.stats.UniformStringSampler;
import org.chombo.util.BaseAttribute;
import org.chombo.util.BasicUtils;
import org.chombo.util.TypedObject;
import org.codehaus.jackson.map.ObjectMapper;

/**
 * @author pranab
 *
 */
public class LearningParameterSearch extends BasicSearchDomain {
	private LearningParameters parameterSpace;
	private List<UniformSampler> paramSamplers;
	private Pattern outputPattern;
	
	@Override
	public void initTrajectoryStrategy(String configFile, int maxStepSize,
			int mutationRetryCountLimit, boolean debugOn) {
		try {
			InputStream fs = new FileInputStream(configFile);
			if (null != fs) {
				ObjectMapper mapper = new ObjectMapper();
				parameterSpace = mapper.readValue(fs, LearningParameters.class);
			}	
		} catch (IOException ex) {
			throw new IllegalStateException("failed to initialize search object " + ex.getMessage());
		}
		numComponents = parameterSpace.getParameters().size();
		withMaxStepSize(maxStepSize);
		withConstantStepSize();
		this.mutationRetryCountLimit = mutationRetryCountLimit;
		compItemDelim = "=";
		
		//create samplers
		for(LearningParameter learnParam : parameterSpace.getParameters()) {
			String type = learnParam.getType();
			
			if (type.equals(BaseAttribute.DATA_TYPE_STRING)) {
				paramSamplers.add(new UniformStringSampler(learnParam.getValues()));
			} else if (type.equals(BaseAttribute.DATA_TYPE_INT)) {
				if (learnParam.getValues().size() == 2) {
					int min =  Integer.parseInt(learnParam.getValues().get(0));
					int max =  Integer.parseInt(learnParam.getValues().get(1));
					paramSamplers.add(new UniformIntSampler(min, max));
				} else {
					throw new IllegalStateException("int parameter does not have right number of values");
				}
			} else if (type.equals(BaseAttribute.DATA_TYPE_FLOAT)) {
				if (learnParam.getValues().size() == 2) {
					float min =  Float.parseFloat(learnParam.getValues().get(0));
					float max =  Float.parseFloat(learnParam.getValues().get(1));
					paramSamplers.add(new UniformFloatSampler(min, max));
				} else {
					throw new IllegalStateException("float parameter does not have right number of values");
				}
			} else {
				throw new IllegalStateException("invalid parameter type");
			}
		}
		
		//output pattern
		outputPattern = Pattern.compile(parameterSpace.getOutputPattern());
	}

	@Override
	public BasicSearchDomain createTrajectoryStrategyClone() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected void replaceSolutionComponent(String[] components, int index) {
		String curComp = components[index];
		addSolutionComponent(components, index);
		while(curComp.equals(components[index])) {
			addSolutionComponent(components, index);
		}
	}

	@Override
	public  double getSolutionCost(String solution) {
		double cost = 0;
		
		List<String> commands = new ArrayList<String>();
		commands.addAll(parameterSpace.getCommands());
		for (String comp : getSolutionComponenets(solution)) {
			commands.add(comp);
		}
		
		//execute ML application
		String result = BasicUtils.execShellCommand(commands, parameterSpace.getExecDir());
		
		//extract performance metric
    	Matcher matcher = outputPattern.matcher(result);
    	if (matcher.matches()) {
    		String metric = matcher.group(1);
    		cost = Double.parseDouble(metric);
    	} else {
    		throw new IllegalStateException("failed to extract metric from output");
    	}
    	
		return cost;
	}
	
	@Override
	protected double calculateCost(String comp) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public boolean isValid(String[] components) {
		return true;
	}

	@Override
	public boolean isValid(String[] componentsex, int index) {
		return true;
	}

	@Override
	protected void addSolutionComponent(String[] components, int index) {
		UniformSampler sampler = paramSamplers.get(index);
		TypedObject sampled = sampler.sample();
		String comp = parameterSpace.getParameters().get(index).getName() + compItemDelim + sampled.getValue();
		components[index] = comp;
	}

	@Override
	protected double getInvalidSolutionCost() {
		// TODO Auto-generated method stub
		return 0;
	}

}
