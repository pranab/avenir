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

import java.util.List;

/**
 * @author pranab
 *
 */
public class LearningParameters {
	private List<LearningParameter> parameters;
	private int floatPrecision;
	private List<String> commands; 
	private String execDir;
	private String outputPattern;

	public List<LearningParameter> getParameters() {
		return parameters;
	}

	public void setParameters(List<LearningParameter> parameters) {
		this.parameters = parameters;
	}
	
	public int getFloatPrecision() {
		return floatPrecision;
	}

	public void setFloatPrecision(int floatPrecision) {
		this.floatPrecision = floatPrecision;
	}

	public List<String> getCommands() {
		return commands;
	}

	public void setCommand(List<String> commands) {
		this.commands = commands;
	}

	public String getExecDir() {
		return execDir;
	}

	public void setExecDir(String execDir) {
		this.execDir = execDir;
	}

	public String getOutputPattern() {
		return outputPattern;
	}

	public void setOutputPattern(String outputPattern) {
		this.outputPattern = outputPattern;
	}
	
}
