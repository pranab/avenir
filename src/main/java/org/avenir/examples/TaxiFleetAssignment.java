
/*
 * avenir-spark: Predictive analytic based on Spark
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
import java.util.List;

import org.avenir.optimize.BasicSearchDomain;
import org.avenir.optimize.TabuSearchDomain;
import org.chombo.util.BasicUtils;
import org.codehaus.jackson.map.ObjectMapper;

/**
 * @author pranab
 *
 */
public class TaxiFleetAssignment extends TabuSearchDomain {
	private TaxiFleet taxiFleet;
	private int numTaxis;
	private int numPassengers;
	
	@Override
	public void initTrajectoryStrategy(String configFile, int maxStepSize,
			int mutationRetryCountLimit, boolean debugOn) {
		try {
			InputStream fs = new FileInputStream(configFile);
			if (null != fs) {
				ObjectMapper mapper = new ObjectMapper();
				taxiFleet = mapper.readValue(fs, TaxiFleet.class);
			}	
		} catch (IOException ex) {
			throw new IllegalStateException("failed to initialize search object " + ex.getMessage());
		}
		this.debugOn = debugOn;
		numTaxis = taxiFleet.getTaxis().size();
		numPassengers = taxiFleet.getPassengers().size();
		if (numTaxis <= taxiFleet.getPassengers().size()) {
			numComponents = numTaxis;
		} else {
			numComponents = numPassengers;
		}
	}


	@Override
	public BasicSearchDomain createTrajectoryStrategyClone() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected void replaceSolutionComponent(String[] comonents, int index) {
		// TODO Auto-generated method stub
		
	}

	@Override
	protected double calculateCost(String comp) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public boolean isValid(String[] components) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean isValid(String[] componentsex, int index) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	protected void addComponent(String[] componenets, int index) {
		List<Taxi> candidateTaxis = taxiFleet.getTaxis();
		List<Passenger> candidatePassengers = taxiFleet.getPassengers();
		if (numTaxis < numPassengers) {
			candidatePassengers = BasicUtils.selectRandomFromList(candidatePassengers, numTaxis);
		} else if (numTaxis > numPassengers) {
			candidateTaxis = BasicUtils.selectRandomFromList(candidateTaxis, numPassengers);
		}
		
		for (Taxi taxi : candidateTaxis) {
			
		}
	}


	@Override
	protected double getInvalidSolutionCost() {
		// TODO Auto-generated method stub
		return 0;
	}
	
}
