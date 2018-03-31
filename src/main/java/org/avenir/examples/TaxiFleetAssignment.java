
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
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.avenir.optimize.BasicSearchDomain;
import org.avenir.optimize.Mutation;
import org.avenir.optimize.SolutionWithCost;
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
	private List<Taxi> candidateTaxis;
	private List<Passenger> candidatePassengers;
	private List<String> idleTaxis = new ArrayList<String>();
	private Set<String> taxiIds = new HashSet<String>();
	private List<String> idlePassengers = new ArrayList<String>();
	private Set<String> passengerIds = new HashSet<String>();
	private static final int PASSENGER_SWAP = 0;
	private static final int EXCESS_PASSENGER_SWAP = 1;
	private static final int EXCESS_TAXI_SWAP = 2;
	
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
	public  double getSolutionCost(String solution) {
		return 0;
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
	protected void addSolutionComponent(String[] componenets, int index) {
		String taxiId  = candidateTaxis.get(index).getId();
		String passengerId = candidatePassengers.get(index).getId();
		String component = taxiId + compItemDelim + passengerId;
		componenets[index] = component;
	}

	@Override
	public void prepareCreateSolution() {
		candidateTaxis = taxiFleet.getTaxis();
		candidatePassengers = taxiFleet.getPassengers();
		if (numTaxis < numPassengers) {
			//get shorter passenger list
			candidatePassengers = BasicUtils.selectRandomFromList(candidatePassengers, numTaxis);
		} else if (numTaxis > numPassengers) {
			//get shorter taxi fleet
			candidateTaxis = BasicUtils.selectRandomFromList(candidateTaxis, numPassengers);
		} else {
			//scramble taxi list
			candidateTaxis = new ArrayList<Taxi>();
			candidateTaxis.addAll(taxiFleet.getTaxis());
			BasicUtils.scramble(candidateTaxis, candidateTaxis.size(), 4);
		}
	}
	public void prepareMutateSolution() {
		super.prepareMutateSolution();
		idleTaxis.clear();
		taxiIds.clear();
		idlePassengers.clear();
		passengerIds.clear();
	}

	@Override
	public String mutateSolution(String solution) {
		String[] components = getSolutionComponenets(solution);
		int numTaxis = taxiFleet.getTaxis().size();
		int numPassengers = taxiFleet.getPassengers().size();
		SolutionWithCost solutionDetails = null;
		Mutation postMutation = null;
		Mutation mutation = null;
		String newSoln = null;
		
		if (numTaxis == numPassengers || Math.random() < 0.5) {
			//swap passengers between to 2 taxis
			int firstIndex = BasicUtils.sampleUniform(numComponents);
			int secondIndex = BasicUtils.sampleUniform(numComponents);
			while(secondIndex == firstIndex) {
				secondIndex = BasicUtils.sampleUniform(numComponents);
			}
			String firstComp = components[firstIndex];
			String secondComp = components[secondIndex];
			String[] firstCompItems = getSolutionComponentItems(firstComp);
			String[] secondCompItems = getSolutionComponentItems(secondComp);
			
			String[] mutationComponents = new String[2];
			boolean reversed = false;
			if (firstCompItems[0].compareTo(secondCompItems[0]) < 0) {
				mutationComponents[0] = firstComp;
				mutationComponents[1] = secondComp;
			} else {
				mutationComponents[0] = secondComp;
				mutationComponents[1] = firstComp;
				reversed = true;
			}
			String mutComps = aggregateSolutionComponenets(mutationComponents);
			mutation = new Mutation(PASSENGER_SWAP, mutComps);
			
			//swap passengers
			String temp = secondCompItems[1];
			secondCompItems[1] = firstCompItems[1];
			firstCompItems[1] = temp;
			firstComp = aggregateSolutionComponenetItems(firstCompItems);
			components[firstIndex] = firstComp;
			secondComp = aggregateSolutionComponenetItems(secondCompItems);
			components[secondIndex] = secondComp;
			newSoln = aggregateSolutionComponenets(components);
			double cost = getSolutionCost(newSoln);
			solutionDetails = new SolutionWithCost(newSoln, cost, mutation);
			
			//check tabu list
			if (!reversed) {
				mutationComponents[0] = firstComp;
				mutationComponents[1] = secondComp;
			} else {
				mutationComponents[0] = secondComp;
				mutationComponents[1] = firstComp;
			}
			mutComps = aggregateSolutionComponenets(mutationComponents);
			postMutation = new Mutation(PASSENGER_SWAP, mutComps);
			if (tabuList.contains(postMutation)) {
				
			}
			
		} else {
			if (numTaxis > numPassengers) {
				//excess taxi 
				if (taxiIds.isEmpty()) {
					for (String component : components) {
						String[] compItems = getSolutionComponentItems(component);
						taxiIds.add(compItems[0]);
					}
				
					for (Taxi taxi : taxiFleet.getTaxis()) {
						if (!taxiIds.contains(taxi.getId())) {
							idleTaxis.add(taxi.getId());
						}
					}
				}
				int selIndex = BasicUtils.sampleUniform(numComponents);
				String selComp = components[selIndex];
				String[] selCompItems = getSolutionComponentItems(selComp);
				
				String selTaxiId = BasicUtils.selectRandom(idleTaxis);
				selCompItems[0] = selTaxiId;
				
				String mutComp = aggregateSolutionComponenetItems(selCompItems);
				components[selIndex] = mutComp;
				mutation = new Mutation(EXCESS_TAXI_SWAP, mutComp);
				
				newSoln = aggregateSolutionComponenets(components);
				double cost = getSolutionCost(newSoln);
				solutionDetails = new SolutionWithCost(newSoln, cost, mutation);
			} else if (numTaxis < numPassengers){
				//excess passengers
				if (passengerIds.isEmpty()) {
					for (String component : components) {
						String[] compItems = getSolutionComponentItems(component);
						passengerIds.add(compItems[1]);
					}
				
					for (Passenger passenger : taxiFleet.getPassengers()) {
						if (!passengerIds.contains(passenger.getId())) {
							idlePassengers.add(passenger.getId());
						}
					}
				}
				int selIndex = BasicUtils.sampleUniform(numComponents);
				String selComp = components[selIndex];
				String[] selCompItems = getSolutionComponentItems(selComp);
				
				String selPassengerId = BasicUtils.selectRandom(idlePassengers);
				selCompItems[1] = selPassengerId;
				
				String mutComp = aggregateSolutionComponenetItems(selCompItems);
				components[selIndex] = mutComp;
				mutation = new Mutation(EXCESS_PASSENGER_SWAP, mutComp);
				
				newSoln = aggregateSolutionComponenets(components);
				double cost = getSolutionCost(newSoln);
				solutionDetails = new SolutionWithCost(newSoln, cost, mutation);
			}
		}
		solutions.add(solutionDetails);
		return newSoln;
	}

	@Override
	protected double getInvalidSolutionCost() {
		// TODO Auto-generated method stub
		return 0;
	}
	
}
