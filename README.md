## Introduction
Set of predictive and exploratory machine learning tools. Runs on Hadoop, Spark and Storm

## Philosophy
* Simple to use
* Input output in CSV format
* Metadata defined in simple JSON file
* Extremely configurable with tons of configuration knobs

## Solution
* Exploratry analytic including correlation, feature subset selection
* Naive Bayes
* Discrimininant analysis
* Nearest neighbor
* Decision tree and Random Forest
* Association Mining
* Reinforcement learning
* Stochastic Optimization


## Blogs
The following blogs of mine are good source of details of avenir. These are the only source
of detail documentation
* http://pkghosh.wordpress.com/2014/03/12/using-mutual-information-to-find-critical-factors-in-hospital-readmission/
* http://pkghosh.wordpress.com/2014/01/09/boost-lead-generation-with-online-reinforcement-learning/
* http://pkghosh.wordpress.com/2013/11/06/retarget-campaign-for-abandoned-shopping-carts-with-decision-tree/
* http://pkghosh.wordpress.com/2013/10/06/predicting-customer-loyalty-trajectory/
* http://pkghosh.wordpress.com/2013/08/25/bandits-know-the-best-product-price/
* http://pkghosh.wordpress.com/2013/06/29/learning-but-greedy-gambler/
* http://pkghosh.wordpress.com/2013/04/15/smarter-email-marketing-with-markov-model/
* http://pkghosh.wordpress.com/2013/03/18/analytic-is-your-doctors-friend/
* http://pkghosh.wordpress.com/2013/02/19/stop-the-customer-separation-pain-bayesian-classifier/
* http://pkghosh.wordpress.com/2013/01/31/explore-with-cramer-index/
* https://pkghosh.wordpress.com/2015/07/06/customer-conversion-prediction-with-markov-chain-classifier/
* https://pkghosh.wordpress.com/2015/05/11/is-bigger-data-better-for-machine-learning/
* https://pkghosh.wordpress.com/2015/12/13/association-mining-with-improved-apriori-algorithm/
* https://pkghosh.wordpress.com/2016/03/14/is-neural-network-better-off-with-big-data/
* https://pkghosh.wordpress.com/2016/04/13/customer-churn-prediction-with-svm-using-scikit-learn/
* https://pkghosh.wordpress.com/2016/06/14/inventory-forecasting-with-markov-chain-monte-carlo/
* https://pkghosh.wordpress.com/2016/07/30/customer-segmentation-based-on-online-behavior-using-scikitlearn/
* https://pkghosh.wordpress.com/2016/10/27/supplier-fulfillment-forecasting-with-continuous-time-markov-chain-using-spark/
* https://pkghosh.wordpress.com/2017/04/30/predicting-call-hangup-in-customer-service-calls-with-decision-tree-and-random-forest/
* https://pkghosh.wordpress.com/2017/06/26/project-assignment-optimization-with-simulated-annealing-on-spark/
* https://pkghosh.wordpress.com/2017/09/18/handling-rare-events-and-class-imbalance-in-predictive-modeling-for-machine-failure/
* https://pkghosh.wordpress.com/2017/10/09/combating-high-cardinality-features-in-supervised-machine-learning/
* https://pkghosh.wordpress.com/2018/02/21/optimizing-discount-price-for-perishable-products-with-thompson-sampling-using-spark/
* https://pkghosh.wordpress.com/2018/03/19/handling-categorical-feature-variables-in-machine-learning-using-spark/
* https://pkghosh.wordpress.com/2018/04/18/predicting-crm-lead-conversion-with-gradient-boosting-using-scikitlearn/
* https://pkghosh.wordpress.com/2018/05/14/auto-training-and-parameter-tuning-for-a-scikitlearn-based-model-for-leads-conversion-prediction/
* https://pkghosh.wordpress.com/2018/06/18/leave-one-out-encoding-for-categorical-feature-variables-on-spark/
* https://pkghosh.wordpress.com/2018/07/18/improving-elastic-search-query-result-with-query-expansion-using-topic-modeling/


## Getting started
Project's resource directory has various tutorial documents for the use cases described in
the blogs.

## Configuration 
All configuration parameters are described in the wiki page
https://github.com/pranab/avenir/wiki/Configuration

## Build
Please refer to resource/dependency.txt for build time and run time dependencies

For Hadoop 1
* mvn clean install

For Hadoop 2 (non yarn)
* git checkout nuovo
* mvn clean install

For Hadoop 2 (yarn)
* git checkout nuovo
* mvn clean install -P yarn

## Help
Please feel free to email me at pkghosh99@gmail.com

## Contribution
Contributors are welcome. Please email me at pkghosh99@gmail.com

