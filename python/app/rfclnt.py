#!/Users/pranab/Tools/anaconda/bin/python

# avenir-python: Machine Learning
# Author: Pranab Ghosh
# 
# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License. You may
# obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
import sys
import requests


op = sys.argv[1]

if op == "post":
	recs = [\
	"1,0,35,154,144,85,0,1,0,1,0,0,15,15,0,0,0,1",\
	"0,1,62,151,138,63,1,0,0,0,0,1,13,17,0,0,0,1",\
	"0,1,53,126,147,91,0,1,0,0,0,1,10,18,0,0,1,0",\
	"0,1,45,155,142,87,0,1,0,0,0,1,16,16,1,0,0,0"]	

	req = ",,".join(recs)
	print req
	res = requests.post('http://localhost:5002/rf/predict/batch', json={"recs":req})
	
	if res.ok:
		print res.json()
    
elif op == "get":
	req = "1,0,47,204,112,88,1,0,0,0,1,0,16,15,1,0,0,0"    
	res = requests.get('http://localhost:5002/rf/predict/' + req)
	if res.ok:
		print res.json()
