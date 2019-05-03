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
	"0,0,0,1,1,0,0,1,0,90,3,10,7,3,57757,0,1",\
	"1,0,0,0,1,0,1,0,0,19,4,6,6,2,41851,0,1",\
	"0,1,0,0,1,0,0,1,0,11,5,13,5,2,42047,1,0",\
	"0,0,1,0,1,0,0,0,1,72,7,5,4,1,30000,0,1"]

	req = ",,".join(recs)
	print req
	res = requests.post('http://localhost:5002/gbt/predict/batch', json={"recs":req})
	
	if res.ok:
		print res.json()
    
elif op == "get":
	req = "0,0,1,0,0,1,0,0,1,49,7,7,6,1,61432,0,1"    
	res = requests.get('http://localhost:5002/gbt/predict/' + req)
	if res.ok:
		print res.json()
