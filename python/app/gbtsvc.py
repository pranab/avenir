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
from flask import Flask, jsonify
from flask import current_app
#from flask.ext.cache import Cache
from flask_cache import Cache
sys.path.append(os.path.abspath("../supv"))
from gbt import *

#REST service gradient boosted tree prediction
app = Flask(__name__)
cache = Cache()
app.config['CACHE_TYPE'] = 'simple'
cache.init_app(app)

configPath = sys.argv[1]

@app.route('/gbt/predict/<string:recs>', methods=['GET'])
def predict(recs):
	print recs
	nrecs = recs.replace(":", "\n")
	model = getModel()
	cls = model.predictProb(nrecs)
	result = cls[:,1]
	result = ["%.3f" %(r) for r in result] 
	result = ",".join(result)
	return jsonify({'predictions': result})

def getModel():
    model = cache.get('gbt_model')
    if model is None:
		model = GradientBoostedTrees(configPath)
		cache.set('gbt_model', model, timeout=600)   
		print "creating and caching gb model"
    return model

if __name__ == '__main__':
    app.run(debug=True, port=5002)