#
# http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

# Package imports
import os
import sys
sys.path.append(os.path.abspath("../lib"))
from util import *

# parses amazon product review data
# data source http://jmcauley.ucsd.edu/data/amazon/

def parse(path):
	"""
	prase json
	"""
	f = open(path, 'r')
	for l in f:
		yield eval(l)

def prSpecific(revFile, prod, comp):
	"""
	extracts product specific review component
	"""
	for review in parse(revFile):
		asin = review['asin']
		if asin == prod:
			print(review[comp])

if __name__ == "__main__":
	revFile = "data/reviews_Cell_Phones_and_Accessories_5.json"
	op  = sys.argv[1]
	if op == "review":
		# all review text
		for review in parse(revFile):
			print(review['asin'] + "," + review['reviewText'])
	elif op == "revCount":
		# review count for all products			
		revCount = dict()
		for review in parse(revFile):
			asin = review['asin']
			addToKeyedCounter(revCount, asin, 1)

		max = 0
		for item in revCount.items():
			print(item)
			cnt = int(item[1])
			if cnt > max:
				max = cnt
				maxRev = item
		print("max review")		
		print(maxRev)
	elif op == "prReview":
		# all review for a product
		prod = sys.argv[2]
		prSpecific(revFile, prod, "reviewText")
	elif op == "prSummary":
		# all review for a product
		prod = sys.argv[2]
		prSpecific(revFile, prod, "summary")
	elif op == "prOverall":
		# all review for a product
		prod = sys.argv[2]
		prSpecific(revFile, prod, "overall")
	else:
		raise ValueError("invalid command")