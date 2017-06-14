#!/usr/bin/python

import os
import sys
import time
sys.path.append(os.path.abspath("../lib"))
from util import *
from sampler import *

id_size = int(sys.argv[1])
num_ids = int(sys.argv[2])

for i in range(0, num_ids):
	print genID(id_size)
