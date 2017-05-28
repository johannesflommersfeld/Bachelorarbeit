import timeit
import time
import datetime
import os
import numpy as np

t1 = timeit.Timer("s.search()", "from SG_Implementation_no_hessian_tm import Searcher; s = Searcher()")
t2 = timeit.Timer("s.search()", "from SG_Implementation_tm import SearcherNH; s = SearcherNH()")

t1_all = t1.repeat(10,1000)
print("Without Hessian: {0}".format(np.mean(t1_all)))
t2_all = t2.repeat(10,1000)
print("With Hessian: {0}".format(np.mean(t2_all)))

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
f = open(os.path.join("results","results_tm_{0}".format(timestamp))+".txt","w")
f.write("Time measurement of flow equation method with and without hessian.\n Given are all values of 10 runs, each with 1000 executions of the code and the mean of it.\n")
f.write("Without Hessian: {0}\n".format(t1_all))
f.write("Without Hessian mean: {0}\n".format(np.mean(t1_all)))
f.write("With Hessian: {0}\n".format(t2_all))
f.write("With Hessian: {0}\n".format(np.mean(t2_all)))
f.close()
