from scipy.stats import rv_continuous
import scipy as sp
import numpy as np

class p_gen(rv_continuous):
	"true probability distribution"
	def _pdf(self,x):
		return 0.5*(1+x)

true_pdf = p_gen(a=-1,b=1,name='p')

def applyf(x):
	return x*x*3.0*(1+x)/2.0

numsamples = 10000

def sampleF():
	print("sampling from f")
	values = [applyf(x) for x in true_pdf.rvs(size=numsamples)]
	print(np.mean(values))
	print(np.var(values))

def compQ1ratio(point):
	return true_pdf.pdf(point)/(sp.stats.norm(3,1).pdf(point))

def sampleQ1():
	print("sampling from q1")
	points = np.random.normal(loc=3.0,size=numsamples)
	values = [applyf(x)*compQ1ratio(x) for x in points]
	ratios = [compQ1ratio(x) for x in points]
	print("importance sampling")
	ismean = np.mean(values)
	print("mean is: " + str(ismean))
	print("variance is " + str(np.var(values))) 
	print("variance is " + str(np.mean([(compQ1ratio(x)*applyf(x)-ismean) ** 2 for x in points])))
	print("variance is " + str(np.mean([applyf(x)*applyf(x)*compQ1ratio(x)*compQ1ratio(x) - 2*ismean*applyf(x)*compQ1ratio(x) + ismean*ismean for x in points])))
	print("weighted importance sampling")
	Z = np.mean(ratios)
	print(np.mean(values)/Z)

def compQ2ratio(point):
	return true_pdf.pdf(point)/(sp.stats.norm(0,1).pdf(point))

def sampleQ2():
	print("sampling from q2")
	points = np.random.normal(loc=0.0,size=numsamples)
	values = [applyf(x)*compQ2ratio(x) for x in points]
	ratios = [compQ2ratio(x) for x in points]
	print("importance sampling")
	print(np.mean(values))
	print(np.var(values))
	print("weighted importance sampling")
	Z = np.mean(ratios)
	print(np.mean(values)/Z)

np.random.seed(seed=403)

sampleF()
sampleQ1()
sampleQ2()