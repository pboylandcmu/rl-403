def sampNorm(mean,std):
	#TODO

class CEM():
	def __init__(self, numseqs, numparts, numelites, predhor,penn):
		self.numseqs = numseqs
		self.numparts = numparts
		self.horizon = predhor
		self.means = [zeros(actionsize) for _ in range(predhor)]
		self.stds = [ones(actionsize) for _ in range(predhor)]
		self.penn = penn
		self.numelites = numelites

	def itermeans(self, initstate):
		actionplan = [sampleactions(self.means,self.stds) for _ in range(self.numseqs)]
		costs = []
		for actionseq in actionplan:
			state = initstate
			cost = 0
			for action in actionseq:
				state = self.penn.evalmodel(randint(self.penn.modelsize),state,action)
				cost += calccost(state)
			costs.append(cost)
		eliteidxs = gettopn(costs,self.numelites)
		elites = zip(*[actionplan[i] for i in eliteidxs])
		self.means = [mean(list(tup)) for tup in elites]
		self.stds = [std(list(tup)) for tup in elites]
		return sampNorm(self.means[0],self,stds[0]) #or 0 std?

	def advance(self):
		self.means = self.means[1:].append(zeros(actionsize))
		self.stds = self.stds[1:].append(ones(actionsize))

	def reset(self):
		self.means = [zeros(actionsize) for _ in range(predhor)]
		self.stds = [ones(actionsize) for _ in range(predhor)]