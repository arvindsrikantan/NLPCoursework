import numpy as np
from scipy.optimize import minimize as mymin 

class MyMaxEnt(object):
	'''
		Python Class for the Maximum Entropy Classifier
	'''
	def __init__(self,hist_list,feature_fn_list,tags):
		'''
			Initialises the Max Ent model by producing the feature vectors for the training data
		'''
		self.hist_list = hist_list
		self.tags = tags
		self.fvectors = self.create_dataset(feature_fn_list)
		
		s = np.array([0]*10)
		for(i in self.fvectors.values()):
			np.add(s,i)
		self.cum_f = s
		

	def init_model(self):
		'''
			Initialises the model parameter
		'''
		self.model = np.array([0]*10)

	def cost(self,model):
		'''
			Given the model, compute the cost 
		'''
		# return L(v)
		
		L_of_v = sum([math.log(self.p_y_given_x(i,tag)) for i in self.fvectors.keys() for tag in tags])
		return L_of_v
		
	def train(self):
		'''
			Train the classifier
		'''
		params = mymin(self.cost, self.model, method = 'L-BFGS-B', jac = gradient)
		self.model = params.x

	def p_y_given_x(self,h,tag):
		'''
			Take the history tuple and the required tag as the input and return the probability
		'''

	def classify(self,h):
		'''
			Performs the classification by determining the tag that maximizes the probability
		'''

	def gradient(self):
		'''
			Maximizes the log-likelihood(Minimizes the negative)
		'''
		temp = np.array([0]*10)
		for h in self.fvectors.keys():
			for tag in fv.keys():
				np.add(temp, (self.fvectors[h][tag] * self.p_y_given_x(h, tag)))
		derivative = temp - self.cum_f
		return derivative
		
	def create_dataset(self, feature_fn_list):
		dataset = {k:v for k in self.hist_list for v in [{}]}
		for i in self.hist_list:
			for tag in self.tags:
				dataset[i][tag] = [fun() for fun in feature_fn_list]
		
		return dataset