import numpy as np
import pandas as pd

class DataSet():
	""" Class for creating DataSet_a for training and DataSet_b for validation
	:param str excel_file: path of the data excel """
	def __init__(self, excel_file):
		self.excel_file = excel_file

	def get_set(self):
		df = pd.read_excel(self.excel_file, sheet_name=0)
		set_all = df.values
		len_set = len(set_all)
		len_parameters = len(set_all[0])
		print(set_all)
		np.random.shuffle(set_all)
		return set_all, len_parameters

	def normal_set(self, set_all):
		max_paramaters = set_all.max(0)
		normal_set_all = np.array(set_all/max_paramaters)
		size = int(len(normal_set_all)/2)
		set_a_result = normal_set_all[:int(size/10), -1]
		set_a = normal_set_all[:int(size/10),: -1]
		set_b_result = normal_set_all[size:, -1]
		set_b = normal_set_all[size:, :-1]
		
		set_bt = set_all[size:, :-1]
		set_bt_result = set_all[size:, -1]

		print(len(set_a), len(set_b))
		return set_a, set_a_result, set_b, set_b_result, set_bt, set_bt_result,  max_paramaters[:-1], max_paramaters[-1]

class Model():
	""" datasets ---> linear regression parameters """ 
	def __init__(self, set_a, set_a_result, set_b, set_b_result, set_b_real, set_b_real_result, max_paramaters, max_paramaters_result, alpha, len_parameters):
		self.set_a = set_a
		self.set_a_result = set_a_result
		self.set_b = set_b
		self.set_b_result = set_b_result
		self.alpha = alpha
		self.len_parameters = len_parameters
		self.parameters = np.random.rand(self.len_parameters - 1)
		self.len_set_a = len(self.set_a)
		self.set_b_real = set_b_real
		self.set_b_real_result = set_b_real_result
		self.max_paramaters = max_paramaters
		self.max_paramaters_result = max_paramaters_result


	def find_row_sum(self, row):
		row_sum = np.sum(np.dot(row, self.parameters))
		return row_sum
	
	def hypothesis(self, row):
		y = 1/(1+np.exp(-(np.dot(self.parameters, row))))
		return y

	def logistic_error(self):
		diff_sum = 0
		for i in range(0, self.len_set_a):
			row_sum = self.hypothesis(self.set_a[i])
			y = self.set_a_result[i]
			diff_sum += y*np.log(row_sum) + (1-y)*np.log(1- row_sum)
			#print(row_sum, self.set_a_result[i])
			#print(self.parameters)
		return -diff_sum/self.len_set_a
	
	def update_parameters(self):
		parameters_new = []
		for j in range(0, len(self.parameters)):
			diff_sum = 0
			for i in range(0, self.len_set_a):
				row_sum = self.hypothesis(self.set_a[i])
				diff_sum += (row_sum - self.set_a_result[i]) * self.set_a[i, j] 
			theta = self.parameters[j] - self.alpha * diff_sum/self.len_set_a
			parameters_new.append(theta)
		self.parameters = np.array(parameters_new)

	def validate(self):
		self.set_a = self.set_b
		self.set_a_result = self.set_b_result
		print(f"testing set_b... the error is {self.logistic_error()}")
		errors = 0
		for i in range(0, len(self.set_b)):
			x = int(self.hypothesis(self.set_b[i]) > 0.5)
			print(np.abs(self.set_b_result[i] - x))
			errors += np.abs(self.set_b_result[i] - x)
		print(f"{int(errors/len(self.set_b_result)*100)}%")


	def training(self):
		while self.logistic_error() > 0.01:
			self.update_parameters()
			print(self.logistic_error())
		print(f"final result of set_a: {self.parameters} len parameters {len(self.parameters)}")

def main():
	dataset = DataSet(r"exp4.xlsx")
	set_all, len_parameters = dataset.get_set()
	set_a, set_a_result, set_b, set_b_result, set_b_real, set_b_real_result, max_paramaters, max_paramaters_result = dataset.normal_set(set_all)

	model = Model(set_a, set_a_result, set_b, set_b_result, set_b_real, set_b_real_result, max_paramaters, max_paramaters_result, 2, len_parameters)
	model.training()
	model.validate()

if __name__=='__main__':
	main()