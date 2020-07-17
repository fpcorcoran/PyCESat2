import matplotlib.pyplot as plt

class waveForm:
	def __init__(self, h, n):
		self.height = h
		self.count = n

	def plot(self):
		plt.figure(figsize=(6,6))
		plt.barh(self.height, self.count)
		plt.show()
