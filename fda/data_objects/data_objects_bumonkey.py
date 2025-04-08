import numpy as np 
import torch 
from torch.utils.data import Dataset

class ObservationPoints3D(Dataset):
	def __init__(self, X1,X2,X3,X4,X5, Y, batch_size):
		"""
		X: N x 2, N spatial coordinates
		Y: N,, N measurements
		"""
		super().__init__()

		self.N = X1.shape[0]
		self.D = X1.shape[1]
		self.Y = Y
		self.X1 = X1
		self.X2 = X2
		self.X3 = X3
		self.X4 = X4
		self.X5 = X5
		self.batch_size = batch_size

	def __len__(self):
		return 1

	def __getitem__(self, idx):
		rix = np.random.choice(self.N, size=self.batch_size, replace=False)
		coord1 = self.X1[rix, :]
		coord2 = self.X2[rix, :]
		coord3 = self.X3[rix, :]
		coord4 = self.X4[rix, :]
		coord5 = self.X5[rix, :]
		yvals = self.Y[rix, :]
		return {"coord1": torch.from_numpy(coord1),"coord2": torch.from_numpy(coord2),"coord3": torch.from_numpy(coord3),"coord4": torch.from_numpy(coord4), "coord5": torch.from_numpy(coord5)}, {"yvals": torch.from_numpy(yvals)}

	def getfulldata(self):
		N = 27095040
		return {"coord1": torch.from_numpy(self.X1[0:N,:]),"coord2": torch.from_numpy(self.X2[0:N,:]),"coord3": torch.from_numpy(self.X3[0:N,:]),"coord4": torch.from_numpy(self.X4[0:N,:]), "coord5": torch.from_numpy(self.X5[0:N,:])}, {"yvals": torch.from_numpy(self.Y[0:N,:])}



