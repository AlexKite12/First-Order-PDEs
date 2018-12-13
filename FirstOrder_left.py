import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
import copy

class Profiler(object):
    def __enter__(self):
        self._startTime = time.time()
         
    def __exit__(self, type, value, traceback):
        print ("Elapsed time: {:.3f} sec".format(time.time() - self._startTime))

class First_Order_PDE_left(object):
	"""
	PDE is presented as df/dx + Q*df/dt = func(t,x)
	"""
	def __init__(self, y, t, x, function = 0, function_left = 0, Q = 1, method = 'explicit'):
		"""
		function_left is left boandary; f(t)
		"""
		self.f0 = y
		self.f = np.empty_like(y)
		self.t = t
		self.x = x
		self.Q = Q
		self.method = method
		self.func = function
		self.left_bx = function_left
		self.__initialize()

	def __initialize(self):
		self.Nt = len(self.t)
		self.Nx = len(self.x)
		self.dt = self.t[-1] - self.t[-2]
		self.dx = self.x[-1] - self.x[-2]
		self.bx = self.Q * self.dt / self.dx
		if self.method == 'explicit':
			self.ax = 1 - self.Q * self.dt / self.dx
		elif self.method == 'implicit':
			self.ax = 1 + self.Q * self.dt / self.dx
		elif self.method == 'z-shema':
			self.ax = 1 + self.Q * self.dt / (2 * self.dx)
			self.bx = 1 - self.Q * self.dt / (2 * self.dx)
			self.cx = self.Q * self.dt / (2 * self.dx)

	def __solve_ex(self):
		U = np.empty((self.Nt, self.Nx))
		U[0, :] = self.f[:]
		for nt in range(1, self.Nt):
			self.f[0] = self.left_bx(self.t[nt])
			for i in range(1, self.Nx):
				self.f[i] = self.ax * self.f0[i] + self.bx * self.f0[i-1] + self.dt * self.func(self.t[nt], self.x[i])
			U[nt, :] = self.f[:]
			self.f0 = copy.copy(self.f)
		return U

	def __solve_im(self):
		U = np.empty((self.Nt, self.Nx))
		U[0, :] = self.f0[:]
		for nt in range(1, self.Nt):
			self.f[0] = self.left_bx(self.t[nt])
			for i in range(1, self.Nx):
				self.f[i] = (self.f0[i] + self.bx * self.f[i-1] + self.dt * self.func(self.t[nt], self.x[i])) / self.ax
			U[nt, :] = self.f[:]
			self.f0 = copy.copy(self.f)
		return U

	def __solve_z(self):
		U = np.empty((self.Nt, self.Nx))
		U[0, :] = self.f0[:]
		for nt in range(1, self.Nt):
			self.f[0] = self.left_bx(self.t[nt])
			for i in range(1, self.Nx - 1):
				self.f[i] = (self.ax * self.f0[i] + self.cx * (self.f[i-1] - self.f0[i+1]) + self.dt * self.func(self.t[nt], self.x[i])) / self.ax
			self.f[-1] = (self.bx * self.f0[-1] + self.cx * (self.f[-2] + self.f0[-2]) + self.dt * self.func(self.t[nt], self.x[-1])) / self.ax

			U[nt, :] = self.f[:]
			self.f0 = copy.copy(self.f)
		return U

	def solve(self):
		if self.method == 'explicit':
			return self.__solve_ex()
		elif self.method == 'implicit':
			return self.__solve_im()
		elif self.method == 'z-shema':
			return self.__solve_z()

def left_bx(t, A = 13):
	return A * t

def psi_func(t, x, A = 13):
	return A * np.exp(x) * (1 + t)

def dfdt(y, t, x_min, x_max, Nx, A):
    x = np.linspace(x_min, x_max, Nx)
    dx = x[-1] - x[-2]
    df = np.empty_like(y)
    df[0] = A * t
    df[1:] = -np.diff(y, 1) / dx + A * np.exp(x[1:]) * (1 + t)
    return df

def main():
	Q = 1
	Nx = 1000
	Nt = 1000
	x_min = 0
	x_max = 1

	x = np.linspace(0, 1, Nx)
	t = np.linspace(0, 1, Nt)
	f0 = np.zeros(Nx)

	with Profiler() as f1:
		u_ex = First_Order_PDE_left(f0, t, x, psi_func, left_bx).solve()
	#print(u_ex[-1])
	plt.plot(x, u_ex[-1], label = 'explicit method', linewidth = 2)
	np.savetxt('arrayexplicit.txt', u_ex[-1])

	# with Profiler() as f3:
	# 	f_stand = odeint(dfdt, f0, t, args=(x_min, x_max, Nx, 13))
	# plt.plot(x, f_stand[-1], label='standard')

	with Profiler() as f2:
		u_im = First_Order_PDE_left(f0, t, x, psi_func, left_bx, method = 'implicit').solve()
	plt.plot(x, u_im[-1], label='implicit method', linewidth = 1)
	np.savetxt('array_implicit.txt', u_im[-1])

	np.savetxt('array_Absolute.txt', u_im[-1])


	plt.legend()
	plt.show()
	plt.close()
	axes = plt.gca()
	axes.set_xlim([0.5,0.51])
	plt.plot(x, u_ex[-1], label = 'explicit', linewidth = 2)
	plt.plot(x, u_im[-1], label='implicit method', linewidth = 1)
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
