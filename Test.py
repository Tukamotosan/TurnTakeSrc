# -*- coding:utf-8 -*-
"""
 This is test script that calculate eq.(1) by using
 Runge-kuttta method.
"""
__author__ = 'mamoru'

import numpy as np
import theano.tensor as T
from theano import function
from theano import Param
from theano import pp
import matplotlib.pyplot as plt

def dvdt(t, v, M, D1, f1, f2):
	"""
	equation 1
	:param t: time
	:param v: velocity
	:param M: parameter
	:param D1: parameter
	:param f1: parameter
	:param f2: parameter
	:return dvdt
	"""
	return -1.0*(D1*v + f1 + f2)/M

def dvdt_step_exp(t0, v0, h, dvdt_exp, *args):
	"""
	step expression for eq.(1)
	:param t0:
	:param v0:
	:param h:
	:param dvdt_exp:
	:param *args:
	:return :
	"""
	half_h = h / 2

	k1 = h * dvdt_exp(t0, v0, *args)

	t2 = t0 + half_h
	v2 = v0 + (k1/2)
	k2 = h * dvdt_exp(t2, v2, *args)

	v3 = v0 + (k2/2)
	k3 = h * dvdt_exp(t2, v3, *args)

	t4 = t0 + h
	v4 = v0 + k3
	k4 = h * dvdt_exp(t4, v4, *args)

	vi = v0 + (k1 + 2*k2 + 2*k3 + k4)/6
	return vi


if __name__ == "__main__":
	print("start")

	D1 = T.dscalar('D1') # param of eq.1
	M  = T.dscalar('M')  # param of eq.1
	f1 = T.dscalar('f1') # param of eq.1
	f2 = T.dscalar('f2') # param of eq.1

	t0 = T.dscalar('t0') # time at step 0
	v0 = T.dscalar('v0') # velocity at step 0
	h = T.dscalar('h')   # fixed time step

	vstep = dvdt_step_exp(t0, v0, h, dvdt, M, D1, f1, f2)
	vstep_fn = function([t0, v0, h, M, D1, f1, f2], vstep, on_unused_input='ignore')

	v = 0
	time = 50
	step = 0.001
	n_steps = int(time / step)
	t = 0

	# outputs
	T = np.zeros(n_steps)
	V = np.zeros(n_steps)

	# static parameters
	f1_, f2_ = 0.5, 0.5
	D1_ = 0.5
	M_ = 0.5

	# loop for time!
	for i in range(n_steps):
		t = i * step
		v = vstep_fn(t, v, step, M_, D1_, f1_, f2_)
		T[i] = t
		V[i] = v

	fig1 = plt.figure()
	plt.plot(T, V)
	plt.show()

	print("end")



























