# -*- coding:utf-8 -*-
__author__ = 'mamoru'
"""
 This script is test program of "Adaptability and Dieversity
 in Simulated Turn-taking Behavior H. Lizuka, T.Ikegami 2007"
"""
from Agent import Agent
import numpy as np
import matplotlib.pyplot as plt
import pprint

if __name__ == "__main__":
	print("start")
	agent1 = Agent(P0=np.array([ [0.0], [0.0]]))
	agent2 = Agent(P0=np.array([[70.0], [0.0]]))
	for i in range(1000):
		agent1.do_1step(agent2.position(), agent2.head_angle())
