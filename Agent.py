# -*- coding:utf-8 -*-
__author__ = 'mamoru'
"""
 This script is test program of "Adaptability and Dieversity
 in Simulated Turn-taking Behavior H. Lizuka, T.Ikegami 2007".
 Especialy , this code defines Agent model.
"""
import numpy as np
import theano
import theano.tensor as T
from theano import function
from theano import Param
from theano import pp
import pprint

def dv_dt(t, v, M, D1, f1, f2):
    """
    equation (1)
    :param t: time
    :param v: velocity
    :param M: parameter
    :param D1: parameter
    :param f1: parameter
    :param f2: parameter
    :return dvdt
    """
    return -1.0*(D1*v + f1 + f2)/M

def dtheta_dt(t, theta_dot, I, D2, f1, f2):
    """
    equation (2)
    :param t: time
    :param theta_dot: theta
    :param I: inertia
    :param D2: D2
    :param f1: f1
    :param f2: f2
    :return:
    """
    return -1.0*(D2*theta_dot + f1 - f2)/I

def f_step_exp(t0, y0, h, dydt_exp, *args):
    """
    :param t0:
    :param y0:
    :param h: fixed time step
    :param dydt_exp: expression of dy/dt
    :param *args: parameters of expression
    :return:
    """
    half_h = h / 2

    k1 = h * dydt_exp(t0, y0, *args)

    t2 = t0 + half_h
    y2 = y0 + (k1/2)
    k2 = h * dydt_exp(t2, y2, *args)

    y3 = y0 + (k2/2)
    k3 = h * dydt_exp(t2, y3, *args)

    t4 = t0 + h
    y4 = y0 + k3
    k4 = h * dydt_exp(t4, y4, *args)

    yi = y0 + (k1 + 2*k2 + 2*k3 + k4)/6
    return yi

class Agent:
    def __init__(self, P0 = np.array([[0.0], [0.0]]), V0 = 0.0, Theta0 = 0.0,
                    M = 0.5, Inertia = 0.5, D1 = 0.5, D2 = 0.5,
                    I = 3, K = 5, J = 10, L = 3,
                    DT1 = 0.01, NTRatio=100,
                    r = 150.0, phi = 0.5235987755982988):
        """
        Agent model.
        This model has 2 motors that can move 2D field and has
        some sensors that can know other agent's position.
        Behavior of this agent is calculated by Recurrent
        Neural Network(RNN). RNN has I input nodes , K output nodes,
        J hidden layer nodes and L context nodes.
        :param M: mass
        :param Inertia: Inertia
        :param D1: resitance coefficients
        :param D2: resitance coefficients
        :param I: number of input nodes
        :param K: number of output nodes
        :param J: number of hidden layer's nodes
        :param L: number of context nodes.
        :param r: radius of RS
        :param phi: angle of RS, default value is 30[deg]
        :return:
        """
        self.I = I # number of input nodes
        self.K = K # number of output nodes (predict:3, motor:2)
        self.J = J # number of hidden layer's nodes
        self.L = L # number of context nodes

        # Recurrent neural network
        self.rnn = RNN(self.I, self.K, self.J, self.L, True)

        # context
        self.C0  = np.random.rand(self.L, 1)

        # position(x,y)
        self.Positions = [P0]

        # velocity
        self.VS = [V0]

        # angle[rad]
        self.Thetas = [Theta0]

        # mass and inertia
        self.M, self.Inertia = M, Inertia

        # resistance coefficients
        self.D1, self.D2 = D1, D2

        # time step
        self.DT1, self.DT2 = DT1, NTRatio*DT1
        self.NTRatio = NTRatio
        self.t_cnt = 0
        self.TS = [0.0]

        self.f1, self.f2 = 0.5, 0.5

        # make functions of eq(1) and eq(2)
        M_  = T.dscalar('M')
        D1_ = T.dscalar('D1')
        f1_ = T.dscalar('f1')
        f2_ = T.dscalar('f2')
        I_  = T.dscalar('I')
        D2_ = T.dscalar('D2')

        t0 = T.dscalar('t0')
        v0 = T.dscalar('v0')
        theta0 = T.dscalar('theta0')
        h = T.dscalar('h')

        # t, v, M, D1, f1, f2):
        # f_step_exp(t0, y0, h, dydt_exp, *args):
        dvdt_step = f_step_exp(t0, v0, h, dv_dt, M_, D1_, f1_, f2_)
        self.dvdt_fn = function([t0, v0, h, M_, D1_, f1_, f2_], dvdt_step, on_unused_input='ignore')

        #dtheta_dt(t, theta_dot, I, D2, f1, f2)
        dtheta_dt_step = f_step_exp(t0, theta0, h, dtheta_dt, I_, D2_, f1_, f2_)
        self.dtheta_dt_fn = function([t0, theta0, h, I_, D2_, f1_, f2_], dtheta_dt_step, on_unused_input='ignore')

        # other agent in in area of RE
        self.r, self.phi = r, phi
        self.in_rs = False


    def position(self):
        """
        current agent's position
        :return:
        """
        return self.Positions[-1]

    def head_angle(self):
        """
        agent's cuurent head angle
        :return:
        """
        return self.Thetas[-1]

    def set_weights(self, W, Wdash, U, Udash, B1, B2, B3):
        """
        update RNN's weight vectors
        :param W: weight vector of eq.3. input->hidden layer's weight
        :param Wdash: context->hidden layer's weight
        :param U: hidden -> output layer's weight
        :param Udash: hidden -> context layer's weight
        :param B1: bias nodes of hidden layer
        :param B2: bias nodes of output layer
        :param B3: bias nodes of context layer
        :return:
        """
        self.rnn.update_weights(W, Wdash, U, Udash, B1, B2, B3)

    def do_1step(self, p_other, h_angle):
        """
        calculate agents at one step.
        :param p_other: position of other agent
        :param h_angle: heading angle of other agent
        :return:
        """
        self.is_in_RS = self.f_is_in_RS(p_other, h_angle)

        # calculate velocity and head_angle
        t = self.TS[-1] + self.DT1
        v     = self.dvdt_fn(t, self.VS[-1], self.DT1, self.M, self.D1, self.f1, self.f2)
        theta = self.dtheta_dt_fn(t, self.Thetas[-1], self.DT1, self.I, self.D2, self.f1, self.f2)
        
        # calculate position of agent
        x = self.Positions[-1][0] + v * np.cos(theta)
        y = self.Positions[-1][1] + v * np.sin(theta)

        # update parameters
        self.TS.append(t)
        self.VS.append(v)
        self.Thetas.append(theta)
        self.Positions.append([x,y])
        self.t_cnt += 1

        # execute rnn
        if (self.t_cnt%self.NTRatio) == 0:
            theta = np.arctan2(self.Positions[-1][1][0] - p_other[1][0], self.Positions[-1][0][0] - p_other[0][0])
            dist = np.linalg.norm(self.Positions[-1] - p_other)
            Z, C1 = self.rnn.calc(np.array([[theta], [dist], [h_angle]]), self.C0)
            self.C0 = C1
            f1, f2 = Z[3][0], Z[4][0]

    def f_is_in_RS(self, p_other, h_angle):
        """

        """
        theta = np.arctan2(self.Positions[-1][1][0] - p_other[1][0], self.Positions[-1][0][0] - p_other[0][0])
        dist = np.linalg.norm(self.Positions[-1] - p_other)

        t = self.Thetas[-1] - np.pi
        phi_min = t - self.phi
        phi_max = t + self.phi
        if t >= phi_min and t <= phi_max:
            if dist < self.r:
                return True
        return False


class RNN(object):
    def __init__(self,I,K,J,L,is_rand=False):
        """
        Recurrent Neural Network
        :param I:
        :param K:
        :param J:
        :param L:
        :return:
        """
        self.I = I
        self.K = K
        self.J = J
        self.L = L

        # weight vector of eq.3
        self.W     = np.zeros((self.J, self.I)) if not is_rand else np.random.rand(self.J, self.I) * 2.0 - 1.0

        # weight vector of eq.3
        self.Wdash = np.zeros((self.J, self.L)) if not is_rand else np.random.rand(self.J, self.L) * 2.0 - 1.0

        # bias vector of eq.3
        self.B1    = np.zeros((self.J, 1))      if not is_rand else np.random.rand(self.J, 1) * 2.0 - 1.0

        # weight vector of eq.4
        self.U     = np.zeros((self.J, self.K)) if not is_rand else np.random.rand(self.K, self.J) * 2.0 - 1.0

        # bias vector of eq.4# bias vector of eq.4
        self.B2    = np.zeros((self.K, 1))      if not is_rand else np.random.rand(self.K, 1) * 2.0 - 1.0

        # weight vector of eq.5
        self.Udash = np.zeros((self.J, self.L)) if not is_rand else np.random.rand(self.L, self.J) * 2.0 - 1.0

        # bias vector of eq.5
        self.B3    = np.zeros((self.L, 1))      if not is_rand else np.random.rand(self.L, 1) * 2.0 - 1.0

        x = T.dmatrix('x')
        s = 1 / (1 + T.exp(-x))
        self.sigmoid = theano.function(inputs=[x], outputs = s) # sigmoid function

    def update_weights(self, W, Wdash, U, Udash, B1, B2, B3):
        """
        update RNN's weight vectors
        :param W: weight vector of eq.3. input->hidden layer's weight
        :param Wdash: context->hidden layer's weight
        :param U: hidden -> output layer's weight
        :param Udash: hidden -> context layer's weight
        :param B1: bias nodes of hidden layer
        :param B2: bias nodes of output layer
        :param B3: bias nodes of context layer
        :return:
        """
        self.W = W
        self.Wdash = Wdash
        self.U = U
        self.Udash = Udash
        self.B1 = B1
        self.B2 = B2
        self.B3 = B3

    def calc(self, Y, C_t0):
        """
        calculate RNN from input vector Y and context vector C(t-1) and
        return output vector Z and context vector C(t)
        :param Y: input vector
        :param C_t0: context that is C(t-1)
        :return:
        """
        # calculate outputs of hidden layer
        H0 = self.W.dot(Y) + self.Wdash.dot(C_t0) + self.B1
        H  = self.sigmoid(H0)

        # calculate outputs of output layer
        Z0 = self.U.dot(H) + self.B2
        Z  = self.sigmoid(Z0)

        # calculate outputs of context layer
        C_t00 = self.Udash.dot(H) + self.B3
        C_t   = self.sigmoid(C_t00)

        return Z, C_t

