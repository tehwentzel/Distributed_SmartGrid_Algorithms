# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 19:38:16 2018

@author: Andrew
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
class GNode():
    #class for a generator node
    def __init__(_self,a,b,ulim,llim):
        _self.a = a
        _self.b = b
        _self.ulim = ulim
        _self.llim = llim
        _self.umax = (_self.ulim - _self.a)/_self.b
        _self.umin = (_self.llim - _self.a)/_self.b
        _self.p = 0

    def u(_self, x):
        u = (x - _self.a)/_self.b
        u = np.clip(u,_self.umin,_self.umax)
        return(u)

    def solvex(_self,lamda):
        if lamda < _self.umin:
            return(_self.llim)
        if lamda >= _self.umax:
            return(_self.ulim)
        else:
            return(_self.a + lamda*_self.b)

    def cost(_self,x):
        cost = ((x - _self.a)**2)/(2*_self.b)
        return(cost)

class SNode():
    #simple placeholder for a node with static demand
    def __init__(_self,p):
        _self.p = p

def consensus(array, W, maxitt = 10000, eps = .000001):
    #Runs an Eigenvector-consensus for a given list of nodes and a normalized adjacency matrix
    new = copy.copy(array)
    hist = [array[:]]
    itt = 0
    while itt < maxitt:
        old = hist[-1]
        for i in np.arange(0,len(array)):
            new[i] = np.dot(W[i,:],copy.copy(old))
        itt += 1
        if np.all(np.abs(new - old) < eps):
            return([new,hist])
        hist.append(copy.copy(new))
    return([new,hist])

def maxconsensus(array,A):
    #runs a minimum consensus algorithm given a list of nodes and an Adjacency Matrix
    diameter = len(array)
    array = copy.copy(array)
    for i in np.arange(0,diameter):
        array[i] = np.max(A[i,:]*array)
    print(array)
    return(array[0])

def minconsensus(array,A):
    #runs a minimum consensus algorithm given a list of nodes and an Adjacency Matrix
    diameter = len(array)
    array = copy.copy(array)
    for i in np.arange(0,diameter):
        array[i] = np.min(A[i,:]*array)
    return(array[0])

def bisection_algorithm(eps = .0000001):
    #define the demand and the Normalized Adjacency Matrices
    p = np.array([0,0,0,.5,1.4,1.1,.9,.2])
    ptot = np.sum(p)
    Q = np.array([1/3,1/5,0,1/4,0,0,0,0,1/3,1/5,1/5,0,0,1/5,0,1/3,0,1/5,1/5,0,0,
                  1/5,1/3,1/3,1/3,0,0,1/4,1/4,1/5,0,0,0,0,0,1/4,1/4,1/5,1/3,0,0,
                  1/5,1/5,1/4,1/4,1/5,0,0,0,0,1/5,0,1/4,0,1/3,0,0,1/5,1/5,0,0,0,
                  0,1/3]).reshape((8,8))
    R = np.array([1/2,1/3,0,1/2,1/3,1/2,0,1/3,1/2]).reshape((3,3))

    ##Initialize the Node Parameters for the Problem Graph
    n1 = GNode(-1,3,2.1,0)
    n2 = GNode(-1,2,1.0,0)
    n3 = GNode(-1,2,5.0,0)
    n4 = SNode(1)
    n5 = SNode(1.4)
    n6 = SNode(1.1)
    n7 = SNode(.9)
    n8 = SNode(.2)
    mvect = [n1,n2,n3,n4,n5,n6,n7,n8]
    nvects = [n1,n2,n3]
    lbounds = np.array([node.llim for node in nvects])
    ubounds = np.array([node.ulim for node in nvects])
    numnodes = len(mvect)
    numgens = len(nvects)
    diameter = 2


    [p,phist] = consensus(p,Q)

    sold = np.zeros((8,))
    for i in np.arange(0,numgens):
        sold[i] = p[i]
    snew = sold
    [snew,hist] = consensus(snew,Q)

    ynew = p*p/snew
    [y,yhist] = consensus(ynew[0:3],R)

    lmin = np.array([node.umin for node in nvects])
    lmax = np.array([node.umax for node in nvects])
    AR = (R > 0)
    lmin = minconsensus(lmin,AR)
    lmax = maxconsensus(lmax,AR)
    lavg = (lmin + lmax)/2
    lhist = [lavg]
    x = [node.solvex(lavg) for node in nvects]
    xhist = [x]
    xsums = [np.sum(x)]
    zhist = [copy.copy(x)]
    signhist = []
    costhist = [np.sum([mvect[i].cost(x[i]) for i in np.arange(numgens)])]
    for idx in np.arange(1,14): #bisection step
        z = copy.copy(xhist[-1])
        done = False
        while not done: #iterate over z until convergence
            for count in np.arange(0,diameter):
                zold = copy.copy(z)
                for i in np.arange(0,numgens):
                    z[i] = np.dot(R[i,:],zold[:])
            sign = (z > y)
            zhist.append(z[:])
            done = np.all(z>y - eps) or np.all(z<= y + eps)
            signhist.append(sign)
        if np.all(z>y):
            lmax = lavg
        else:
            lmin = lavg
        lavg = (lmin + lmax)/2
        lhist.append(lavg)
        x = [node.solvex(lavg) for node in nvects]
        xhist.append(x)
        xsums.append(np.sum(x))
        costhist.append(np.sum([mvect[i].cost(x[i]) for i in np.arange(numgens)]))
    return((xhist,costhist,lhist))

(x,cost,l) = bisection_algorithm()