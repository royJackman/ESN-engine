import sys
import time
import random
import functools

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.linear_model import Perceptron, SGDClassifier

"""
Copyright 2018 R. Jackman, UMass Amherst

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
"""        

###############################################################################
# scm -         The general form for the Sine Circle Mapping
# 
# Inputs:
# t:            Theta n, the current value of the cell
# o:            Omega, the driving signal
# k:            K, the nonlinearity constant
#
# Outputs:
# theta n+1     The next value in the cell
###############################################################################

def scm(t,o=0.16,k=1.0): return (t + o - (k/(2*np.pi))*np.sin(2*t*np.pi)) % 1.0

###############################################################################
# Echo State Network Class
#
# Inputs:
# reservoirSize:    The number of nodes in the entire reservoir including all
#                   input nodes and outuput nodes
# inputNodes:       The number of nodes that input will be forced through
# outputNodes:      The number of nodes that output will be read from
# iterationThr:     Size of the buffer for holding previous values to test for
#                   oscillations, in the case of 1, will not test for 
#                   oscillations, as it would be converged
# convLinit:        Limit to number iterations before solution convergence
# tolerance:        Error tolerance for delimiting teaching
# learningRate:     How much of each node is rewritten every iteration
# weightMatrix:     Internal weight matrix, directed from row to column, must
#                   be of shape (n,n)
# solver:           The weight training solver for output weights of reservoir
#                   Currently supported:
#                       adams:  Adams solver using gradients
#                   Coming soon:
#                       sgd:    Stochastic Gradient Descent
#                       batch:  Batch learning
# updateStructure:  Currently supported:
#                       hop:    Hopfield network
#                       lat:    Lattice
#                       tor:    Torus
#                   Coming soon:
#                       async:  Asynchronous updating
# matrixType:       What matrix to generate if one is not provided
#                   Currently supported:
#                       csw:    Connected small world
#                       erg:    Pseudo-random Erdos-Renyi graph
#                       ran:    Random Graph
#                   Coming soon:
#                       osci:   Forced oscillation graph (hopfield recommended)
# datatype:         Type of input data
#                   Currently supported:
#                       ts, timeseries: Continuous data
#                       st, static:     Static data
# seed:             Random seed, used for all generators
# prob:             Weight addition probability, used for weight matrix gen
# thresholdFunction:Thresholding function to replace nodes
# ins:              Input node locations, indexed from zero
# outs:             Output node locations, indexed from zero
# quiet:            If true, no text will be printed except errors, if false
#                   will print general updates
# verbose:          If true, will print more detailed updates, quiet must be
#                   false for this option
# omega:            The driving force (SCM only)
# K:                Nonlinearity constant (SCM only)
# omegas:           Array of omega values (Lattice and Torus only)
# Ks:               Array of K values (Lattice and Torus only)
# lat:              Lattice shape (Lattice and Torus only)
#
# Attributes:
# q:                quiet flag
# v:                verbose flag
# inputNodes:       Locations of input nodes (indexed at 0)
# outputNodes:      Locations of output nodes (indexed at 0)
# inp:              The number of input nodes
# outp:             The number of output nodes
# res:              Reservoir size
# s:                Random seed
# W:                Weight matrix
# stopIt:           Limit of iterations or loops of timeseries
# p:                Probability for adding edges in graph generation
# perc:             Perceptron for learning the output weights
#
# Output:
# self:             New echo state network class object
###############################################################################

class esn:
    def __init__(self,reservoirSize,inputNodes,outputNodes,iterationThr=30, convLimit=1000,tolelrance=1e-3,learningRate=0.3,weightMatrix=None, solver='adams',updateStructure='hop',matrixType='csw',datatype='st', seed=13,prob=0.34,thresholdFunction='tanh',ins=None,outs=None, quiet=False,verbose=False,omega=0.16,K=1.0,omegas=None,Ks=None,shape=(5,5)):
        # Define loquaciousness of program
        self.q = quiet
        self.v = verbose

        # Check for input node output node overlap, exit in case of overlap
        if ins != None and outs != None:
            for i in ins:
                if i in outs:
                    sys.exit("A node cannot be both in ins and outs, would bias continual timeseries prediction")

        # If locations are not defined, generate some new ones
        if outs == None and ins == None:
            nodes = random.sample(range(reservoirSize),inputNodes+outputNodes)
            self.inp = inputNodes
            self.outp = outputNodes
            self.inputNodes = nodes[:inputNodes]
            self.outputNodes = nodes[inputNodes:]
        # If only outs is undefines, generate some output locations
        elif ins != None:
            if len(ins) != inputNodes:
                if not self.q:
                    print("Length of input node array inequal to inputNodes, overriding inputNodes")
                self.inp = len(ins)
            else:
                self.inp = inputNodes
            self.inputNodes = ins
            x = range(reservoirSize)
            toUse = [i for i in x if i not in ins]
            self.outputNodes = random.sample(toUse,outputNodes)
        # And vice versa for inputs locations
        elif outs != None:
            if len(outs) != outputNodes:
                if not self.q:
                    print("Length of output node array inequal to outputNodes, overriding outputNodes")
                self.outp = len(outs)
            else:
                self.outp = outputNodes
            self.outputNodes = outs
            x = range(reservoirSize)
            toUse = [i for i in x if i not in outs]
            self.inputNodes = random.sample(toUse,inputNodes)
        
        # Save the reservoir size and the random seed for future use
        self.res = reservoirSize
        self.s = seed

        # If weight matrix is defined,
        if weightMatrix != None:
            wm = weightMatrix.shape
            # Check for correctness,
            if wm[0] != wm[1]:
                sys.exit("Predefined weight matrix is not square, cannot perform surjective mapping computation")
            if wm[0] != reservoirSize:
                if not self.q:
                    print('reservoirSize is not equal to the size of weightMatrix, using size of weightMatrix')
                reservoirSize = wm[0]
            # And assign it
            self.W = weightMatrix
        else:
            # We only need to grab probability if we are generating a weight matrix
            self.p = prob
            # Generate new matricies accordingly
            if matrixType == 'osci':
                ### TODO: insert oscillatory graph generator
                print("IMPLEMENT ME")
            elif matrixType == 'csw':
                self.W = nx.to_numpy_matrix(nx.connected_watts_strogatz_graph(self.res, int(np.floor(np.sqrt(self.res) + 1)), self.p, tries=100, seed=self.s))
            elif matrixType == 'erg':
                self.W = nx.to_numpy_matrix(nx.erdos_renyi_graph(self.res, self.p, seed=self.s, directed=True))
            elif matrixType == 'ran':
                self.W = np.random.rand(self.res,self.res)
            else:
                sys.exit("Matrix type not recognized")
        
        # Check for threshold legality
        if iterationThr > 0:
            self.stopIt = iterationThr
        else:
            if not self.q:
                print("Iteration threshold is less than 1, defaulting to 100")
        
        # Grab some dangling variables
        self.tol = tolelrance
        self.a = learningRate
        self.sol = solver
        self.up = updateStructure
        self.coefs = None
        self.cl = convLimit
        self.o = omega
        self.k = K
        self.os = omegas
        self.ks = Ks
        self.shp = shape

        # Last check for legal variables
        if datatype.lower() in ['st','static','ts','timeseries']:
            self.dt = datatype.lower()
        else:
            sys.exit("datatype not recognized")

        # Define the thresholding function
        if thresholdFunction == 'tanh':
            self.thresh = np.tanh
        else:
            self.thresh = None
        # elif thresholdFunction == 'scm0':
        #     self.thresh = scm(,o=omega,k=0.0)
        # elif thresholdFunction == 'scm1':
        #     self.thresh = scm(,o=omega,k=1.0)
                
        self.perc = SGDClassifier(random_state=seed)
        
###############################################################################
# fit -         A function to train output weights for esn
#
# Inputs:
# x:            Input array with (# of observations) rows and (size of 
#               observation) columns
# y:            Output array to train on with (# of observations) rows and 
#               (size of y actual) columns
# iterations:   Number of input repetitions, 0 for continuous until convergance
#               (overdampened)
# warn:         Overwriting warning, set to False to overwrite existing data
###############################################################################
    
    def fit(self, x, y, iterations=0, warn=True):
        # Check for trained esn, break if warn is on and net is trained
        # if self.coefs != None and warn:
        #     print("Are you sure you want to rewrite existing observations?")
        #     sys.exit("If yes, rerun with 'warn' set to False")

        # Check for correct observation shape
        if x.shape[1] != self.inp and self.dt in ['st','static']:
            sys.exit("Input size != observation size")
        
        # Alignment check
        if x.shape[0] != y.shape[0]:
            sys.exit("Number of observations != number of yactuals")
        else:
            obvs = x.shape[0]
        
        # An array to hold outputs from the network
        X = np.zeros((obvs,self.outp))

        # Loop through all data and propogate through network, storing output
        for i in range(obvs):
            X[i,:] = self.run(np.reshape(x[i,:], x.shape[1]), iterations)
                
        # Train on stored values and save into coefs array
        # self.coefs = self.train(X,y)
        self.perc = self.perc.fit(X,y)

###############################################################################
# run -     Getting the output from the reservoir on certain input
# 
# Inputs:
# inpt:     Observation input
# itr:      Iteration number, zero for overdampened
#
# Outputs:
# output:   Data from the output nodes that has converged to a solutioin
###############################################################################

    def run(self, inpt, itr):
        # Check the structure
        if self.up == 'hop':
            # Predefine buffer for convergence checking
            buffer = np.ones((self.stopIt,self.outp))


            # Predefine all reservoir nodes, convergence flag, and iterator
            fullReservoir = 0.5*np.ones((self.res,))
            converged = False
            i = 0
            
            # Loop thorough until convergence or solution limit
            while i < self.cl and not converged:
                # Force the input for the difference input types
                if self.dt in ['st','static'] and i < self.stopIt and (itr == 0 or i < itr):
                    for j in range(len(self.inputNodes)):
                        fullReservoir[self.inputNodes[j]] = inpt[j]
                elif self.dt in ['ts', 'timeseries'] and i < self.stopIt and (itr == 0 or i < len(inpt)*itr):
                    for j in range(len(self.inputNodes)):
                        fullReservoir[self.inputNodes[j]] = inpt[i%len(inpt)]
                
                # Perform hopfield step and update buffer
                if self.thresh == np.tanh:
                    fullReservoir = self.thresh(self.a*np.dot(self.W,fullReservoir) + (1-self.a)*fullReservoir)
                else:
                    fullReservoir = scm(self.a*np.dot(self.W,fullReservoir) + (1-self.a)*fullReservoir, o=self.o, k=self.k)
                    
                fullReservoir = np.resize(fullReservoir, self.res)
                for j in range(self.outp):
                    buffer[:,j] = np.hstack((buffer[1:,j],fullReservoir[self.outputNodes[j]]))

                # Begin convergence checks when buffer is filled
                if i > self.stopIt:
                    converged = self.converge(buffer)
                
                # Iterate iterator iteratively
                i += 1
            
            # if convergence is not reached, send error
            # if not converged:
            #     sys.exit("Solution not converged in limit")
            # else:
            #     # Else return the related information
            #     g = self.converge(buffer, boo=False)
            #     m = int(max(g))
            return buffer[-1:,:] #np.mean(buffer[-m:,:], axis=1) if m > 1 else 

        elif self.up == 'lat':
            if self.os != None:
                if self.shp != self.os.shape:
                    sys.exit('Lattice unaligned with omega array')
            else:
                self.os = self.o * np.ones(self.shp)
            
            if self.ks != None:
                if self.shp != self.ks.shape:
                    sys.exit('Lattice unaligned with K array')
            else:
                self.ks = self.k * np.ones(self.shp)

            for i in range(self.shp[1]):
                for j in range(self.shp[0]):
                    inpt[j] = scm(inpt[j], self.os[j,i], self.ks[j,i])
            
            return inpt

        elif self.up == 'tor':
            if self.os != None:
                if self.shp != self.os.shape:
                    sys.exit('Lattice unaligned with omega array')
            else:
                self.os = self.o * np.ones(self.shp)
            
            if self.ks != None:
                if self.shp != self.ks.shape:
                    sys.exit('Lattice unaligned with K array')
            else:
                self.ks = self.k * np.ones(self.shp)

            # Predefine buffer for convergence checking
            buffer = np.ones((self.stopIt,len(inpt)))
            
            for i in range(self.shp[0]):
                buffer[0:i] = inpt[i]

            converged = False
            i = 0
            j = 0

            while i < self.cl and not converged:
                temp = inpt

                temp[j] = scm(inpt[j], self.os[j,i], self.ks[j,i])

                for k in range(self.shp[0]):
                    t = sum(temp[k-2:k+1])
                    inpt[k] = t
                    buffer[(i+1)%self.stopIt,k] = t
                
                if i > self.stopIt:
                    converged = self.converge(buffer)

                if j % self.shp[0] == 0:
                    i += 1
            
            # if convergence is not reached, send error
            if not converged:
                sys.exit("Solution not converged in limit")
            else:
                # Else return the related information
                g = self.converge(buffer, boo=False)
                m = int(max(g))
                return buffer[-1:,:] #np.mean(buffer[-m:,:], axis=1)


###############################################################################
# train -   Train an output weight matrix from reservoir output and yactual
#
# Inputs:
# X:        Outputs from reservoir
# y:        yactual
#
# Outputs:  
# Theta:    Trained perceptron weight matrix
###############################################################################

    def train(self, X, y):
        # Check solver type
        if self.sol == 'adams':
            #Perform the Adam optimizaion algorithm from arXiv:1412.6980
            stepsize = 0.001
            beta1 = 0.9
            beta2 = 0.999
            theta = np.ones((X.shape[1],y.shape[1]))
            converged = False
            m = 0
            v = 0
            t = 0
            while not converged and t < self.cl:
                t += 1
                g = np.gradient(theta, axis=0)*self.objective(theta,X,y)
                m = beta1*m + (1-beta1)*g
                v = beta2*v + (1-beta2)*np.square(g)
                mhat = m/(1-(beta1**t))
                vhat = v/(1-(beta2**t))
                theta = theta - (stepsize*mhat)/(np.sqrt(vhat) + self.tol)
                converged = self.converge(theta)

            # If convergence is met, return theta, else break
            if not converged:
                sys.exit('self.coef did not converge within limit')
            else:
                return theta

###############################################################################
# converge -    Returns either boolean or int values representing the frequency
#
# Inputs:
# series:       The continuous series were checking for convergence in (size of
#               buffer) rows by (number of variables)
# boo:          If true, return boolean determining convergence to either 
#               stable state or oscillation
#
# Outputs:
# retval:       (boolean) True if converged, else false
#               (list-like, dtype=int) The wavelength of series, wavelength of
#               1 is a constant
###############################################################################

    def converge(self, series, boo=True):
        # Predefine data holding matrix to values that are not 0 and retval 
        datum = -1 * np.ones((series.shape[0],series.shape[0]))
        retval = np.zeros((series.shape[1],))
        series = series.T

        # Loop through the number of variables
        for i in range(series.shape[0]):
            # Then loop through possible wavelengths
            for j in range(1,series.shape[1]-1):
                # Then loop through coupled indecies 
                for k in range(series.shape[1] - j):
                    # Find the error
                    temp = abs(series[i,k]-series[i,j+k])

                    # If within tolerance, store it, else break
                    if temp <= self.tol:
                        datum[j,k] = temp
                    else:
                        break

                # Check for confirmed wavelength, save if found
                if datum[j,k] != -1 and datum[j,k] <= self.tol:
                    retval[i] = j
                    break
            
            # If not converged, decide what to do
            if retval[i] != j:
                if boo:
                    return False
                else:
                    retval[i] = 0
        
        # Final return value
        if boo:
            return True
        else:
            return retval
    
###############################################################################
# objective -   A function that returns the error of the current weight matrix
#
# Input:
# curr:         The current weight matrix guess
# x:            ESN output
# y:            yactual, to test against
# p:            If true, return a percentage from 0.0 to 1.0, else return MSE
#
# Output:
# return value: Mean Squared Error of the current weights
###############################################################################

    def objective(self, curr, x, y, p = False):
        # Preset total for scoring later
        total = 0.0

        # Loop through examples, add the L2 norm of the product of the example
        # and the array
        for i in range(x.shape[0]):
            if p:
                total += (1 if abs(self.perc.predict(x[i,:].reshape(1,-1)) - y[i,:]) < self.tol else 0)
            else:
                total += np.linalg.norm(self.perc.predict(x[i,:].reshape(1,-1)) - y[i,:])

        # Return total/numOfEntries, or the average
        return total/x.shape[0]

###############################################################################
# score -       A function that scores the reservoir on test set
#
# Input:
# x:            ESN output
# y:            yactual, to test against
# raw:          True if the data being scored has not been run through the 
#               reservoir
# p:            Fercentage flag for objective function
#
# Output:
# Score:        The score of the model, either in MSE or percentage
###############################################################################

    def score(self, x, y, raw = False, p = True):
        if raw:
            # An array to hold outputs from the network
            X = np.zeros((x.shape[0],self.outp))

            # Loop through all data and propogate through network, storing 
            # output (This code looks...familiar..)
            for i in range(x.shape[0]):
                X[i,:] = self.run(x[i,:], 0)
        else:
            X = x

        # Return the l2 norm as error
        return self.objective(self.coefs, X, y, p=p)