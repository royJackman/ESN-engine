import sys
import time
import random
import functools

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from graphBuilder import graphGen
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
# out -         A printing function for important messages.
#
# Input:
# str:          String to be printed
# v:            Verbose addition
###############################################################################

def out(str,quiet,verbose,v=''):
    # If not quiet
    if not quiet:

        # And string contains some value
        if str != '':
            # Print it!
            print(str)
        
        # Same check for the verbose, but only if !quiet is true
        if verbose and (v != ''):
            print(v)

###############################################################################
# graphGen -    Oscillatory weight matrix builder. This function creates a 
#               graph that has only odd directed cycles, which in an echo state
#               network incentivises oscillations which is the frequency of the 
#               odd cycles. 
#
# Inputs:
# n:            Size of the reservoir
# p:            Probability for building new edges on odd cycles
# seed:         Random seed for reproducible
#
# Output:
# retval:       Oscillatory weight matrix
###############################################################################

# weights = [-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

# def graphGen(n=128, p=0.3, seed=1618, circ=False, verbose=False):
#     # Set the random seed for reproducibility
#     np.random.seed(seed)

#     # Define weight matrix
#     retval = np.zeros((n,n))

#     if not circ:
#         # Loop through nodes
#         for x in range(n):

#             # Make it into a ring oscillator using the power of math
#             retval[x,(x+1)%n] = np.random.randint(-15,16)

#             # Loop through the rest of the nodes
#             for y in range(n):

#                 # If it will make an odd loop...
#                 if abs(x - y)%2 == 0 and y != (x-1) and y != x:

#                     # Give it some probability
#                     t = np.random.rand()

#                     # Add edge if the random number was within range
#                     if t <= p:
#                         retval[x,y] = np.random.randint(-15,16)
    
#     else:
#         # Make a degree 2 ring oscillator first
#         for x in range(n):
#             retval[x,(x+1)%n] = random.choice(weights)
#             retval[(x+2)%n,x] = random.choice(weights)

#         # Build the fractal pattern within the network by filling a loops array
#         # then adding the edges, emptying it, and repeating with a loop half
#         # the size of the last 
#         o = n
#         loops = []

#         # At size 4 and below, nothing else is needed because it was taken care
#         # of during the degree 2 ring oscillator
#         while o > 4:
#             if verbose: print(o)
            
#             # If the loop is odd...
#             if o % 2 == 1:
#                 # Get the 'half'
#                 p = (o-1)/2
#                 # If the loops array is empty...
#                 if len(loops) == 0:
#                     # Must be the first one, so add two loops and that's it
#                     loops.append((0,p+1))
#                     loops.append((p,0))
#                 # If the loops array is NOT empty...
#                 else:
#                     # Copy the existing loops and reset the loops array
#                     temp = loops
#                     loops = []
#                     # Iterate through the existing loops and cut them in two
#                     for t in temp:
#                         loops.append(((t[1]+p),t[1]))
#                         loops.append((t[0],(t[1]+p)))
#             # If the loop is even...
#             else:
#                 # Do things that are similar
#                 p = (o-2)/2
#                 if len(loops) == 0:
#                     loops.append((p,0))
#                     loops.append((-1,p+1))
#                 else:
#                     temp = loops
#                     loops = []
#                     for t in temp:
#                         loops.append(((t[1]+p),t[1]))
#                         loops.append((t[0],(t[1]+p+1)))
#             o = p+1
#             if verbose: print(loops)
#             for l in loops:
#                 retval[int(l[0]),int(l[1])] = random.choice(weights)
#     # Return the weight matrix
#     return retval

###############################################################################
# scm -         The general form for the Sine Circle Mapping. This does not 
#               include the alpha term as it is not used in any of the current
#               supported models.
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
# Echo State Network Class - 
#       This class contains all of the models presented in the paper, and all 
#       of their possible options. The class architecture was built in the 
#       image of a scikit-learn model, and follows their terminology with terms
#       such as 'score', 'fit', and 'train'. Some instances of this class will 
#       not require all inputs, denoted by various default None values. 
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
        # Start Timer
        st = time.time()

        # Define loquaciousness of program
        self.q = quiet
        self.v = verbose
        out('This model will not be verbose...', self.q, self.v, 'Just kidding!')

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
            out('Input and output locations generated', self.q, self.v, 'Inputs located at nodes ' + str(self.inputNodes) + ' and outputs located at nodes ' + str(self.outputNodes))

        # If only outs is undefined, generate some output locations
        elif ins != None:
            if len(ins) != inputNodes:
                if not self.q:
                    out("Length of input node array inequal to inputNodes, overriding inputNodes", self.q, self.v)
                self.inp = len(ins)
            else:
                self.inp = inputNodes
            self.inputNodes = ins
            x = range(reservoirSize)
            toUse = [i for i in x if i not in ins]
            self.outputNodes = random.sample(toUse,outputNodes)
            out("Output nodes generated", self.q, self.v, "Outputs located at nodes " + str(self.outputNodes))

        # And vice versa for inputs locations
        elif outs != None:
            if len(outs) != outputNodes:
                out("Length of output node array inequal to outputNodes, overriding outputNodes", self.q, self.v)
                self.outp = len(outs)
            else:
                self.outp = outputNodes
            self.outputNodes = outs
            x = range(reservoirSize)
            toUse = [i for i in x if i not in outs]
            self.inputNodes = random.sample(toUse,inputNodes)
            out("Input nodes generated", self.q, self.v, "Inputs located at nodes " + str(self.inputNodes))
        
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
                out('reservoirSize is not equal to the size of weightMatrix, using size of weightMatrix', self.q, self.v)
                reservoirSize = wm[0]

            # And assign it
            self.W = weightMatrix
        else:
            # We only need to grab probability if we are generating a weight matrix
            self.p = prob

            # Generate new matricies accordingly
            if matrixType == 'osci':
                self.W = graphGen(self.res, p=prob, seed=self.s, type='osci')
                out("Oscillatory graph generated successfully", self.q, self.v)
            elif matrixType == 'csw':
                self.W = nx.to_numpy_matrix(nx.connected_watts_strogatz_graph(self.res, int(np.floor(np.sqrt(self.res) + 1)), self.p, tries=100, seed=self.s))
                out("Connected Small World graph generated successfully", self.q, self.v)
            elif matrixType == 'erg':
                self.W = nx.to_numpy_matrix(nx.erdos_renyi_graph(self.res, self.p, seed=self.s, directed=True))
                out("Erdos-Reyni Graph generated successfully", self.q, self.v)
            elif matrixType == 'ran':
                self.W = np.random.rand(self.res,self.res)
                out("Random Graph generated", self.q, self.v)
            else:
                sys.exit("Matrix type not recognized")
        
        # Check for threshold legality
        if iterationThr > 0:
            self.stopIt = iterationThr
        else:
            out("Iteration threshold is less than 1, defaulting to 100", self.q, self.v)
        
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
                
        self.perc = SGDClassifier(random_state=seed)
        out("ESN defined successfully!", self.q, self.v, "Completed in " + str(time.time()-st) + " seconds.")
        
###############################################################################
# fit -         A function to train output weights for esn. This will only fit 
#               the weights of the output transformation to the data, will not
#               print or return anything.
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
            t = self.run(np.reshape(x[i,:], x.shape[1]), iterations)

            if len(t.shape) > 1:
                t = np.reshape(t, max(t.shape))

            j = 0
            while j < len(t) and j < X.shape[1]:
                X[i,j] = t[j]
                j += 1
                
        # Train on stored values and save into coefs array
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
            # Important preprocessing
            out("Hopfield architecture initialized", self.q, self.v)
            st = time.time()

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
            
            '''
            This next green block is a fossil of old code. This was important 
            to make sure continuous data has converged. The project model did
            not need it, however. 
            '''

            # if convergence is not reached, send error
            # if not converged:
            #     sys.exit("Solution not converged in limit")
            # else:
            #     # Else return the related information
            #     g = self.converge(buffer, boo=False)
            #     m = int(max(g))

            out("Hopfield architecture complete", self.q, self.v, "Finished in " + str(time.time() - st) + " seconds")
            return buffer[-1:,:] 

        elif self.up == 'lat':
            # Preprocessing
            out("Lattice structure initialized", self.q, self.v)
            st = time.time()

            # Alignment checks
            if (self.os is not None):
                if self.shp != self.os.shape:
                    sys.exit('Lattice unaligned with omega array')
            else:
                self.os = self.o * np.ones(self.shp)
            
            if (self.ks is not None):
                if self.shp != self.ks.shape:
                    sys.exit('Lattice unaligned with K array')
            else:
                self.ks = self.k * np.ones(self.shp)

            # Propogating the information through the lattice
            for i in range(self.shp[1]):
                for j in range(self.shp[0]):
                    inpt[j] = scm(inpt[j], self.os[j,i], self.ks[j,i])
                out('', self.q, self.v,"Row " + str(i) + " propogated")
            
            out("Lattice completed", self.q, self.v, "Finished in " + str(time.time()-st) + ' seconds')
            return inpt

        elif self.up == 'tor':
            # Preprocessing
            out("Torus structure initiated", self.q, self.v)
            st = time.time()

            # Alignment checks
            if self.os is not None:
                if self.shp != self.os.shape:
                    sys.exit('Lattice unaligned with omega array')
            else:
                self.os = self.o * np.ones(self.shp)
            
            if self.ks is not None:
                if self.shp != self.ks.shape:
                    sys.exit('Lattice unaligned with K array')
            else:
                self.ks = self.k * np.ones(self.shp)

            # Predefine buffer for convergence checking
            buffer = np.ones((self.stopIt,len(inpt)))
            
            # Insert the inputs
            for i in range(self.shp[0]):
                buffer[0:i] = inpt[i]

            # Some variables to be used later
            converged = False
            i = 0
            j = 0

            # While under convergence limit and not yet converged
            while i < self.cl and not converged:
                out('', self.q, self.v, 'Iteration ' + str(i) + ' initiated')

                # Define a new vector to store values
                temp = inpt

                # Propogate the values in the vector to the next layer
                temp[j % self.shp[0]] = scm(inpt[j % self.shp[0]], self.os[j % self.shp[0],i % self.shp[1]], self.ks[j % self.shp[0],i % self.shp[1]])

                # Update the buffer
                for k in range(self.shp[0]):
                    t = sum(temp[k-2:k+1])
                    inpt[k] = t
                    buffer[(i+1)%self.stopIt,k] = t
                
                # When the buffer is filled, check for convergence
                if i > self.stopIt:
                    converged = self.converge(buffer)

                # Iterate Torus layers 
                if j % self.shp[0] == 0:
                    i += 1
            
            out("Calculations over", self.q, self.v, "Finished in " + str(time.time() - st) + " seconds")

            # if convergence is not reached, send error
            if not converged:
                sys.exit("Solution not converged in limit")
            else:
                # Else return the related information
                # g = self.converge(buffer, boo=False)
                # m = int(max(g))
                return buffer[-1:,:] #np.mean(buffer[-m:,:], axis=1)


###############################################################################
# train -   Train an output weight matrix from reservoir output and yactual. 
#           This will return the weight matrix that was generated from the
#           training process. This function is also a fossil as it is not as
#           efficient as many existing programs, however this is the way to 
#           perform Adam's optimizer. 
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
# converge -    Returns either boolean or integer values representing the 
#               frequency of the output. If the output converges to a constant,
#               it will return a frequency of 0, or oscillator death. Given any
#               other return value means that the results are oscillating at 
#               that frequency. This is done by generating an nxn upper 
#               triangular matrix, and filling each cell at (i,j) with the 
#               absolute difference between buffer[i] and buffer[j]. Then,
#               the diagonal closest to the main one that contains all zeros
#               (or values below the tolerance) will be the frequency of the
#               current values in the buffer. 
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
# objective -   A function that returns the error of the current weight matrix.
#               Currently either returns percent correct or MSE.
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
# score -       A function that scores the reservoir on test set. Just runs 
#               through all test values and checks for correct class 
#               prediction.
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