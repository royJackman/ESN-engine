import random

import numpy as np

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

weights = [-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

def graphGen(n=128, p=0.3, seed=1618, type="circular", verbose=False):
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Define weight matrix
    retval = np.zeros((n,n))

    if type in ['circular','circ']:
        # Loop through nodes
        for x in range(n):

            # Make it into a ring oscillator using the power of math
            retval[x,(x+1)%n] = np.random.randint(-15,16)

            # Loop through the rest of the nodes
            for y in range(n):

                # If it will make an odd loop...
                if abs(x - y)%2 == 0 and y != (x-1) and y != x:

                    # Give it some probability
                    t = np.random.rand()

                    # Add edge if the random number was within range
                    if t <= p:
                        retval[x,y] = np.random.randint(-15,16)
    
    elif type in ['osci','oscillatory']:
        # Make a degree 2 ring oscillator first
        for x in range(n):
            retval[x,(x+1)%n] = random.choice(weights)
            retval[(x+2)%n,x] = random.choice(weights)

        # Build the fractal pattern within the network by filling a loops array
        # then adding the edges, emptying it, and repeating with a loop half
        # the size of the last 
        o = n
        loops = []

        # At size 4 and below, nothing else is needed because it was taken care
        # of during the degree 2 ring oscillator
        while o > 4:
            if verbose: print(o)
            
            # If the loop is odd...
            if o % 2 == 1:
                # Get the 'half'
                p = (o-1)/2
                # If the loops array is empty...
                if len(loops) == 0:
                    # Must be the first one, so add two loops and that's it
                    loops.append((0,p+1))
                    loops.append((p,0))
                # If the loops array is NOT empty...
                else:
                    # Copy the existing loops and reset the loops array
                    temp = loops
                    loops = []
                    # Iterate through the existing loops and cut them in two
                    for t in temp:
                        loops.append(((t[1]+p),t[1]))
                        loops.append((t[0],(t[1]+p)))
            # If the loop is even...
            else:
                # Do things that are similar
                p = (o-2)/2
                if len(loops) == 0:
                    loops.append((p,0))
                    loops.append((-1,p+1))
                else:
                    temp = loops
                    loops = []
                    for t in temp:
                        loops.append(((t[1]+p),t[1]))
                        loops.append((t[0],(t[1]+p+1)))
            o = p+1
            if verbose: print(loops)
            for l in loops:
                retval[int(l[0]),int(l[1])] = random.choice(weights)
    elif type == 'oscirc':
        for x in range(n):
            retval[x,(x+1)%n] = random.choice(weights)
            retval[(x+2)%n,x] = random.choice(weights)
    # Return the weight matrix
    return retval