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
# out -         A printing function for important messages. Riveting, I know.
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