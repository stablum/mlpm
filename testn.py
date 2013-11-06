# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab inline

# Signal generators
def sawtooth(x, period=0.2, amp=1.0, phase=0.):
    return (((x / period - phase - 0.5) % 1) - 0.5) * 2 * amp

def sine_wave(x, period=0.2, amp=1.0, phase=0.):
    return np.sin((x / period - phase) * 2 * np.pi) * amp

def square_wave(x, period=0.2, amp=1.0, phase=0.):
    return ((np.floor(2 * x / period - 2 * phase - 1) % 2 == 0).astype(float) - 0.5) * 2 * amp

def triangle_wave(x, period=0.2, amp=1.0, phase=0.):
    return (sawtooth(x, period, 1., phase) * square_wave(x, period, 1., phase) + 0.5) * 2 * amp

def random_nonsingular_matrix(d=2):
    """
    Generates a random nonsingular (invertible) matrix of shape d*d
    """
    epsilon = 0.1
    A = np.random.rand(d, d)
    while abs(np.linalg.det(A)) < epsilon:
        A = np.random.rand(d, d)
    return A

def plot_signals(X, showAxes=False):
    """
    Plot the signals contained in the rows of X.
    """
    figure()
    for i in range(X.shape[0]):
        # see http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot
        # subplot params: numrows, numcols, fignum where fignum ranges from 1 to numrows*numcols.
        ax = plt.subplot(X.shape[0], 1, i + 1)
        plot(X[i, :])
        if not showAxes:
            ax.set_xticks([])
            ax.set_yticks([])

# <codecell>

sine_wave(np.array([1,2,3]))

# <codecell>

num_sources = 5
signal_length = 500
t = linspace(0, 1, signal_length)
# np.c_ Translates slice objects to concatenation along the second axis.
S = np.c_[sawtooth(t), sine_wave(t, 0.3), square_wave(t, 0.4), triangle_wave(t, 0.25), np.random.randn(t.size)].T

# <codecell>

print num_sources

# <codecell>

t2 = linspace(0,2,10)
s1 = sawtooth(t2)
s2 = sine_wave(t2, 0.3)
S2 = np.array([s1, s2])
print S2
print "" 
print np.c_[s1,s2].T

# <codecell>

print S

# <codecell>

plot_signals(S2,showAxes=True)

# <codecell>

plot_signals(S,showAxes=True)

# <codecell>

print random_nonsingular_matrix()

# <codecell>

B =np.array([[1,2,3,4],[5,3,6,7]])
print B
print B.shape
print B.T
B = B.T
A = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]).T
print "A = %s " % A
print "B = %s" % B

print "A x B: %s" % np.dot(A,B)

# <codecell>

def make_mixtures(S, A):
    X = np.dot(A,S)
    return X




Mnr = Snr = S.shape[0]
Tnr = S.shape[1]

A = random_nonsingular_matrix(Snr)

X = make_mixtures(S,A)

plot_signals( X)

# <codecell>

# prints a histogram for each row of X
def plot_histograms(X, nrBins=10,showAxes=False):
    nrRows = X.shape[0]
    nrColumns = X.shape[1]
    if nrBins > nrColumns:
        raise Exception("nr bins > nr columns of matrix X")
    figure()
    for i in range(nrRows):
        #print X[i,:]
        hist, bin_edges = histogram(X[i,:],nrBins)
        #print hist
        #print bin_edges
        lowest = bin_edges[:1][0]
        highest =  bin_edges[-1:][0]
        interval = (highest-lowest)/float(nrBins)
        #print interval
        ax = plt.subplot(nrRows, 1, i + 1)
        bar(bin_edges[:-1],hist, width=interval)
        if not showAxes:
            ax.set_xticks([])
            ax.set_yticks([])
            

plot_histograms(X,nrBins=50,showAxes=False)

# <codecell>

act0 = lambda x : -tanh(x)
sourceDistr0 = lambda x : math.exp ( - math.log( cosh(x)))

act1 = lambda x : -x + tanh(x)
sourceDistr1 = lambda x : math.exp(  -( math.pow(x,2) / float(2) ) + math.log(cosh(x)) )

act2 = lambda x : (-1)*(math.pow(x,3))
sourceDistr2 = lambda x : math.exp( -((math.pow(x,4))/float(4)))

act3 = lambda x : -((6*x)/((math.pow(x,2))+5))
sourceDistr3 = lambda x : float(1)/(math.pow(math.pow(x,2)+5,3))

actFunctions = [act0, act1, act2, act3]
sourceDistrFunctions = [sourceDistr0, sourceDistr1, sourceDistr2, sourceDistr3]

actT = linspace(-4,4,500)
#actM = [x for x in     

actMl = []
sourceDistrMl = []
for i in range (0, len(actFunctions)):
    actMl.append(map (actFunctions[i], actT))
    sourceDistrMl.append(map (sourceDistrFunctions[i], actT))


actM = np.array(actMl)
sourceDistrM = np.array(sourceDistrMl)

def plot_signals2(X1,X2, showAxes=False):
    """
    Plot the signals contained in the rows of X.
    """
    figure()
    for i in range(X1.shape[0]):
        # see http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot
        # subplot params: numrows, numcols, fignum where fignum ranges from 1 to numrows*numcols.
        ax = plt.subplot(X.shape[0], 1, i + 1)
        plot(X1[i, :])
        plot(X2[i,:])
        if not showAxes:
            ax.set_xticks([])
            ax.set_yticks([])
            
plot_signals2(actM, sourceDistrM, showAxes=True)

plot_signals(actM)

plot_signals(sourceDistrM)

# <codecell>

l = []
l.append([1,2,3]) 
l.append([3,3,6])
print l

la = np.array(l)
print la

# <codecell>


