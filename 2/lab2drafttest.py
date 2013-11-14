# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Lab 2: Inference in Graphical Models

# <markdowncell>

# ## Part 1: The sum-product algorithm

# <markdowncell>

# Given code

# <codecell>

%pylab inline
class Node(object):
    """
    Base-class for Nodes in a factor graph. Only instantiate sub-classes of Node.
    """
    def __init__(self, name):
        # A name for this Node, for printing purposes
        self.name = name
        
        # Neighbours in the graph, identified with their index in this list.
        # i.e. self.neighbours contains neighbour 0 through len(self.neighbours) - 1.
        self.neighbours = []
        
        # Reset the node-state (not the graph topology)
        self.reset()
        
    def reset(self):
        # Incomming messages; a dictionary mapping neighbours to messages.
        # That is, it maps  Node -> np.ndarray.
        self.in_msgs = {}
        
        # A set of neighbours for which this node has pending messages.
        # We use a python set object so we don't have to worry about duplicates.
        self.pending = set([])

    def add_neighbour(self, nb):
        self.neighbours.append(nb)

    def send_sp_msg(self, other):
        # To be implemented in subclass.
        raise Exception('Method send_sp_msg not implemented in base-class Node')
   
    def send_ms_msg(self, other):
        # To be implemented in subclass.
        raise Exception('Method send_ms_msg not implemented in base-class Node')
    
    def receive_msg(self, other, msg):
        # Store the incomming message, replacing previous messages from the same node
        self.in_msgs[other] = msg

        # TODO: add pending messages
        # self.pending.update(...)
    
    def __str__(self):
        # This is printed when using 'print node_instance'
        return self.name


class Variable(Node):
    def __init__(self, name, num_states):
        """
        Variable node constructor.
        Args:
            name: a name string for this node. Used for printing. 
            num_states: the number of states this variable can take.
            Allowable states run from 0 through (num_states - 1).
            For example, for a binary variable num_states=2,
            and the allowable states are 0, 1.
        """
        self.num_states = num_states
        
        # Call the base-class constructor
        super(Variable, self).__init__(name)
    
    def set_observed(self, observed_state):
        """
        Set this variable to an observed state.
        Args:
            observed_state: an integer value in [0, self.num_states - 1].
        """
        # Observed state is represented as a 1-of-N variable
        # Could be 0.0 for sum-product, but log(0.0) = -inf so a tiny value is preferable for max-sum
        self.observed_state[:] = 0.000001
        self.observed_state[observed_state] = 1.0
        
    def set_latent(self):
        """
        Erase an observed state for this variable and consider it latent again.
        """
        # No state is preferred, so set all entries of observed_state to 1.0
        # Using this representation we need not differentiate observed an latent
        # variables when sending messages.
        self.observed_state[:] = 1.0
        
    def reset(self):
        super(Variable, self).reset()
        self.observed_state = np.ones(self.num_states)
        
    def marginal(self, Z=None):
        """
        Compute the marginal distribution of this Variable.
        It is assumed that message passing has completed when this function is called.
        Args:
            Z: an optional normalization constant can be passed in. If None is passed, Z is computed.
        Returns: Z. Either equal to the input Z, or computed (if Z=None was passed).
        """
        # TODO: compute marginal
        return None, None
    
    def send_sp_msg(self, other):
        # TODO: implement Variable -> Factor message for sum-product
        pass
   
    def send_ms_msg(self, other):
        # TODO: implement Variable -> Factor message for max-sum
        pass

class Factor(Node):
    def __init__(self, name, f, neighbours):
        """
        Factor node constructor.
        Args:
            name: a name string for this node. Used for printing
            f: a numpy.ndarray with N axes, where N is the number of neighbours.
               That is, the axes of f correspond to variables, and the index along that axes corresponds to a value of that variable.
               Each axis of the array should have as many entries as the corresponding neighbour variable has states.
            neighbours: a list of neighbouring Variables. Bi-directional connections are created.
        """
        # Call the base-class constructor
        super(Factor, self).__init__(name)

        assert len(neighbours) == f.ndim, 'Factor function f should accept as many arguments as this Factor node has neighbours'
        
        for nb_ind in range(len(neighbours)):
            nb = neighbours[nb_ind]
            assert f.shape[nb_ind] == nb.num_states, 'The range of the factor function f is invalid for input %i %s' % (nb_ind, nb.name)
            self.add_neighbour(nb)
            nb.add_neighbour(self)

        self.f = f
        
    def send_sp_msg(self, other):
        # TODO: implement Factor -> Variable message for sum-product
        pass
   
    def send_ms_msg(self, other):
        # TODO: implement Factor -> Variable message for max-sum
        pass

    def __str__(self):
        return "name:\n\t%s \nneighbours:\n\t%s\n probs:\n\t%s" % (self.name, self.neighbours, self.f)

# <markdowncell>

# ### 1.1 Instantiate network (10 points)
# Convert the directed graphical model ("Bayesian Network") shown below to a factor graph. Instantiate this graph by creating Variable and Factor instances and linking them according to the graph structure. 
# To instantiate the factor graph, first create the Variable nodes and then create Factor nodes, passing a list of neighbour Variables to each Factor.
# Use the following prior and conditional probabilities.
# 
# $$
# p(\verb+Influenza+) = 0.05 \\\\
# p(\verb+Smokes+) = 0.2 \\\\
# $$
# 
# $$
# p(\verb+SoreThroat+ = 1 | \verb+Influenza+ = 1) = 0.3 \\\\
# p(\verb+SoreThroat+ = 1 | \verb+Influenza+ = 0) = 0.001 \\\\
# p(\verb+Fever+ = 1| \verb+Influenza+ = 1) = 0.9 \\\\
# p(\verb+Fever+ = 1| \verb+Influenza+ = 0) 0.05 \\\\
# p(\verb+Bronchitis+ = 1 | \verb+Influenza+ = 1, \verb+Smokes+ = 1) = 0.99 \\\\
# p(\verb+Bronchitis+ = 1 | \verb+Influenza+ = 1, \verb+Smokes+ = 0) = 0.9 \\\\
# p(\verb+Bronchitis+ = 1 | \verb+Influenza+ = 0, \verb+Smokes+ = 1) = 0.7 \\\\
# p(\verb+Bronchitis+ = 1 | \verb+Influenza+ = 0, \verb+Smokes+ = 0) = 0.0001 \\\\
# p(\verb+Coughing+ = 1| \verb+Bronchitis+ = 1) = 0.8 \\\\
# p(\verb+Coughing+ = 1| \verb+Bronchitis+ = 0) = 0.07 \\\\
# p(\verb+Wheezing+ = 1| \verb+Bronchitis+ = 1) = 0.6 \\\\
# p(\verb+Wheezing+ = 1| \verb+Bronchitis+ = 0) = 0.001 \\\\
# $$

# <codecell>

from IPython.core.display import Image 
Image(filename='/run/media/root/ss-ntfs/3.Documents/huiswerk_20132014/MLPR/lab/git/2/bn.png') 

# <codecell>

nrStates = 2

I = Variable("Influenza",nrStates)
f1 = Factor("f1",asanyarray([0.05,0.95],dtype=float),[I])

print f1

print np.ndarray(shape=(1,nrStates),dtype=float)

print np.ndarray(shape=(1,2),dtype=float,offset=np.int_().itemsize,buffer=np.array([0,1,2]))

bla = asanyarray([[[1,2],[3,4]],[[5,6],[7,8]]],dtype=float)

print type(bla)

print bla
print "==="

print bla[0,0,1]

# <markdowncell>

# ### 1.2 Factor to variable messages (20 points)
# Write a method `send_sp_msg(self, other)` for the Factor class, that checks if all the information required to pass a message to Variable `other` is present, computes the message and sends it to `other`. "Sending" here simply means calling the `receive_msg` function of the receiving node (we will implement this later). The message itself should be represented as a numpy array (np.array) whose length is equal to the number of states of the variable.
# 
# An elegant and efficient solution can be obtained using the n-way outer product of vectors. This product takes n vectors $\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}$ and computes a $n$-dimensional tensor (ndarray) whose element $i_0,i_1,...,i_n$ is given by $\prod_j \mathbf{x}^{(j)}_{i_j}$. In python, this is realized as `np.multiply.reduce(np.ix_(*vectors))` for a python list `vectors` of 1D numpy arrays. Try to figure out how this statement works -- it contains some useful functional programming techniques. Another function that you may find useful in computing the message is `np.tensordot`.

# <markdowncell>

# ### 1.3 Variable to factor messages (10 points)
# 
# Write a method `send_sp_message(self, other)` for the Variable class, that checks if all the information required to pass a message to Variable var is present, computes the message and sends it to factor.

# <markdowncell>

# ### 1.4 Compute marginal (10 points)
# Later in this assignment, we will implement message passing schemes to do inference. Once the message passing has completed, we will want to compute local marginals for each variable.
# Write the method `marginal` for the Variable class, that computes a marginal distribution over that node.

# <markdowncell>

# ### 1.5 Receiving messages (10 points)
# In order to implement the loopy and non-loopy message passing algorithms, we need some way to determine which nodes are ready to send messages to which neighbours. To do this in a way that works for both loopy and non-loopy algorithms, we make use of the concept of "pending messages", which is explained in Bishop (8.4.7): 
# "we will say that a (variable or factor)
# node a has a message pending on its link to a node b if node a has received any
# message on any of its other links since the last time it send (sic) a message to b. Thus,
# when a node receives a message on one of its links, this creates pending messages
# on all of its other links."
# 
# Keep in mind that for the non-loopy algorithm, nodes may not have received any messages on some or all of their links. Therefore, before we say node a has a pending message for node b, we must check that node a has received all messages needed to compute the message that is to be sent to b.
# 
# Modify the function `receive_msg`, so that it updates the self.pending variable as described above. The member self.pending is a set that is to be filled with Nodes to which self has pending messages. Modify the `send_msg` functions to remove pending messages as they are sent.

# <markdowncell>

# ### 1.6 Inference Engine (10 points)
# Write a function `sum_product(node_list)` that runs the sum-product message passing algorithm on a tree-structured factor graph with given nodes. The input parameter `node_list` is a list of all Node instances in the graph, which is assumed to be ordered correctly. That is, the list starts with a leaf node, which can always send a message. Subsequent nodes in `node_list` should be capable of sending a message when the pending messages of preceding nodes in the list have been sent. The sum-product algorithm then proceeds by passing over the list from beginning to end, sending all pending messages at the nodes it encounters. Then, in reverse order, the algorithm traverses the list again and again sends all pending messages at each node as it is encountered. For this to work, you must initialize pending messages for all the leaf nodes, e.g. `influenza_prior.pending.add(influenza)`, where `influenza_prior` is a Factor node corresponding the the prior, `influenza` is a Variable node and the only connection of `influenza_prior` goes to `influenza`.
# 
# 

# <markdowncell>

# ### 1.6 Observed variables and probabilistic queries (15 points)
# We will now use the inference engine to answer probabilistic queries. That is, we will set certain variables to observed values, and obtain the marginals over latent variables. We have already provided functions `set_observed` and `set_latent` that manage a member of Variable called `observed_state`. Modify the `Variable.send_msg` and `Variable.marginal` routines that you wrote before, to use `observed_state` so as to get the required marginals when some nodes are observed.

# <markdowncell>

# ### 1.7 Sum-product and MAP states (5 points)
# A maximum a posteriori state (MAP-state) is an assignment of all latent variables that maximizes the probability of latent variables given observed variables:
# $$
# \mathbf{x}_{\verb+MAP+} = \arg\max _{\mathbf{x}} p(\mathbf{x} | \mathbf{y})
# $$
# Could we use the sum-product algorithm to obtain a MAP state? If yes, how? If no, why not?

# <markdowncell>

# ## Part 2: The max-sum algorithm
# Next, we implement the max-sum algorithm as described in section 8.4.5 of Bishop.

# <markdowncell>

# ### 2.1 Factor to variable messages (10 points)
# Implement the function `Factor.send_ms_msg` that sends Factor -> Variable messages for the max-sum algorithm. It is analogous to the `Factor.send_sp_msg` function you implemented before.

# <markdowncell>

# ### 2.2 Variable to factor messages (10 points)
# Implement the `Variable.send_ms_msg` function that sends Variable -> Factor messages for the max-sum algorithm.

# <markdowncell>

# ### 2.3 Find a MAP state (10 points)
# 
# Using the same message passing schedule we used for sum-product, implement the max-sum algorithm. For simplicity, we will ignore issues relating to non-unique maxima. So there is no need to implement backtracking; the MAP state is obtained by a per-node maximization (eq. 8.98 in Bishop). Make sure your algorithm works with both latent and observed variables.

# <markdowncell>

# ## Part 3: Image Denoising and Loopy BP
# 
# Next, we will use a loopy version of max-sum to perform denoising on a binary image. The model itself is discussed in Bishop 8.3.3, but we will use loopy max-sum instead of Iterative Conditional Modes as Bishop does.
# 
# The following code creates some toy data. `im` is a quite large binary image, `test_im` is a smaller synthetic binary image. Noisy versions are also provided.

# <codecell>

from pylab import imread, gray
# Load the image and binarize
im = np.mean(imread('dalmatian1.png'), axis=2) > 0.5
imshow(im)
gray()

# Add some noise
noise = np.random.rand(*im.shape) > 0.9
noise_im = np.logical_xor(noise, im)
figure()
imshow(noise_im)

test_im = np.zeros((10,10))
#test_im[5:8, 3:8] = 1.0
#test_im[5,5] = 1.0
figure()
imshow(test_im)

# Add some noise
noise = np.random.rand(*test_im.shape) > 0.9
noise_test_im = np.logical_xor(noise, test_im)
figure()
imshow(noise_test_im)

# <markdowncell>

# ### 3.1 Construct factor graph (10 points)
# Convert the Markov Random Field (Bishop, fig. 8.31) to a factor graph and instantiate it.

# <markdowncell>

# ### 3.2 Loopy max-sum (10 points)
# Implement the loopy max-sum algorithm, by passing messages from randomly chosen nodes iteratively until no more pending messages are created or a maximum number of iterations is reached. 
# 
# Think of a good way to initialize the messages in the graph.

