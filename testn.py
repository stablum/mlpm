# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# Signal generators
def sawtooth(x, period=0.2, amp=1.0, phase=0.):
  return (((x / period - phase - 0.5) % 1) - 0.5) * 2 * amp
def sine_wave(x, period=0.2, amp=1.0, phase=0.):
  return np.sin((x / period - phase) * 2 * np.pi) * amp
def square_wave(x, period=0.2, amp=1.0, phase=0.):
  return ((np.floor(2 * x / period - 2 * phase - 1) % 2 == 0).astype(float) - 0.5)
def triangle_wave(x, period=0.2, amp=1.0, phase=0.):
  return (sawtooth(x, period, 1., phase) * square_wave(x, period, 1., phase) + 0.5)

def random_nonsingular_matrix(d=2):
  epsilon = 0.1
  A = np.random.rand(d, d)
  while abs(np.linalg.det(A)) < epsilon:
    A = np.random.rand(d, d)
  return A
  
def plot_signals(X):
  """
  Plot the signals contained in the rows of X.
  """
  figure()
  for i in range(X.shape[0]):
    ax = plt.subplot(X.shape[0], 1, i + 1)
    plot(X[i, :])
    ax.set_xticks([])
    ax.set_yticks([])

# <codecell>

sawtooth(5)

# <codecell>

# Generate data
num_sources = 4
signal_length = 500
t = linspace(0, 1, signal_length)
# np.c_ Translates slice objects to concatenation along the second axis.
S = np.c_[sawtooth(t), sine_wave(t, 0.3), square_wave(t, 0.4), triangle_wave(t, 0.25)]

# <codecell>

print num_sources

# <codecell>

print S

# <codecell>

print t

# <codecell>

plot_signals(S)

# <codecell>

plot(S[1:])

# <codecell>

plot(S[:1])

# <codecell>


