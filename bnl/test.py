import cudaq
from cudaq import spin
import cupy as cp
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np

# FIXME 1: set these parameters
theta = 0.0
n_site  = 4
a = 1.0
M = 1.0
g = 1.0
N = n_site
m = 1.0

t_max = int(0.5*n_site)
step_max = 100*t_max
dt = t_max/step_max

# FIXME 2: setup the Hamiltonian
# you can set spin_H = Hxx_yy + Hzz + Hz after defining each of these 
hamiltonian = cudaq.SpinOperator()
Hxx_yy = cudaq.SpinOperator()
Hzz = cudaq.SpinOperator()
Hz = cudaq.SpinOperator()


# H_XX+YY term
for n in range(N-1):
    coef = 0.25/a - m*0.25*((-1)**n)*np.sin(theta)
    Hxx_yy += coef*(spin.x(n)*spin.x(n+1) + spin.y(n)*spin.y(n+1))

for n in range(N):
    coef = m*((-1)**n)* np.cos(theta)*0.2
    Hzz += coef* spin.z(n)

for n in range(N):
    Hz += coef* spin.z(n)

Hz *= a*g*g*0.5

# FIXME 3: Define the other terms  Hzz + Hz

hamiltonian = Hxx_yy
print(type(hamiltonian))

# Convert the Hamiltonian to a matrix 
mat = hamiltonian.to_matrix()
# Convert to a cupy array
mat = cp.array(mat)

# Compute the eigenvalues and eigenvectors using cupy
eigenvalues, eigenvectors = cp.linalg.eigh(mat)

# Find the index of the smallest eigenvalue
index_of_smallest = cp.argmin(eigenvalues)

# Get the eigenvector corresponding to the smallest eigenvalue
psi0 = eigenvectors[:, index_of_smallest]
# Convert it to numpy error for qiskit state prep
psi0 = cp.asnumpy(psi0)

# prepare the initial state
qr = QuantumRegister(n_site, 'q')
cr = ClassicalRegister(n_site, 'c')
qc = QuantumCircuit(qr,cr)
qc.initialize(psi0)

# create cudaq circuit from qiskit
kernel = cudaq.make_kernel(qc.qasm())
#counts = cudaq.sample(kernel)

# Create without .qasm()
#kernel = cudaq.make_kernel(qc)
#counts = cudaq.sample(kernel)



