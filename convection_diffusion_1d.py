#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

# # Steady 1D convection-diffusion equation
# # -U0 dphi/dx + Gamma d^2 phi/ dx^2 = 0
# # phi (x = 0) = 0; phi ( x = 2pi) = 0

# # Play with the values of U0 and Gamma 

# #  Set convection velocity
U0      = 3.0

# #  Set diffusivity
Gamma   = 5.0

def upwind_central (U0,dx,Gamma):

    a_w =  ( U0/dx) + (  Gamma/(dx*dx))
    a_p =  (-U0/dx) - (2*Gamma/(dx*dx))
    a_e =      0    + (  Gamma/(dx*dx))

    return (a_w, a_p, a_e)

def downwind_central (U0,dx,Gamma):
    a_w =      0    + (  Gamma/(dx*dx))
    a_p =  ( U0/dx) - (2*Gamma/(dx*dx))
    a_e =  (-U0/dx) + (  Gamma/(dx*dx))
    
    return (a_w, a_p, a_e)

def central_central (U0,dx,Gamma):
    a_w = (U0/(2*dx))  + (   Gamma /(dx*dx))
    a_p =      0       + (-2*Gamma)/(dx*dx)
    a_e = -(U0/(2*dx)) + (   Gamma /(dx*dx))
    
    return (a_w, a_p, a_e)

# #  Discrete spacing in space!
xend    = 2.0*np.pi
npoints = 50 + 1
dx      = xend/(npoints-1)

# #  x locations
x       = np.linspace (0.0, xend, npoints)

# #  Initialize
phi     = np.zeros(npoints)           # # vector field phi
b       = np.zeros(npoints)           # # vector b
A       = np.zeros((npoints,npoints)) # # matrix A upwind
B       = np.zeros((npoints,npoints)) # # matrix B downwind
C       = np.zeros((npoints,npoints)) # # matrix C central

# #  Boundary condition
phi_0   = 0.0
phi_end = 1.0

# #  Populate the matrix

for i in range (1 , npoints-1 ):    # boundary points excluded
    A[i,i-1] , A[i,i  ], A[i,i+1] = upwind_central   (U0,dx,Gamma)
    B[i,i-1] , B[i,i  ], B[i,i+1] = downwind_central (U0,dx,Gamma)
    C[i,i-1] , C[i,i  ], C[i,i+1] = central_central  (U0,dx,Gamma)

# #  Boundary conditions
b[ 0]        = 0
b[-1]        = 1

A[ 0, 0]     = 1
A[-1,-1]     = 1

B[ 0, 0]     = 1
B[-1,-1]     = 1

C[ 0, 0]     = 1
C[-1,-1]     = 1

# #  Solution of the linear system

phi_UC,resid,rank,s = np.linalg.lstsq(A,b) # phi = A\b
phi_DC,resid,rank,s = np.linalg.lstsq(B,b)
phi_CC,resid,rank,s = np.linalg.lstsq(C,b)

plt.plot(x,phi_UC, 'r', label='Upwind - Central')
plt.plot(x,phi_DC, 'b', label='Downwind - Central')
plt.plot(x,phi_CC, 'g', label='Central - Central')

plt.legend(loc=2)
plt.show()
