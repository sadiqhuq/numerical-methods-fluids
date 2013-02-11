#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

# # unsteady 1D convection equation
# #   d phi            d phi
# #  ------- =  -U0 * -------
# #    dt               dx
# # 
# # Solved by Explicit and Implicit Euler in Time
# # Central in Space

def explicit_conv_1d ( phi, dt, U0, npoints, phinew ):
    # # Periodic BC at x = 0
    phinew[0] = phi[-1];

    # # Loop over points in space
    # for j  in range (1, npoints):
    for j  in range (1, npoints):
        phinew[j] = phi[j] - ( U0*dt/2./dx ) * ( phi[j+1] - phi[j-1] )

    # # Periodic BC at x = 2 pi
    phinew[-1] =  phi[0] 

    return phinew

def implicit_conv_1d ( phi, A, b, dt, dx, U0, npoints, phinew ):
    
    a_w = -U0*dt/2./dx
    a_p = 1
    a_e = -a_w
    
    # #  Periodic BC at x=0
    A[0,-2]        = a_w
    A[0, 0]        = a_p
    A[0, 1]        = a_e
    
    b[0]           = phi[0]

    for j in range ( 1 , npoints - 1):
        A[j,j-1]   = a_w
        A[j,j]     = a_p
        A[j,j+1]   = a_e
        b[j]       = phi[j] 

    # #  Periodic BC at x=2*pi
    A[-1,1]        = a_e
    A[-1,-1]       = a_p
    A[-1,-2]       = a_w
    b[-1]          = phi [-1]

    # #  Solve the linear system of equations
    phinew,resid,rank,s = np.linalg.lstsq(A,b)   # # A\b

    return phinew

def conv_analytical_1d ( t, npoints, phi_a, case ):
    
    if ( case == 'sine' ):
     # # Analytical solution
       for j in range( npoints+1):
           # t = i * dt
           phi_a[j] = np.sin( x[j]-(U0*t) )
    
    return phi_a
      
npoints = 40
x       = np.linspace(0,2*np.pi,npoints+1)  # x E [0,2pi]
dx      = x[1]-x[0]
U0      = 1.
dt      = 0.001
tsteps  = 1000

# # Initial Condition
phi0     = np.sin(x)        # phi(x,t) = sin(x) at t=0
# phi0      = np.exp(-((x-np.pi)/0.01)**2) # impulse
# # Explicit Euler:
# # phinew is phi at time n+1

phi    = phi0.copy()
phinew = np.zeros(np.shape(phi))
phi_a  = np.zeros(np.shape(phi))

for i in range(0,tsteps+1):

  phinew = explicit_conv_1d ( phi, dt, U0, npoints, phinew )
  # # Write new phi to old phi
  phi = phinew
  
  phi_a = conv_analytical_1d (i*dt, npoints, phi_a, 'sine' )

phi_exp = phi.copy()

# # Implicit Euler:

# #  Initialize 
# #  and solution vector phi
A       = np.zeros((npoints+1,npoints+1))  # coefficient matrix A
b       = np.zeros(npoints+1)              # constant vector b

phi_imp = phi0.copy()
phi     = phi0.copy()

for i in range(0,tsteps+1):
    phi_imp =  implicit_conv_1d ( phi, A, b, dt, dx, U0, npoints, phi_imp )
    phi     = phi_imp


plt.plot(x, phi0,   'k-', label='t=0')
plt.plot(x, phi_a,  'm--', label='Analytical')
plt.plot(x, phi_exp,'r-', label='Explicit Euler')
plt.plot(x, phi_imp,'c-', label='Implicit Euler')

plt.legend(loc=4)

plt.show()