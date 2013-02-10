#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

def explicit_conv_1d ( phi, dt, U0, npoints, phinew ):
    # # Periodic BC at x = 0
    phinew[0] = phi[-1];

    # # Loop over points in space
    # for j  in range (1, npoints):
    for j  in range (0, npoints):
        phinew[j] = phi[j] - ( U0*dt/2./dx ) * ( phi[j+1] - phi[j-1] )

    # # Periodic BC at x = 2 pi
    phinew[-1] =  phi[0] 

    return phinew

def conv_analytical_1d ( t, npoints, phi_a ):
    # # Analytical solution
    for j in range( npoints+1):
        # t = i * dt
        phi_a[j] = np.sin(x[j]-(U0*t))
    return phi_a
      
npoints = 40
x       = np.linspace(0,2*np.pi,npoints+1)  # x E [0,2pi]
dx      = x[1]-x[0]
U0      = 1.
dt      = 0.001
tsteps  = 1000

# # Initial Condition
phi     = np.sin(x)        # phi(x,t) = sin(x) at t=0

# # Explicit Euler:
# # phinew is phi at time n+1

phinew = np.zeros(np.shape(phi))

for i in range(tsteps+1):

  phinew = explicit_conv_1d ( phi, dt, U0, npoints, phinew )

  # # Write new phi to old phi
  phi = phinew
  
  phi_a = conv_analytical_1d (i*dt, npoints, phi_a )


  # Wave Transport Anmiation
  # plt.plot(x, phi, 'r-', x, phi_a, 'k-')
  # plt.show()

plt.plot(x, np.sin(x), 'k.',label='t=0')
plt.plot(x, phi_a, 'k-', label='Analytical')
plt.plot(x, phi,   'r-', label='Explicit Euler')

plt.legend()

plt.show()