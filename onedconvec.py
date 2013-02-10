#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

def conv_1d ( phi, dt, U0, npoints, phinew ):
    # # Periodic BC at x = 0
    # phinew[0] = phi[0];

    # # Loop over points in space
    for j  in range (1, npoints):
        phinew[j] = phi[j]-(U0*dt/2./dx)*(phi[j+1]-phi[j-1])

    # # Periodic BC at x = 2 pi
    # phinew[-1] =  phi[-1] 
    return phinew

def conv_analytical_1d ( t, npoints, phi_a ):
    # # Analytical solution
    for j in range( npoints):
        # t = i * dt
        phi_a[j] = np.sin(x[j]-(U0*t))
    return phi_a
      
npoints = 40
x       = np.linspace(0,2*np.pi,npoints+1)  # x E [0,2pi]
dx      = x[1]-x[0]
U0      = 0.
dt      = 0.1
tsteps  = 1000

# # Initial Condition
phi     = np.sin(x)        # phi(x,t) = sin(x) at t=0

# # Explicit Euler:
# # phinew is phi at time n+1

phinew = np.zeros(np.shape(phi))

for i in range(tsteps+1):

  phinew = conv_1d ( phi, dt, U0, npoints, phinew ) # save memory,if phi reused

  # # Write new phi to old phi
  phi = phinew
  

  phi_a = conv_analytical_1d (i*dt, npoints, phi_a )


  # Wave Transport Anmiation
  plt.plot(x, phi, 'r', x, phi_a, 'g');
  # hold off;
  # pause(0.03);

# end

# plt.plot (x,phi0,c='k')
# plt.plot (x,phi, 'r-')


plt.show()
# dphi_np = 