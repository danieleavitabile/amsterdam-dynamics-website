# Demonstration simulation for passive and active brownian particles
# (c) 2019 Silke Henkes, University of Bristol

import argparse
import numpy as np 
from copy import deepcopy
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sys import platform


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--phi",  type=float, default=0.5, help="packing fraction")
parser.add_argument("-v", "--v0",  type=float, default=0.1, help="active driving velocity")
parser.add_argument("-T", "--Temp",  type=float, default=0.00001, help="temperature")
parser.add_argument("-D", "--Dr",  type=float, default=0.01, help="rotational diffusion constant")
parser.add_argument("-J", "--Jval",  type=float, default=0.0, help="polar alignment strength")
parser.add_argument("-L", "--L",  type=float, default=40, help="system size (square)")
parser.add_argument("-p", "--poly",  type=float, default=0.2, help="polydispersity (standard deviation)")
args = parser.parse_args()


# there will be some hard-coded values in here
# particle mean radius
sigma = 1.0
# polydispersity
poly = args.poly
# potential stiffness
krep = 1.0
# time step (v0<1, otherwise it will all go through each other anyway)
dt = 0.1
dtinv = 1/dt
# Total simulation duration: unitl the window is closed
# Given the time plotting takes, run a number of steps between plotting
steps = 10
#circscale = int(25*40/args.L)
#circscale = 50000/args.L
# At this packing fraction, the number of particles is
N = int(args.L**2*args.phi/(np.pi*sigma**2))
print("We have " + str(N) + " particles in our system.")
# stochastic fluctuation amplitudes at this time step
Damp = np.sqrt(2*args.Dr*dt)
Tamp = np.sqrt(2*args.Temp*dt)

# helper functions
# periodic boundary conditons
def periodic(dl,bound):
    return dl - bound*np.round(dl/bound)

# Note: forego boxes and neighbour lists in favour of vectorial, should be faster in numpy
# initialise the system
xval = np.random.uniform(-args.L/2.0,args.L/2.0,N)
yval = np.random.uniform(-args.L/2.0,args.L/2.0,N)
phival = np.random.uniform(0,2*np.pi,N)
Rval = sigma*np.random.uniform(1-poly,1+poly,N)

# needs to be defined and finite to start with
vx = 0.01*xval
vy = 0.01*yval

# Creating the plotting environment
# Create new Figure and an Axes which fills it.
fig = plt.figure(figsize=(7, 7))
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
ax.set_xlim(-args.L/2.0, args.L/2.0), ax.set_xticks([])
ax.set_ylim(-args.L/2.0, args.L/2.0), ax.set_yticks([])

# Because operating systems are stupid
if platform == "linux" or platform == "linux2":
    # linux
    mult = 1.0
elif platform == "darwin":
    # OS X
    mult = 0.5
elif platform == "win32":
    # windows
    mult = 1.0
else:
    print("Unknonwn operating system!")
    mult=1.0
Rplot = mult*Rval*(ax.transData.transform([1,0])[0] - ax.transData.transform([0,0])[0])
splot=7*Rplot**2

# Set unit length of arrows in this system (once and for all)
# significantly trickier than it looks ...
scales=[]
if args.phi>0.4:
    scales.append(0.006*krep/args.phi)
else:
    scales.append(0.01)
scales.append(np.sqrt(args.Temp))
scales.append(0.5*args.v0)
scale = max(scales)
print(scales)
print(scale)
#flow = ax.quiver(xval,yval,args.v0*np.cos(phival),args.v0*np.sin(phival),scale_units='xy',scale=scale)

# Main loop (as simple as that)
start_time = time.time()

def mainloop(t):
    # because python is idiotic for arrays (but not for ints or doubles, or composite types, go figure)
    global phival
    global xval
    global yval
    global Rval
    global vx
    global vy
    #print time.time() - start_time
    # Potential part of the update
    forces = np.zeros((N,2))
    torques = np.zeros((N,))
    for u in range(steps):
        #if (t%100==0):
        #   print "step " + str(t) + ", simulation time elapsed"
        # keep the velocity values for plotting purposes
        xval0 = deepcopy(xval)
        yval0 = deepcopy(yval)
        # Deterministic repulsive force update (O(N**2) like this, but slower if not vectorial)
        for k in range(N):
            dx = periodic(xval-xval[k],args.L)
            dy = periodic(yval-yval[k],args.L)
            r2 = (Rval[k]+Rval)**2
            # now determine neighbours
            # throw out self
            dx[k] = 3*sigma
            dy[k] = 3*sigma
            neighbours = np.nonzero((r2 - (dx**2+dy**2) )>0.0)
            # f = - k delta vec r /r, delta = overlap
            dr = np.sqrt(dx[neighbours]**2+dy[neighbours]**2)
            delta = (Rval[k]+Rval[neighbours])-dr
            forcex = - krep * delta * dx[neighbours]/dr
            forcey = - krep * delta * dy[neighbours]/dr

            forces[k,0] = np.sum(forcex)
            forces[k,1] = np.sum(forcey)

            # polar alignment torques
            torques[k] = args.Jval*np.sum(np.sin(phival[neighbours]-phival[k]))

        # synchronously update the positions and angles
        for k in range(N):
            xval[k] += forces[k,0] * dt
            yval[k] += forces[k,1] * dt
            phival[k] += torques[k] * dt

        # stochastic part of the update (can be vectorised)
        # angular coordinate
        phival += Damp * np.random.normal(0,1,N) 
        # apply periodic boundary conditions
        phival = periodic(phival,2*np.pi)
        # hence update positions with self-propulsion
        xval += args.v0 * np.cos(phival) * dt
        yval += args.v0 * np.sin(phival) * dt

        # temperature (note different stochastic components in x and y)
        xval += Tamp * np.random.normal(0,1,N)
        yval += Tamp * np.random.normal(0,1,N) 
        # define velocities
        vx = (xval-xval0) * dtinv
        vy = (yval-yval0) * dtinv
        # periodic boundary conditions
        xval = periodic(xval,args.L)
        yval = periodic(yval,args.L)

    # plotting
    
    ax.clear()
    ax.set_xlim(-args.L/2.0, args.L/2.0)
    ax.set_ylim(-args.L/2.0, args.L/2.0)
    
    
    ax.quiver(xval,yval,np.cos(phival),np.sin(phival),color='y',lw=2, scale_units='xy', angles='xy', scale=1)
    ax.quiver(xval,yval,vx,vy,color='r',lw=1, width=0.01, scale_units='xy', angles='xy', scale=0.03)
    ax.scatter(xval,yval, s=splot, edgecolors='tab:blue', facecolors='none')


    return ax,

animation = FuncAnimation(fig, mainloop, interval=0.1, blit=True)

#writervideo = animation.FFMpegWriter(fps=10) 
#animation.save('ABP_corr.mp4', writer=writervideo)
#animation.save('ABP_corr.avi') 
#plt.close() 
plt.show()


