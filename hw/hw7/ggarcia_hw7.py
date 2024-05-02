'''
Gilberto Garcia
ASTR5900
Homework #7
24 April 2024
'''

#import required libraries
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 4]


#we update the code to match the conditions required for the hw
def wave_equation(nx, nt, c,plot=True):
    # Set spatial parameters
    xmin = -3.0
    xmax = 3.0
    dx = (xmax - xmin) / nx
    print(f"{xmin} <= x <= {xmax} (Spatial domain)")
    print(f"dx = {dx} (Spatial step size)")

    x_points = np.linspace(xmin, xmax, nx + 1)

    # Set temporal parameters
    tmin = 0.
    tmax = 3.
    dt = (tmax - tmin) / nt
    print(f"{tmin} <= x <= {tmax} (Temporal domain)")
    print(f"dt = {dt} (Temporal step size)")

    t_points = np.linspace(tmin, tmax, nt + 1)

    # Set speed for linear wave equation
    print(f"c = {c} (Wave speed)")

    # Calculate Courant Number
    C = c * dt / dx
    print(f"C = {C} (Courant Number)")

    # Define numerical solver step functions
    def upstream_step(u):
        # u_old = u.copy()
        # for x_index in range(1, nx + 1):
        #   u[x] = u_old[x] - C * (u_old[x] - o_old[x - 1])
        u[1:] = u[1:] - C * (u[1:] - u[:-1])
        
    def upstream_periodic_step(u):
        up = u.copy()
        u[1:] = up[1:] - C * (up[1:] - up[:-1])
        # Periodic boundary condition at x = xmin
        u[0] = up[0] - C * (up[0] - up[-2])
        # Periodic boundary condition at x = xmax
        u[-1] = u[0]

    def lax_step(u):
        u[1:-1] = 0.5 * (u[2:] + u[:-2]) - 0.5 * C * (u[2:] - u[:-2])

    def lax_periodic_step(u):
        up = u.copy()
        u[1:-1] = 0.5 * (up[2:] + up[:-2]) - 0.5 * C * (up[2:] - up[:-2])
        # Periodic boundary condition at x = xmin
        u[0] = 0.5 * (up[1] + up[-2]) - 0.5 * C * (up[1] - up[-2])
        # Periodic boundary condition at x = xmax
        u[-1] = u[0]

    def lax_wendroff_step(u):
        u[1:-1] = u[1:-1] - 0.5 * C * (u[2:] - u[:-2]) + 0.5 * C ** 2 * (u[2:] - 2 * u[1:-1] + u[:-2])

    def lax_wendroff_periodic_step(u):
        # Periodic boundary condition by wrapping at both ends
        up = np.concatenate(( [u[-2]], u, [u[1]] ))
        u[:] = up[1:-1] - 0.5 * C * (up[2:] - up[:-2]) + 0.5 * C ** 2 * (up[2:] - 2 * up[1:-1] + up[:-2])
        return 0

    def triangle(x,t=0):
        if -1.0 + c * t <= x <= 1.0 + c * t:
            return 1.0 - abs(x - c*t)
        else:
            return 0

    def periodic_triangle(x,t):
        x_wrapped = x
        while  x_wrapped < xmin + c * t: ############ FIX MEEEEEEEE :( #######
            x_wrapped += (xmax - xmin)
        while x_wrapped > xmax + c * t:
            x_wrapped -= (xmax - xmin)
        return triangle(x_wrapped, t)

    # Save the initial and final analytical solutions
    u0 = np.array([periodic_triangle(x, tmin) for x in x_points])
    uf = np.array([periodic_triangle(x, tmax) for x in x_points])

  # Initialize numerical solutions
    u_upstream = u0.copy()
    u_lax = u0.copy()
    u_lax_wendroff = u0.copy()

    # Advance numerical solution by nt timesteps
    for t in t_points[1:]:
        upstream_periodic_step(u_upstream)
        lax_periodic_step(u_lax)
        lax_wendroff_periodic_step(u_lax_wendroff)

    if plot == True:
        # Plot analytical and numerical solutions
        plt.plot(x_points, u0, 'b:', label="Initial conditions")
        plt.plot(x_points, uf, 'b-', label="Analytical solution")
        plt.plot(x_points, u_upstream, 'r-', label="Upstream (1st order)")
        plt.plot(x_points, u_lax, 'g-', label="Lax (1st order)")
        plt.plot(x_points, u_lax_wendroff, 'c-', label="Lax-Wendroff (2nd order)")
        plt.ylim(-0.2, 1.2)
        plt.legend()
        plt.show()
    
    #calculate relative error:
    abs_err = (u_upstream - uf)
    #return x points, analytical, and 1st order upstream numerical soln
    return x_points,abs_err,C

#1a - centering the propogation waves
wave_equation(nx = 80, nt = 100, c = 2)
#1b - we play around with nx and nt to find where Cmax is not stable
wave_equation(nx = 100, nt = 100, c = 2)
#we find that c_max of 1 is the upper limit for stability.

#1c - c_max for less than, equal to and greater than Cmax
#less than
wave_equation(nx = 50, nt = 100, c = 2)
#equal to 
wave_equation(nx = 100, nt = 100, c = 2)
#greater than
wave_equation(nx = 150, nt = 100, c = 2)

#2a
print()
print('2a')
xpts1,abserr1,C1 = wave_equation(nx = 100, nt = 110, c = 2,plot=False)
xpts2,abserr2,C2 = wave_equation(nx = 100, nt = 120, c = 2,plot=False)
xpts3,abserr3,C3 = wave_equation(nx = 100, nt = 130, c = 2,plot=False)
xpts4,abserr4,C4 = wave_equation(nx = 100, nt = 100, c = 2,plot=False)
print(xpts1)
plt.plot(xpts1,abserr1,label='C={0:.2f}'.format(C1))
plt.plot(xpts2,abserr2,label='C={0:.2f}'.format(C2))
plt.plot(xpts3,abserr3,label='C={0:.2f}'.format(C3))
plt.plot(xpts4,abserr4,label='C={0:.2f}'.format(C4))
plt.legend(loc='lower right')
plt.show()

#no, reducing delta t while keeping delta x constant does not improve accuracy.
# bigger nt means smaller delta t but also bigger error so less accuracy.