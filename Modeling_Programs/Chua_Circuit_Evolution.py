'''
Chua_Circuit_Evolution.py

Computes the numerical solution to the system of differential equations that 
govern the behavior of a Chua circuit with user-supplied initial conditions
using the SciPy routine odeint.

Differential equations of motion & formulae for dimensionless parameters taken from
    
Liu et al. "Synchronization control between two Chua's circuits via capacitive coupling"
    Applied Mathematics and Computing 360 (2019) pp. 96-104.

Kennedy, "Robust Op-Amp realization of Chua's circuit" Frequenz vol 40 no. 3-4 
    (March-April 1992) pp. 66-80.

Adapted from "Duffing-Holmes_E_evolution_v1.py," written by Dr. Ernest Behringer 
(Dept. of Physics and Astronomy, Eastern Michigan University),
by the PHY420 class of Winter 2026.

'''

from numpy import array,arange,sin,vstack,transpose,savetxt,pi,sqrt,abs
from pylab import plot,scatter,xlim,ylim,xlabel,ylabel,grid,title,show,figure
from scipy.integrate import odeint

# Specify the input parameters
## The numbering of the resistors is taken from Kennedy (1992).

## Components of the conventional linear oscillation circuit
C_1 = 10.0*(10**(-9)) #F
C_2 = 100.0*(10**(-9)) #F
R = 1.8*(10**3) #Ohms 
L = 18.0*(10**(-3)) #H
R_L = 0.0 #Ohms ## resistance associated with the inductor

## Resistors in the Chua diode
## first non-linear resistor
R_1 = 220.0 #Ohms
R_2 = R_1
R_3 = 2.2*(10**3) #Ohms
## second non-linear resistor
R_4 = 22.0*(10**3) #Ohms
R_5 = R_4
R_6 = 3.3*(10**3) #Ohms

# Dimensionless circuit parameters 
## The formulae for m_0 and m_1 are taken from Kennedy (1992) 
## with a factor of R included to make the values dimensionless.

## Formulae for m_0 and m
## first non-linear resistor
m_01 = 1.0/R_1
m_11 = -1.0/R_3
## second non-linear resistor
m_02 = 1.0/R_4
m_12 = -1.0/R_6
## full circuit
m_0 = (m_11 + m_12)*R
m_1 = (m_11 + m_02)*R

## The formulae for alpha, beta, and gamma are taken from Liu et al. (2019)
alpha = C_2/C_1
beta = (C_2*(R**2))/L
gamma = (R_L*R*C_2)/L

#The following are the default dimensionless parameter values used in Liu et al. (2019) 
#m_1 = -0.738
#m_0 = -1.296
#alpha = 10.0
#beta = 18.0
#gamma = 0.0


## The non-linear function f(x) and the differential equations
## As shown in Liu et al. (2019).
## Other literature may use different sign conventions.

def f_E(x):
    f_x = (m_1*x) + 0.5*(m_0 - m_1)*(abs(x+1)-abs(x-1))
    return f_x

def derivatives(r,t):
    x = r[0]
    y = r[1]
    z = r[2]
    dxdt = alpha*(y - x) - alpha*f_E(x)
    dydt = x - y + z
    dzdt = -1*beta*y - gamma*z
    return array([dxdt, dydt, dzdt], float)

# Specify initial conditions
x0 = 0.1 ## dimensionless
y0 = 0.1 ## dimensionless
z0 = 0.01 ## dimensionless
r0 = array([x0,y0,z0],float)

# Calculate the numerical solution using
# fourth-order Runge-Kutta algorithm
t1 = 0.0 # initial scaled time
t2 = 900.0 # final scaled time
h = 0.01 # time step size

tpoints = arange(t1,t2,h) 
print(len(tpoints))

r = odeint(derivatives,r0,tpoints)

# print len(r[:,0]) # Just checking on the length of the output array

# Plot the evolution
figure("Long-term Evolution", figsize=(10,4))
plot(tpoints,r[:,0],"cyan")
xlim((7*t2/9),t2)
ylim(min(r[:,0]),max(r[:,0]))
xlabel("Scaled Time (t)",fontsize=16)
ylabel("Scaled Voltage Across $C_1$ (x)",fontsize=16)
grid(True)
title(r'$ \alpha =$ %.2f, $ \beta =$ %.2f, $\gamma = $ %.2f, $m_0 =$ %.3f, $m_1 = $ %.3f'%(alpha,beta,gamma,m_0,m_1),fontsize=12)
show()

# Plot the phase space trajectory
figure("Phase Space Plot")
scatter(r[int(len(tpoints)/10):len(tpoints),0],r[int(len(tpoints)/10):len(tpoints),1],s=0.05,color="cyan")
xlabel("Scaled Voltage Across $C_1$ (x)",fontsize=16)
ylabel("Scaled Voltage Across $C_2$ (y)",fontsize=16)
grid(True)
title(r'$ \alpha =$ %.2f, $ \beta =$ %.2f, $\gamma = $ %.2f, $m_0 =$ %.3f, $m_1 = $ %.3f'%(alpha,beta,gamma,m_0,m_1),fontsize=12)
show()


'''
# Output the data in columns
output_array_rows = vstack([tpoints,transpose(r[:,0]),transpose(r[:,1])])
savetxt("R_V_is_8.2.txt",output_array_rows)
output_array_columns = transpose(output_array_rows)
savetxt("R_V_is_8.2_columns.txt",output_array_columns)
'''