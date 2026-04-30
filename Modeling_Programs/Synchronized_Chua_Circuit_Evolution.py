'''
Synchronized_Chua_Circuit_Evolution.py

Computes the numerical solution to the system of differential equations that 
govern the behavior of two similar, capacitively coupled Chua circuits with user-supplied 
initial conditions using the SciPy routine odeint.

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
from pylab import plot,scatter,xlim,ylim,xlabel,ylabel,grid,title,show,figure,subplot,subplots_adjust
from scipy.integrate import odeint

# Specify the input parameters
## The numbering of the resistors is taken from Kennedy (1992).

## QUANTITIES THAT ARE THE SAME IN BOTH CIRCUITS
## According to Figure 12 in Liu et al (2019).
## Components of the conventional linear oscillation circuit
C_2 = 100.0*(10**(-9)) #F
R = 1.8*(10**3) #Ohms 
L = 18.0*(10**(-3)) #H
R_L = 0.0 #Ohms

## Resistors in the Chua diode
## first non-linear resistor
R_1 = 220.0 #Ohms
R_2 = R_1
R_3 = 2.4*(10**3) #Ohms
## second non-linear resistor
R_4 = 22.0*(10**3) #Ohms
R_5 = R_4
R_6 = 3.3*(10**3) #Ohms

# Dimensionless circuit parameters 
## The formulae for m_0 and m_1 are taken from Kennedy (1992) 
## with a factor of R included to make the values dimensionless.
## The formulae for beta and gamma are taken from Liu et al. (2019)

## first non-linear resistor
m_01 = 1.0/R_1
m_11 = -1.0/R_3
## second non-linear resistor
m_02 = 1.0/R_4
m_12 = -1.0/R_6
## full circuit
#m_0 = (m_11 + m_12)*R
#m_1 = (m_11 + m_02)*R
beta = (C_2*(R**2))/L
gamma = (R_L*R*C_2)/L

## QUANTITIES THAT DIFFER BETWEEN THE CIRCUITS
C_1 = 10.0*(10**(-9)) #F
alpha = C_2/C_1
C_1_prime = 12.5*(10**(-9)) #F
alpha_prime = C_2/C_1_prime

## Default values used in Liu et al. (2019)
#beta = 18.0
#gamma = 0.0
m_0 = -1.296
m_1 = -0.738
#alpha = 10.0
#alpha_prime = 8.0

## COMBINED SYSTEM
## coupling capacitance
C = 1.25*(10**(-9)) #F ## The values tested in Liu et al. are 0.0, 1.25, 10.0, and 2000.0 nF.
C_nF = C/(10**(-9)) #nF
C_scaled = C/((C_1*C_1_prime)+(C*C_1)+(C*C_1_prime)) #dimensionless

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
    x_prime = r[3]
    y_prime = r[4]
    z_prime = r[5]
    
    dxdt = alpha*(y - x - f_E(x)) + C_scaled*( alpha_prime*C_1_prime*(y_prime - x_prime - f_E(x_prime)) - alpha*C_1*(y - x - f_E(x)) )
    dydt = x - y + z
    dzdt = -1*beta*y - gamma*z
    
    dxdt_prime = alpha_prime*(y_prime - x_prime - f_E(x_prime)) - C_scaled*( alpha_prime*C_1_prime*(y_prime - x_prime - f_E(x_prime)) - alpha*C_1*(y - x - f_E(x)) )
    dydt_prime = x_prime - y_prime + z_prime
    dzdt_prime = -1*beta*y_prime - gamma*z_prime
    
    return array([dxdt, dydt, dzdt, dxdt_prime, dydt_prime, dzdt_prime], float)

# Specify initial conditions
x0 = 0.1 ## dimensionless
y0 = 0.1 ## dimensionless
z0 = 0.01 ## dimensionless
x0_prime = 0.1 ## dimensionless
y0_prime = 0.1 ## dimensionless
z0_prime = 0.2 ## dimensionless
r0 = array([x0,y0,z0,x0_prime,y0_prime,z0_prime],float)

# Calculate the numerical solution using
# fourth-order Runge-Kutta algorithm
t1 = 0.0 # initial scaled time
t2 = 900.0 # final scaled time
h = 0.01 # time step size

tpoints = arange(t1,t2,h) 
print(len(tpoints))

r = odeint(derivatives,r0,tpoints)

# print len(r[:,0]) # Just checking on the length of the output array

## Find error function (given in Liu et al. (2019)).
theta = sqrt( (r[:,0] - r[:,3])**2 + (r[:,1] - r[:,4])**2 + (r[:,2] - r[:,5])**2 )

## Plotting
## In the phase space plots, the first 1/10th of the arrays are not plotted
## because the plots look better when the initial transient is excluded.

# Plot the evolution of both
figure("Long-term Evolution", figsize=(10,4))
plot(tpoints,r[:,0],color='cyan')
xlim((7*t2/9),t2)
ylim(min(r[:,0]),max(r[:,0]))
xlabel("Scaled Time (t)",fontsize=16)
ylabel("Scaled Voltage Across $C_1$ (x)",fontsize=16)
title(r'$ \alpha =$ %.1f, $\beta =$ %.1f, $m_0 =$ %.1f, $m_1 = $ %.3f' '\n' r'C = %.2f nF'%(alpha,beta,m_0,m_1,C_nF),fontsize=14)
grid(True)
show()

figure("Long-term Evolution Prime", figsize=(10,4))
plot(tpoints,r[:,3],color='magenta')
xlim((7*t2/9),t2)
ylim(min(r[:,0]),max(r[:,0]))
xlabel("Scaled Time (t)",fontsize=16)
ylabel("Scaled Voltage Across $C_1$` (x`)",fontsize=16)
title(r'$ \alpha$` = %.1f, $\beta =$ %.1f, $m_0 =$ %.1f, $m_1 = $ %.3f' '\n' r'C = %.2f nF'%(alpha_prime,beta,m_0,m_1,C_nF),fontsize=14)
grid(True)
show()

# Plot the phase space portrait
## x vs y
figure("Phase Space Plot: x vs. y",figsize=(6,5))
scatter(r[int(len(tpoints)/10):len(tpoints),0],r[int(len(tpoints)/10):len(tpoints),1],color="cyan",s=0.01) 
xlabel("Scaled Voltage Across $C_1$ (x)",fontsize=16)
ylabel("Scaled Voltage Across $C_2$ (y)",fontsize=16)
grid(True)
title(r'$ \alpha =$ %.1f, $\beta =$ %.1f, $m_0 =$ %.3f, $m_1 = $ %.3f' '\n' r'C = %.2f nF'%(alpha,beta,m_0,m_1,C_nF),fontsize=14)
show()

## x' vs y'
figure("Phase Space Plot: x' vs. y'",figsize=(6,5))
scatter(r[int(len(tpoints)/10):len(tpoints),3],r[int(len(tpoints)/10):len(tpoints),4],color="magenta",s=0.01) 
xlabel("Scaled Voltage Across $C_1$` (x`)",fontsize=16)
ylabel("Scaled Voltage Across $C_2$` (y`)",fontsize=16)
grid(True)
title(r'$ \alpha$` = %.1f, $\beta =$ %.1f, $m_0 =$ %.3f, $m_1 = $ %.3f' '\n' r'C = %.2f nF'%(alpha_prime,beta,m_0,m_1,C_nF),fontsize=14)
show()

# Plot the error function
figure("Error Function",figsize=(6,5))
scatter(tpoints,theta,color ='blueviolet',s=0.01)
xlim(0,400)
ylim(0,12)
xlabel("Scaled Time (t)",fontsize=16)
ylabel(r"Error Function ($\theta$)",fontsize=16)
grid(True)
title(r'$ \alpha$ = %.1f, $\alpha$` = %.1f, $\beta =$ %.1f, $m_0 =$ %.3f, $m_1 = $ %.3f' '\n' r'C = %.2f nF'%(alpha,alpha_prime,beta,m_0,m_1,C_nF),fontsize=12)
show()


'''
# Output the data in columns
output_array_rows = vstack([tpoints,transpose(r[:,0]),transpose(r[:,1])])
savetxt("R_V_is_8.2.txt",output_array_rows)
output_array_columns = transpose(output_array_rows)
savetxt("R_V_is_8.2_columns.txt",output_array_columns)
'''