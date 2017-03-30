#!/usr/bin/python

# Structure assignment: Structure model & UV integrations
# Ian Fare
# March 30, 2017

# Use: ./structure.py
# Shows one plot at a time; close to move on to the next one
# After the plots with mass fraction on the x axis are closed,
#  prints R,L,Teff,Pc,Tc for the different masses
# Wait for plots to come up, don't click through them before they can load
#  (that causes an error)



import matplotlib.pyplot as plt
from math import pi,floor,log10

###########################################################
###########################################################

# ~~~~~~~~ System of ODEs information goes here ~~~~~~~~~~~


# Define any number of functions
# i.e. the RHS of ODEs

pc = 0.71208080006

x_0 = 0.01
q_0 = 1.0/3.0*(pc*x_0**3.0)
p_0 = pc - 1.0/6.0*((pc**2.0)*(x_0**2.0))
f_0 = 1.0/3.0*(pc**2.0*x_0**3.0)
t_0 = 1.0 - 1.0/6.0*(pc**4.0*x_0**4.0)

# x first, then q, then p, then f, then t
# order of arguments is important

def f1(x,q,p,f,t):
  return p*x**2.0/t

def f2(x,q,p,f,t):
  return -1*p*q/(t*(x**2.0))

def f3(x,q,p,f,t):
  return (p**2.0)*(t**2.0)*(x**2.0)

def f4(x,q,p,f,t):
  return -1*(p**2.0)*f/((t**(8.5))*(x**2.0))

# Add them to the functions list
functions = [f1,f2,f3,f4]

# Define initial conditions, at the lower bound of x
ics = [q_0,p_0,f_0,t_0]

# Specify domain to run, and h
h = 0.0001
x_lim = [x_0,20.0]

###########################################################
###########################################################


# ~~~~~~~~~~~~~~~ Runge-Kutta starts here ~~~~~~~~~~~~~~~~~

# Here, the 0th index in the functions list, the k lists,
# and the y list refers to information relating to y_0
# (or y_1 if you're starting with y_1), and so forth

# Skipping the first subroutine to find derivatives
# because it's just one line, calling the functions...

# Second subroutine to call derivatives, calculate ks and ys
# Arguments should be:
#   0  : x
#   1+ : y_i
def find_ys(argList):

  y_results = []

  k1 = []
  # Calculate k_1,i
  # For each function (from the DEs),
  for func_i,func in enumerate(functions):
    k1_args = []
    # Add the previous x value to the function's arguments
    k1_args.append(argList[0])
    # Add all the y values to the function's arguments
    for i,element in enumerate(argList):
      if i > 0:
        k1_args.append(element)
    # Call the functions, and record k values for each
    k1.append(h * func(*k1_args))

  k2 = []
  # Calculate k_2,i
  # For each function (from the DEs),
  for func_i,func in enumerate(functions):
    k2_args = []
    # Add the previous x value + h/2 to the function's arguments
    k2_args.append(argList[0] + h/2.0)
    # Add all the y values + (k1,i)/2 to the function's arguments
    for i,element in enumerate(argList):
      if i > 0:
        k2_args.append(element + k1[i-1]/2.0)
    # Call the functions, and record k values for each
    k2.append(h * func(*k2_args))

  k3 = []
  # Calculate k_3,i, as above
  for func_i,func in enumerate(functions):
    k3_args = []
    k3_args.append(argList[0] + h/2.0)
    for i,element in enumerate(argList):
      if i > 0:
        k3_args.append(element + k2[i-1]/2.0)
    k3.append(h * func(*k2_args))

  k4 = []
  # Calculate k_4,i
  for func_i,func in enumerate(functions):
    k4_args = []
    k4_args.append(argList[0] + h)
    for i,element in enumerate(argList):
      if i > 0:
        k4_args.append(element + k3[i-1])
    k4.append(h * func(*k4_args))

  # Calculate the new y_i values 
  for func_i,func in enumerate(functions):
    y_results.append(argList[func_i+1] + k1[func_i]/6.0 + k2[func_i]/3.0 + k3[func_i]/3.0 + k4[func_i]/6.0)
  
  # Return the newly calculated y values
  return y_results

def parse_int():
  # Read UV integration data
  uvint_data = []
  with open("uvintegrations.dat","r") as f:
    uvint_file = f.read()
  for char in uvint_file:
    if char == "T":
      uvint_data.append([])
    elif len(uvint_data) > 0:
      if char == "\n":
        uvint_data[-1].append([])
      elif len(uvint_data[-1]) > 0:
        uvint_data[-1][-1].append(char)

  # Parse UV integration data into:
  # uvint_data[i][j]
  #   i = E value
  #   j = point = [U,V]
  for i in uvint_data:
    for j in range(len(i)):
      i[j]="".join(i[j]).split(" ")
      for k in range(10):
        if "" in i[j]:
          i[j].remove("")
      for k in range(len(i[j])):
        i[j][k] = float(i[j][k])
    for k in range(10):
      if [] in i:
        i.remove([])
  
  return uvint_data


def plotstuff(xs,Us,Vs,ys):
  # Called when n+1<=2.5
  # Does UV plot, and makes plots/answers questions from assignment



  # Plot Runge Kutta model solution
  plt.plot(Us,Vs,label="Runge-Kutta solution")

  # Plot uv integrations
  uvint_data = parse_int()

  # Get E values corresponding to tracks for labelling
  track_numbers = [1.0,1.5,2.0,3.0,4.0,5.0,6.0,7.0,8.0]
  # Plot UV integrations
  for track_i,track in enumerate(uvint_data):
    track_Us = []
    track_Vs = []
    for point in track:
      track_Us.append(point[0])
      track_Vs.append(point[1])
    plt.plot(track_Us,track_Vs,label="E = " + str(track_numbers[track_i]))

  plt.legend()
  plt.xlabel("U")
  plt.ylabel("V")
  plt.show()
  plt.close()


  # Plot density, temperature, pressure as a function of mass fraction (m/M)
  # Rename them from Runge-Kutta solution for simplicity
  qs = ys[0]
  ps = ys[1]
  fs = ys[2]
  ts = ys[3]

  # These are values from the UV integration (E=7.0) where my model fits
  # I pulled them from the pdf and did y=10^log(y) calculations by hand, sorry
  xstar_match = 0.8222426499
  qstar_match = 0.9912198715
  pstar_match = 0.0152805856
  fstar_match = 1.0
  tstar_match = 0.0862223787

  # These are the x,q,p,f,t values where n+1=2.5
  # I just printed them off from when I found the working pc value
  # And copied them here
  # Sorry for hard-coding
  xmatch = 10.0116 # = xs[-1]
  qmatch = 15.8904896167 # = ys[0][-1]
  pmatch = 0.000154476422363 # = ys[1][-1]
  fmatch = 1.30646023557 # = ys[2][-1]
  tmatch = 0.105298613559 # = ys[3][-1]

  # Compute core values
  x0 = xstar_match/xmatch
  q0 = qstar_match/qmatch
  p0 = pstar_match/pmatch
  f0 = fstar_match/fmatch
  t0 = tstar_match/tmatch

  # Solar composition
  X = 0.7
  Z = 0.02
  Y = 1 - X - Z

  # These are all cgs
  sig = 5.6704e-5 # Stefan-Boltzmann constant
  K0 = 4.35e25*Z*(1.0+X) # Core Kramer's opacity
  E0 = 10.0**(-29.0)*X**2 # Core energy from p-p chain
  a = 4.0*(5.67e-5)/2.998e10 # Radiation constant
  c = 2.998e10 # Speed of light
  R = 8.31e7 # Solar radius
  mu = 1.0/(2.0*X+0.75*Y+0.5*Z) # Equation of state
  G = 6.674e-8 # Gravitational constant
  M = 1.98855e33 # Solar mass
  Lsun = 3.839e33 # Solar luminosity

  # Calculate constants C & D
  C = t0**(9.5)*x0/(p0**2*f0)
  D = f0/(p0**2*t0**2*x0**3)

  # Calculate solar radius
  Radius = (1.0/(C*D)*E0*(R/mu)**3.5*(4.0*pi)**(-4.0)*G**(-3.5)*M**(0.5)*3.0*K0/(4.0*a*c))**(1.0/6.5)


  # Calculate r,m,l,T,P for each x to convert model to physical units
  xstar = []
  qstar = []
  pstar = []
  fstar = []
  tstar = []
  r = []
  m = []
  l = []
  T = []
  P = []
  mass_fraction = []
  density = []

  # Calculate x*,q*,p*,f*,t*,r,m,l,T,P,mass_fraction,density for each x 
  for point in range(len(ps)):
    xstar.append(x0*xs[point])
    qstar.append(q0*qs[point])
    pstar.append(p0*ps[point])
    fstar.append(f0*fs[point])
    tstar.append(t0*ts[point])
    r.append(xstar[-1]*Radius)
    m.append(qstar[-1]*M)
    l.append(fstar[-1]*Lsun)
    T.append(tstar[-1]*mu*G*M/(R*Radius))
    P.append(pstar[-1]*G*M**2/(4*pi*Radius**4))
    mass_fraction.append(m[-1]/M)
    density.append(P[-1]*mu/(R*T[-1]))
  # Plot density as a function of mass fraction for model
  plt.plot(mass_fraction,density)
  plt.xlabel("Mass fraction m/M")
  plt.ylabel("Density (g/cm^3)")
  plt.show()
  plt.close()
  # Plot temperature as a function of mass fraction for model
  plt.plot(mass_fraction,T)
  plt.xlabel("Mass fraction m/M")
  plt.ylabel("Temperature (K)")
  plt.show()
  plt.close()
  # Plot pressure as a function of mass fraction for model
  plt.plot(mass_fraction,P)
  plt.xlabel("Mass fraction m/M")
  plt.ylabel("Pressure (baryes)")
  plt.show()
  plt.close()


  # Calculate L,Teff,Tc,Pc,R for different masses
  masses = [0.7*M,0.8*M,0.9*M,1.0*M,2.0*M,3.0*M,4.0*M,5.0*M]
  radii_cm = []
  radii_rsun = []
  luminosities = []
  luminosities_lsun = []
  Tcs = []
  Pcs = []
  Teffs = []
  for mass in masses:
    # Calculate radius in cm and in solar radii
    radii_cm.append((1.0/(C*D)*E0*(R/mu)**3.5*(4.0*pi)**(-4.0)*G**(-3.5)*mass**(0.5)*3.0*K0/(4.0*a*c))**(1.0/6.5))
    radii_rsun.append(radii_cm[-1]/Radius)
    # Calculate luminosity from definition of constant D
    luminosities.append(E0*(mu/R)**4*G**4/(4*pi)*mass**6/(D*radii_cm[-1]**7))
    # Or, calculate luminosity from deminition of constant C
#    luminosities.append(C*4*a*c/(3*K0)*(4*pi)**3*(mu/R)**7.5*G**7.5*M**(5.5)/radii_cm[-1]**(0.5))
    luminosities_lsun.append(luminosities[-1]/Lsun)
    # Calculate effective temperature
    Teffs.append((luminosities[-1]/(4*pi*radii_cm[-1]**2*sig))**(0.25))
    # Calculate core temperature
    Tcs.append(tstar[0]*mu*G*mass/(R*radii_cm[-1]))
    # Calculate core pressure
    Pcs.append(pstar[0]*G*mass**2/(4*pi*radii_cm[-1]**4))

    
  # Print it all out for each mass
  for i in range(len(masses)):
    print str(masses[i]/M) + " Msun: R = " + str(radii_rsun[i]) + " Rsun" 
    print "          L = " + str(luminosities_lsun[i]) + " Lsun"
    print "       Teff = " + str(Teffs[i]) + " K"
    print "         Tc = " + str(Tcs[i]) + " baryes"
    print "         Pc = " + str(Pcs[i]) + " K"
    print ""

  # Do HR diagram
  # Get observed main sequence from HRDiagram.dat
  # And parse it into log(L/Lsun) and log(T)
  obs_log_Ts = []
  obs_log_Lratios = []
  with open("./HRDiagram.dat","r") as f:
    obs = f.readlines()
  for i in range(len(obs)):
    obs[i] = obs[i].strip().split(" ")
    if i > 1:
      for j in range(len(obs[i])):
        if j > 1 and len(obs[i][j]) > 1:
          if j < 9:
            obs_log_Ts.append(float(obs[i][j]))
          else:
            obs_log_Lratios.append(float(obs[i][j]))

  # Make lists of log(L/Lsun) and log(T), from values calculated above
  log_Lratios = []
  log_Ts = []
  for i in luminosities:
    log_Lratios.append(log10(i/Lsun))
  for i in Teffs:
    log_Ts.append(log10(i))
  
  # Plot stuff
  plt.plot(log_Ts,log_Lratios,"ro",label="model")
  plt.plot(obs_log_Ts,obs_log_Lratios,"bo",label="observational")
  plt.xlabel("log(Teff (K))")
  plt.ylabel("log(L/Lsun)")
  plt.legend()
  plt.gca().invert_xaxis()
  plt.show()
  plt.close()

# Main routine:
#   1) Prepares x values to evaluate
#   2) Prepares lists as elements of list for y_i values
#   3) Puts initial conditions into those lists
#   4) Runs find_ys() for each x value sequentially
#      Calculate U,V,n+1 values
#      If n+1<=2.5, plot/calculate stuff for assignment and stop
def main_routine(functions,ics,h,x_lim):
  # Prepare xs to evaluate
  xs = []
  Us = []
  Vs = []
  np1 = []
  num_x = int(floor((x_lim[1] - x_lim[0])/h))
  for i in range(num_x):
    xs.append(x_lim[0] + i * h)

  # Prepare arrays to put y values in
  ys = []
  for i in range(len(functions)):
    ys.append([])

  # Put in ics as first element of y lists
  for i,element in enumerate(ys):
    element.append(ics[i])

  # Run find_ys()
  for x_i,x in enumerate(xs):
    if x_i > 0:
      new_args = [x]
      for i in ys:
        new_args.append(i[-1])
      new_ys = find_ys(new_args)
      for i,element in enumerate(ys):
        element.append(new_ys[i])
      # Calculate U,V,n+1 values
      Us.append(ys[1][x_i]*xs[x_i]**3.0/(ys[3][x_i]*ys[0][x_i]))
      Vs.append(ys[0][x_i]/(ys[3][x_i]*xs[x_i]))
      np1.append((ys[3][x_i]**(8.5))*ys[0][x_i]/(((ys[1][x_i]**2.0))*ys[2][x_i]))
      # If at any point n+1<=2.5, plot /calculate stuff for assignment and stop
      if np1[-1] <= 2.5:
        plotstuff(xs,Us,Vs,ys)
        return "found 2.5 (^-^)"

  # If main_func() gets here, it hasn't found n+1=2.5 before x=20, which is bad
  # Plot anyways to see what's going on, I guess

  # Plot Runge Kutta
  plt.plot(Us,Vs,label="Runge-Kutta solution")

  # Plot uv integrations
  uvint_data = parse_int()
  for track in uvint_data:
    track_Us = []
    track_Vs = []
    for point in track:
      track_Us.append(point[0])
      track_Vs.append(point[1])
    plt.plot(track_Us,track_Vs,label="UV integrations")

  plt.xlabel("U")
  plt.ylabel("V")
  plt.show()

  return ys
  
main_routine(functions,ics,h,x_lim)
