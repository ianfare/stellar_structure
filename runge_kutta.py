#!/usr/bin/python

import matplotlib.pyplot as plt
import math

###########################################################
###########################################################

# ~~~~~~~~ System of ODEs information goes here ~~~~~~~~~~~


# Define any number of functions
# i.e. the RHS of ODEs
def f1(x,y1,y2):
  return y2

def f2(x,y1,y2):
  b = 1.0
  c = 5.0
  return -b*y2 - c**2*y1

# Add them to the functions list
functions = [f1,f2]

# Define initial conditions, at the lower bound of x
A = 1.0
b = 1.0
c = 5.0
ics = [A,-A*b/2]

# Specify domain to run, and h
h = 0.001
x_lim = [0.0,7.0]

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
    k2_args.append(argList[0] + h/2)
    # Add all the y values + (k1,i)/2 to the function's arguments
    for i,element in enumerate(argList):
      if i > 0:
        k2_args.append(element + k1[i-1]/2)
    # Call the functions, and record k values for each
    k2.append(h * func(*k2_args))

  k3 = []
  # Calculate k_3,i, as above
  for func_i,func in enumerate(functions):
    k3_args = []
    k3_args.append(argList[0] + h/2)
    for i,element in enumerate(argList):
      if i > 0:
        k3_args.append(element + k2[i-1]/2)
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
    y_results.append(argList[func_i+1] + k1[func_i]/6 + k2[func_i]/3 + k3[func_i]/3 + k4[func_i]/6)
  
  # Return the newly calculated y values
  return y_results

analytic = []

# Main routine:
#   1) Prepares x values to evaluate
#   2) Prepares lists as elements of list for y_i values
#   3) Puts initial conditions into those lists
#   4) Runs find_ys() for each x value sequentially
#   5) Plots analytic solution (can remove this if there is none)
def main_routine(functions,ics,h,x_lim):
  # Prepare xs to evaluate
  xs = []
  num_x = int(math.floor((x_lim[1] - x_lim[0])/h))
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
    if x_i%1000000==0:
      print str(int(math.floor((x - x_lim[0])*100/(x_lim[1]-x_lim[0])))) + "%"
    if x_i > 0:
      new_args = [x]
      for i in ys:
        new_args.append(i[-1])
      new_ys = find_ys(new_args)
      for i,element in enumerate(ys):
        element.append(new_ys[i])

  # Compute analytic solution
  for i in xs:
    analytic.append(A*math.exp(-b*i/2)*math.cos(c*i))

  # Plot stuff
  plt.plot(xs,ys[0],label="Runge-Kutta solution")
  plt.plot(xs,analytic,label="Analytic solution")
  plt.xlabel("x")
  plt.ylabel("y_1")
  plt.title("Runge-Kutta and Analytic Solutions from x="+str(x_lim[0])+" to x="+str(x_lim[1])+"\nwith b=" +str(b)+",c="+str(c)+",h="+str(h))
  plt.legend()
  plt.show()

  return ys
  
main_routine(functions,ics,h,x_lim)
