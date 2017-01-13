#!/usr/bin/python

import matplotlib.pyplot as plt
import math

def f1(x,y1,y2):
  return y2

def f2(x,y1,y2):
  b = 0.5
  c = 1.0
  return -b*y2 - c*c*y1

functions = [f1,f2]

# Initial conditions, at x_0
A = 1.0
b = 0.5
c = 1.0
ics = [A,-A*b/2]

# Values of x to evaluate
h = 0.000001
x_lim = [0.0,20.0]

# Skipping the first subroutine to find derivatives
# because it's just one line, calling the functions...

# Second subroutine to call derivatives, calculate ks and tys
# Arguments should be:
#   0  : x
#   1+ : y_i
def find_ys(argList):

  y_results = []
  for func_i,func in enumerate(functions):
    k1_args = []
    # x value is unchanged
    k1_args.append(argList[0])
    for i,element in enumerate(argList):
      if i > 0:
        k1_args.append(element)
    k1 = h * func(*k1_args)

    k2_args = []
    k2_args.append(argList[0] + h/2)
    for i,element in enumerate(argList):
      if i > 0:
        k2_args.append(element + k1/2)
    k2 = h * func(*k2_args)

    k3_args = []
    k3_args.append(argList[0] + h/2)
    for i,element in enumerate(argList):
      if i > 0:
        k3_args.append(element + k2/2)
    k3 = h * func(*k2_args)

    k4_args = []
    k4_args.append(argList[0] + h)
    for i,element in enumerate(argList):
      if i > 0:
        k4_args.append(element + k3)
    k4 = h * func(*k4_args)

    y_results.append(argList[func_i+1] + k1/6 + k2/3 + k3/3 + k4/6)
  
  return y_results

analytic = []

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

  # Plot analytic solution
  for i in xs:
    analytic.append(A*math.exp(-b*i/2)*math.cos(c*i))

  plt.plot(xs,ys[0],label="Runge-Kutta solution")
  plt.plot(xs,analytic,label="Analytic solution")
  plt.xlabel("x")
  plt.ylabel("y_1")
  plt.title("Runge-Kutta and Analytic Solutions from x="+str(x_lim[0])+" to x="+str(x_lim[1])+" with h="+str(h))
  plt.legend()
  plt.show()

  return ys
  
main_routine(functions,ics,h,x_lim)
