#!/usr/bin/python

# Introductory assignment: Runge-Kutta
# Ian Fare
# Jan. 15, 2017

import matplotlib.pyplot as plt
import math
import sys

###########################################################
###########################################################

# ~~~~~~~~ System of ODEs information goes here ~~~~~~~~~~~


# Define any number of functions
# i.e. the RHS of ODEs


# WORKING (?) pc: 0.7120808


#pc = 0.712
#pc = 0.7120754429
pc = float(sys.argv[1])
#pc = 0.712081097

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
#  return -p**2.0*f/(t**(2.5)*x)

# Add them to the functions list
functions = [f1,f2,f3,f4]

# Define initial conditions, at the lower bound of x
ics = [q_0,p_0,f_0,t_0]

# Specify domain to run, and h
# h = 0.0001: pc = 0.712080798960425
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

#  for i in uvint_data[0]:
#    print i

# Main routine:
#   1) Prepares x values to evaluate
#   2) Prepares lists as elements of list for y_i values
#   3) Puts initial conditions into those lists
#   4) Runs find_ys() for each x value sequentially
#   5) Plots analytic solution (can remove this if there is none)
def main_routine(functions,ics,h,x_lim):
  # Prepare xs to evaluate
  xs = []
  Us = []
  Vs = []
  np1 = []
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
 #   if x_i%1000000==0:
 #     print str(int(math.floor((x - x_lim[0])*100/(x_lim[1]-x_lim[0])))) + "%"
    if x_i > 0:
      new_args = [x]
      for i in ys:
        new_args.append(i[-1])
      new_ys = find_ys(new_args)
      for i,element in enumerate(ys):
        element.append(new_ys[i])
      Us.append(ys[1][x_i]*xs[x_i]**3.0/(ys[3][x_i]*ys[0][x_i]))
      Vs.append(ys[0][x_i]/(ys[3][x_i]*xs[x_i]))
      np1.append((ys[3][x_i]**(8.5))*ys[0][x_i]/(((ys[1][x_i]**2.0))*ys[2][x_i]))
      if np1[-1] <= 2.5:
        print "found 2.5 (^-^)"
        print Us[-1]
        print Vs[-1]
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
#  plt.title("Runge-Kutta and Analytic Solutions from x="+str(x_lim[0])+" to x="+str(x_lim[1])+"\nwith b=" +str(b)+",c="+str(c)+",h="+str(h))
#  plt.legend()
        plt.show()
#  plt.close()
#  plt.plot(xs,ys[0])
#  plt.xlabel("x")
#  plt.ylabel("q")
#  plt.show()
#  plt.close()
#  plt.plot(xs,ys[1])
#  plt.xlabel("x")
#  plt.ylabel("p")
#  plt.show()
#  plt.close()
#  plt.plot(xs,ys[2])
#  plt.xlabel("x")
#  plt.ylabel("f")
#  plt.show()
#  plt.close()
#        plt.plot(xs,ys[3])
#        plt.xlabel("x")
#        plt.ylabel("t")
#        plt.show()

        return "found 2.5 (^-^)"

  # Get U,V values
  # Remember, order is q,p,f,t
#  Us = []
#  Vs = []
#  np1 = []
#  for point in range(len(ys[0])):
#    Us.append(ys[1][point]*xs[point]**3.0/(ys[3][point]*ys[0][point]))
#    Vs.append(ys[0][point]/(ys[3][point]*xs[point]))
#    np1.append((ys[3][point]**(8.5))*ys[0][point]/(((ys[1][point]**2.0))*ys[2][point]))
#    np1.append(ys[3][point]**(8.5))
  #  print np1[-1]
  #  print ys[3][point]
  #  print ys[0][point]
  #  print ys[1][point]
  #  print ys[2][point]
  #  print ""

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
#  plt.title("Runge-Kutta and Analytic Solutions from x="+str(x_lim[0])+" to x="+str(x_lim[1])+"\nwith b=" +str(b)+",c="+str(c)+",h="+str(h))
#  plt.legend()
#  plt.show()
#  plt.close()
#  plt.plot(xs,ys[0])
#  plt.xlabel("x")
#  plt.ylabel("q")
#  plt.show()
#  plt.close()
#  plt.plot(xs,ys[1])
#  plt.xlabel("x")
#  plt.ylabel("p")
#  plt.show()
#  plt.close()
#  plt.plot(xs,ys[2])
#  plt.xlabel("x")
#  plt.ylabel("f")
#  plt.show()
#  plt.close()
  plt.plot(xs,ys[3])
  plt.xlabel("x")
  plt.ylabel("t")
#  plt.show()

  return ys
  
main_routine(functions,ics,h,x_lim)
