#!/usr/bin/python

def f1(x,y1,y2):
  return y2

def f2(x,y1,y2):
  b = 1.0
  c = 2.0
  return -b*y2 - c**2.0*y1

functions = [f1,f2]

# Initial conditions, at x_0
A = 1.0
ics = [A,0.0]

# Values of x to evaluate
h = 0.1
x_0 = 0.0

# First subroutine to evaluate the derivatives
def derivatives(*args):
  return f(*args)

# Second subroutine to call derivatives, calculate ks and tys
def find_ys(*argList):

  k1_args = []
  # x value is unchanged
  k1_args.append(argList[0])
  for i,element in enumerate(argList):
    if i > 0:
      k1_args.append(element)
  k1 = h * derivatives(*k1_args)

  k2_args = []
  k2_args.append(argList[0] + h/2)
  for i,element in enumerate(argList):
    if i > 0:
      k2_args.append(element + k1/2)
  k2 = h * derivatives(*k2_args)

  k3_args = []
  k3_args.append(argList[0] + h/2)
  for i,element in enumerate(argList):
    if i > 0:
      k3_args.append(element + k2/2)
  k3 = h * derivatives(*k2_args)

  k4_args = []
  k4_args.append(argList[0] + h)
  for i,element in enumerate(argList):
    if i > 0:
      k4_args.append(element + k3)

#  k1 = h * derivatives(xval,y1[-1],y2[-1])
#  k2 = h * derivatives(xval + h/2, y1[-1] + k1/2, y2[-1] + k1/2)
#  k3 = h * derivatives(xval + h/2, y1[-1] + k2/2, y2[-1] + k2/2)
#  k4 = h * derivatives(xval + h, y1[-1] + k3, y2[-1] + k3)

  y_results = []
  for i,element in enumerate(argList):
    if i > 0:
      y_results.append(element + k1/6 + k2/3 + k3/3 + k4/6)
  return y_results

def main_routine(functions,ics,h,x_0,num_x):
  # Prepare xs to evaluate
  x = []
  for i in range(num_x):
    x.append(x_0 + i * h)

  # Prepare arrays to put y values in
  ys = []
  for i in range(len(functions)):
    ys.append([])

  # Put in ics as first element of y lists
  for i,element in enumerate(ys):
    element.append(ics[i])

main_routine()
print ys
