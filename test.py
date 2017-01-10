def f1():
  print "f1"

def f2():
  print "f2"

functions = []
functions.append(f1)
functions.append(f2)

for i in functions:
  i()
