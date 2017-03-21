#!/usr/bin/python

import subprocess

for i in range(20000000):
  print str(0.711+float(i)/10000000000)
  subprocess.Popen("~/school/stellar_structure/structure/structure.py " + str(0.711+float(i)/10000000000),shell=True).wait()
