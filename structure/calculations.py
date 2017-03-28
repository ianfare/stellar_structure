#!/usr/bin/python

# NOTE: from Alison, f* at point where the core meets the envelope (i.e. at the position as all the other * here) = 1


from math import *

X = 0.7
Z = 0.02
Y = 1 - X - Z

K0 = 4.35e25*Z*(1.0+X)
E0 = 10.0**(-29.0)*X**2
a = 4.0*(5.67e-5)/2.998e10
c = 2.998e10
R = 8.31e7
mu = 1.0/(2.0*X+0.75*Y+0.5*Z)
G = 6.674e-8
M = 1.98855e33

# From notes.txt:
xstar = 0.8222426499
qstar = 0.9912198715
pstar = 0.0152805856
fstar = 1.0
tstar = 0.0862223787
#tstar = 10**(-1.0862223787)

xmatch = 10.0116
qmatch = 15.8904896167
pmatch = 0.000154476422363
fmatch = 1.30646023557
tmatch = 0.105298613559

x0 = xstar/xmatch
q0 = qstar/qmatch
p0 = pstar/pmatch
f0 = fstar/fmatch
t0 = tstar/tmatch

print "x0 = " + str(x0)
print "q0 = " + str(q0)
print "p0 = " + str(p0)
print "f0 = " + str(f0)
print "t0 = " + str(t0)

C = t0**(9.5)*x0/(p0**2*f0)
D = f0/(p0**2*t0**2*x0**3)

print "C = " + str(C)
print "D = " + str(D)

Radius = (1.0/(C*D)*E0*(R/mu)**3.5*(4.0*pi)**(-4.0)*G**(-3.5)*M**(0.5)*3.0*K0/(4.0*a*c))**(1.0/6.5)

print "Radius = " + str(Radius) + " cm"
print "       = " + str(Radius/1e10) + "e10 cm"
print "       = " + str(Radius/6.957e10) + " Rsun"

###### HERE AND BELOW WRONG R CALCULATION
alphabeta = 3.0*K0*E0/(4.0*a*c*(4.0*pi)**4.0)*(R/mu)**3.5*G**(-3.5)*sqrt(M)
#print "alphabeta = " + str(alphabeta)

radius = (p0**4.0*x0**2.0/(t0**(7.5))*alphabeta)**(1.0/6.5)
print "radius = " + str(radius) + " cm"
print "       = " + str(radius/1e10) + "e10 cm"

