from pylab import *
g,L = 1,1
xvalues, yvalues = meshgrid(arange(-8, 8, 0.1), arange(-3, 3, 0.1))
xdot = yvalues
ydot = -g/L*sin(xvalues)
streamplot(xvalues, yvalues, xdot, ydot)
grid(); show()
