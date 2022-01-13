# Documentation 

The design variable discipline handles the variables that are used to drive the optimization.

# B-Splines

To reduce the number of degree of freedom for the optimizer, the design variables are reconstructed from b-spline functions.
A b-spline function is a continuous composite function, that creates smooth curves according to control points.

The optimizer has control over the poles, and moves them (on the y-axis) to modify the B-Spline function. Then the value
of the design variables for each year are calculated using the B-Spline function.
  
![](BSpline_example.PNG)