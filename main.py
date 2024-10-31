import numpy as np
from numpy.linalg import norm, inv

def interior_point_method(C, A, x0, epsilon, alpha):
    m, n = A.shape
    max_iter = 100

    x = x0
    for iteration in range(1, max_iter):
        v = x
        D = np.diag(x)
        AA = np.dot(A, D)
        cc = np.dot(D, C)
        I = np.eye(n)
        F = np.dot(AA, np.transpose(AA))
        FI = inv(F)
        H = np.dot(np.transpose(AA), FI)
        P = np.subtract(I, np.dot(H, AA))
        cp = np.dot(P, cc)
        nu = np.abs(np.min(cp))
        y = np.add(np.ones(n, float), (alpha / nu) * cp)
        yy = np.dot(D, y)
        x = yy

        if iteration == 1 or iteration == 2 or iteration == 3 or iteration == 4:
            print(f"In iteration {iteration} we have x = {x}\n")

        if norm(np.subtract(yy, v), ord=2) < epsilon:
            break

    return x, np.dot(C, x)

# Test the function with different inputs
C = np.array([1, 1, 0, 0], float)  # Objective function coefficients
A = np.array([[2, 4, 1, 0], [1, 3, 0, -1]], float)  # Constraint coefficients
b = np.array([16, 9], float)  # Right-hand side numbers
x0 = np.array([1/2, 7/2, 1, 2], float)  # Initial starting point
epsilon =  0.00001  # Approximation accuracy

# Solve using Interior-Point method with alpha = 0.5
alpha_05 = 0.5
x_star_05, obj_value_05 = interior_point_method(C, A, x0, epsilon, alpha_05)

# Solve using Interior-Point method with alpha = 0.9
alpha_09 = 0.9
x_star_09, obj_value_09 = interior_point_method(C, A, x0, epsilon, alpha_09)

# Output the results
if x_star_05 is not None:
    print(f"Alpha = 0.5:")
    print(f"Decision variables: {x_star_05}")
    print(f"Objective function value: {obj_value_05}")
else:
    print("The method is not applicable or the problem does not have a solution!")

if x_star_09 is not None:
    print(f"Alpha = 0.9:")
    print(f"Decision variables: {x_star_09}")
    print(f"Objective function value: {obj_value_09}")
else:
    print("The method is not applicable or the problem does not have a solution!")
