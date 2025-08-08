import sys
import math
import cmath
import statistics
import hashlib
import shutil
import random
import time
import base64
import re
from collections import deque, Counter
from typing import List, Tuple, Dict, Any, Callable, Union

# Optional libraries (uses them if installed)
try:
    import numpy as np

    numpy_available = True
except ImportError:
    numpy_available = False

try:
    import sympy as sp

    sympy_available = True
except ImportError:
    sympy_available = False

try:
    import plotext as plt

    plotext_available = True
except ImportError:
    plotext_available = False


# --------------------
# Helpers and utilities
# --------------------

def read_float(prompt: str = "Enter a number: ") -> float | None:
    """Reads a float with error handling; returns None if invalid."""
    s = input(prompt).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        print("Invalid input. Use a number (e.g., 3.14).")
        return None


def read_int(prompt: str = "Enter an integer: ") -> int | None:
    """Reads an integer with error handling; returns None if invalid."""
    s = input(prompt).strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        print("Invalid input. Use an integer (e.g., 42).")
        return None


def ask_yes_no(prompt: str) -> bool:
    """Asks a yes/no question."""
    r = input(prompt + " (y/n): ").strip().lower()
    return r.startswith('y')


def clear_console():
    """Clears the console screen."""
    try:
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    except Exception:
        pass


# --------------------
# Extra math functions
# --------------------

def numerical_derivative(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    """Calculates the numerical derivative of f at x using central finite differences."""
    return (f(x + h) - f(x - h)) / (2 * h)


def bisection_method(f: Callable[[float], float], a: float, b: float, tol: float = 1e-8, max_iter: int = 100) -> Tuple[
    float, int]:
    """Bisection method to find a root of f between [a,b]."""
    fa = f(a);
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    for i in range(max_iter):
        c = (a + b) / 2.0
        fc = f(c)
        if abs(fc) < tol or (b - a) / 2 < tol:
            return c, i + 1
        if fa * fc <= 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return (a + b) / 2.0, max_iter


def newton_method(f: Callable[[float], float], df: Callable[[float], float], x0: float, tol: float = 1e-10,
                  max_iter: int = 100) -> Tuple[float, int]:
    """Newton-Raphson method, returns (root, iterations)."""
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ZeroDivisionError("Derivative is zero; choose another x0.")
        x_next = x - fx / dfx
        if abs(x_next - x) < tol:
            return x_next, i + 1
        x = x_next
    return x, max_iter


def trapezoidal_integral(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Numerical integral using the trapezoidal method."""
    if n < 1: n = 1
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b))
    for k in range(1, n):
        s += f(a + k * h)
    return s * h


def simpson_integral(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Numerical integral using Simpson's rule (n must be even)."""
    if n % 2 == 1: n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for k in range(1, n):
        coef = 4 if k % 2 == 1 else 2
        s += coef * f(a + k * h)
    return s * h / 3


def sieve_of_eratosthenes(n: int) -> List[int]:
    """Generates a list of prime numbers up to n (inclusive)."""
    if n < 2: return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    p = 2
    while p * p <= n:
        if sieve[p]:
            for multiple in range(p * p, n + 1, p):
                sieve[multiple] = False
        p += 1
    return [i for i, prime in enumerate(sieve) if prime]


def factorization(n: int) -> Dict[int, int]:
    """Returns prime factorization as a dictionary {p:exp}."""
    if n == 0: return {0: 1}
    n = abs(int(n))
    res = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            res[d] = res.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        res[n] = res.get(n, 0) + 1
    return res


def fibonacci_n(n: int) -> int:
    """Calculates the n-th Fibonacci number."""
    if n < 0: raise ValueError("n must be >= 0")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def euler_totient(n: int) -> int:
    """Calculates Euler's totient function phi(n)."""
    if not isinstance(n, int) or n <= 0: raise ValueError("n must be a positive integer.")
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0: n //= p
            result -= result // p
        p += 1
    if n > 1: result -= result // n
    return result


def modular_exponentiation(base: int, exp: int, mod: int) -> int:
    """Calculates (base^exp) % mod efficiently."""
    return pow(base, exp, mod)


def gcd(a: int, b: int) -> int:
    """Calculates the Greatest Common Divisor (GCD) using the Euclidean Algorithm."""
    while b:
        a, b = b, a % b
    return a


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Calculates the GCD and Bézout's coefficients (ax + by = gcd)."""
    if a == 0:
        return b, 0, 1
    d, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return d, x, y


def polynomial_roots():
    """Finds the roots of a polynomial."""
    print("Enter the coefficients of the polynomial, from highest to lowest. Ex: 1 -3 2 for x^2 - 3x + 2")
    s_coeffs = input("Coefficients separated by space: ").strip().split()
    try:
        coeffs = [float(c) for c in s_coeffs]
    except ValueError:
        print("Invalid input. Use only numbers.")
        return
    if not coeffs:
        print("No coefficients entered.")
        return

    if numpy_available:
        try:
            roots = np.roots(coeffs)
            print("Roots:")
            for root in roots:
                if abs(root.imag) < 1e-12:
                    print(f"  {root.real:.6g}")
                else:
                    print(f"  {root:.6g}")
        except Exception as e:
            print(f"Error calculating roots: {e}")
    else:
        print("numpy not available. Cannot calculate roots generally.")
        if len(coeffs) == 3:
            a, b, c = coeffs
            delta = b ** 2 - 4 * a * c
            if delta >= 0:
                print(
                    f"Roots (Bhaskara's formula): {(-b + math.sqrt(delta)) / (2 * a):.6g} and {(-b - math.sqrt(delta)) / (2 * a):.6g}")
            else:
                print(
                    f"Roots (Bhaskara's formula): {(-b + cmath.sqrt(delta)) / (2 * a):.6g} and {(-b - cmath.sqrt(delta)) / (2 * a):.6g}")


# --------------------
# Linear algebra (uses numpy if available)
# --------------------

def matrix_from_input() -> Union['np.ndarray', List[List[float]], None]:
    """Reads a matrix row by row with numbers separated by space."""
    print("Enter the matrix row by row. Empty line to finish.")
    rows = []
    while True:
        line = input().strip()
        if line == "":
            break
        vals = line.split()
        try:
            row = [float(v) for v in vals]
            rows.append(row)
        except ValueError:
            print("Invalid line. Use numbers separated by space.")
    if not rows: return None
    ncols = len(rows[0])
    for r in rows:
        if len(r) != ncols:
            print("All rows must have the same number of columns.")
            return None
    return np.array(rows, dtype=float) if numpy_available else rows


def vector_from_input() -> Union['np.ndarray', List[float], None]:
    """Reads a vector from a line of numbers separated by space."""
    s = input("Enter the vector (numbers separated by space): ")
    try:
        vec = [float(v) for v in s.split()]
        return np.array(vec, dtype=float) if numpy_available else vec
    except ValueError:
        print("Invalid input. Use numbers separated by space.")
        return None


def add_matrices(A: Any, B: Any) -> Any:
    if numpy_available:
        return A + B
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def multiply_matrices(A: Any, B: Any) -> Any:
    if numpy_available:
        return A.dot(B)
    if len(A[0]) != len(B):
        raise ValueError("Number of columns in A must equal number of rows in B.")
    n, m, p = len(A), len(A[0]), len(B[0])
    C = [[0.0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i][j] += A[i][k] * B[k][j]
    return C


def determinant(A: Any) -> float:
    if numpy_available:
        return float(np.linalg.det(A))
    n = len(A)
    if n != len(A[0]):
        raise ValueError("Non-square matrix.")
    if n == 1: return A[0][0]
    if n == 2: return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    total = 0
    for col in range(n):
        sub = [row[:col] + row[col + 1:] for row in A[1:]]
        sign = (-1) ** col
        total += sign * A[0][col] * determinant(sub)
    return total


def inverse_matrix(A: Any) -> Any:
    if numpy_available:
        return np.linalg.inv(A)
    n = len(A)
    if n != len(A[0]):
        raise ValueError("Non-square matrix.")
    M = [list(map(float, row)) + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    for i in range(n):
        pivot = M[i][i]
        if abs(pivot) < 1e-12:
            for r in range(i + 1, n):
                if abs(M[r][i]) > 1e-12:
                    M[i], M[r] = M[r], M[i]
                    pivot = M[i][i]
                    break
        if abs(pivot) < 1e-12: raise ValueError("Singular matrix")
        for j in range(2 * n): M[i][j] /= pivot
        for r in range(n):
            if r != i:
                fac = M[r][i]
                for j in range(2 * n): M[r][j] -= fac * M[i][j]
    inv = [row[n:] for row in M]
    return inv


def transpose_matrix(A: Any) -> Any:
    if numpy_available:
        return A.T
    if not A or not A[0]: return []
    rows, cols = len(A), len(A[0])
    return [[A[j][i] for j in range(rows)] for i in range(cols)]


def eigen_values_vectors(A: Any):
    if numpy_available:
        try:
            w, v = np.linalg.eig(A)
            print("Eigenvalues:");
            print(w)
            print("\nEigenvectors (columns):");
            print(v)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("numpy not available. Cannot calculate eigenvalues/eigenvectors without numpy.")


def solve_linear_system(A: Any, b: Any):
    if numpy_available:
        try:
            x = np.linalg.solve(A, b)
            print("Solution x:");
            print(x)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("numpy not available. Cannot solve linear systems without numpy.")


def lu_decomposition(A: Any) -> Tuple[Any, Any]:
    if numpy_available:
        from scipy.linalg import lu
        P, L, U = lu(A)
        print("LU decomposition (with pivoting)")
        print("P matrix (permutation):\n", P)
        print("L matrix (lower):\n", L)
        print("U matrix (upper):\n", U)
        return L, U
    n = len(A)
    if n != len(A[0]): raise ValueError("Non-square matrix.")
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [row[:] for row in A]
    for j in range(n):
        L[j][j] = 1.0
        if abs(U[j][j]) < 1e-12:
            raise ValueError("Cannot perform LU decomposition without pivoting.")
        for i in range(j + 1, n):
            factor = U[i][j] / U[j][j]
            L[i][j] = factor
            for k in range(j, n):
                U[i][k] -= factor * U[j][k]
    return L, U


def qr_decomposition(A: Any) -> Tuple[Any, Any]:
    if numpy_available:
        Q, R = np.linalg.qr(A)
        return Q, R
    m, n = len(A), len(A[0])
    if m < n: raise ValueError("Matrix A must have more or the same number of rows as columns.")
    Q = [[0.0 for _ in range(n)] for _ in range(m)]
    R = [[0.0 for _ in range(n)] for _ in range(n)]
    V = [row[:] for row in A]
    for j in range(n):
        vj = [V[i][j] for i in range(m)]
        mag_vj_sq = sum(x ** 2 for x in vj)
        R[j][j] = math.sqrt(mag_vj_sq)
        for i in range(m): Q[i][j] = vj[i] / R[j][j]
        for k in range(j + 1, n):
            vk = [V[i][k] for i in range(m)]
            R[j][k] = sum(Q[i][j] * vk[i] for i in range(m))
            for i in range(m): V[i][k] -= R[j][k] * Q[i][j]
    return Q, R


def covariance_matrix(data: Any) -> Any:
    if numpy_available:
        return np.cov(data, rowvar=False)

    if not isinstance(data, list) or not data or not isinstance(data[0], list):
        raise ValueError("Data must be a list of lists (samples per column).")

    m, n = len(data), len(data[0])
    if m < 2:
        raise ValueError("At least two samples are required to calculate covariance.")

    data_t = [[data[j][i] for j in range(m)] for i in range(n)]
    means = [sum(col) / m for col in data_t]

    cov_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            soma = 0.0
            for k in range(m):
                soma += (data[k][i] - means[i]) * (data[k][j] - means[j])
            cov_matrix[i][j] = soma / (m - 1)
    return cov_matrix


def vandermonde_matrix(v: List[float], n: int = None) -> Any:
    if numpy_available:
        return np.vander(v, N=n)
    if n is None: n = len(v)
    m = len(v)
    if n <= 0 or m <= 0: raise ValueError("Dimensions must be positive.")
    matrix = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            matrix[i][j] = v[i] ** (n - 1 - j)
    return matrix


# --------------------
# Geometry and Vector Calculus
# --------------------

def geometry_solids():
    """Calculates the surface area and volume of common geometric solids."""
    print("\n--- SOLIDS GEOMETRY ---")
    print("a) Sphere\nb) Cube\nc) Cylinder\nd) Cone\n0) Back")
    choice = input("Choose: ").strip().lower()

    try:
        if choice == 'a':
            r = read_float("Radius (r): ")
            if r is None or r < 0: return
            area = 4 * math.pi * r ** 2
            volume = (4 / 3) * math.pi * r ** 3
            print(f"Surface area = {area:.6g}")
            print(f"Volume = {volume:.6g}")
        elif choice == 'b':
            l = read_float("Side (l): ")
            if l is None or l < 0: return
            area = 6 * l ** 2
            volume = l ** 3
            print(f"Surface area = {area:.6g}")
            print(f"Volume = {volume:.6g}")
        elif choice == 'c':
            r = read_float("Radius (r): ")
            h = read_float("Height (h): ")
            if None in (r, h) or r < 0 or h < 0: return
            base_area = math.pi * r ** 2
            lateral_area = 2 * math.pi * r * h
            total_area = 2 * base_area + lateral_area
            volume = base_area * h
            print(f"Surface area = {total_area:.6g}")
            print(f"Volume = {volume:.6g}")
        elif choice == 'd':
            r = read_float("Radius (r): ")
            h = read_float("Height (h): ")
            if None in (r, h) or r < 0 or h < 0: return
            sl = math.sqrt(r ** 2 + h ** 2)
            base_area = math.pi * r ** 2
            lateral_area = math.pi * r * sl
            total_area = base_area + lateral_area
            volume = (1 / 3) * base_area * h
            print(f"Surface area = {total_area:.6g}")
            print(f"Volume = {volume:.6g}")
        elif choice == '0':
            return
        else:
            print("Invalid option.")
    except Exception as e:
        print(f"Error: {e}")


def distance_between_points():
    """Calculates the distance between two points in 2D or 3D."""
    print("a) 2D\nb) 3D")
    dim = input("Dimensions: ").strip().lower()
    try:
        if dim == 'a':
            x1 = read_float("x1: ");
            y1 = read_float("y1: ")
            x2 = read_float("x2: ");
            y2 = read_float("y2: ")
            if None in (x1, y1, x2, y2): return
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            print(f"Distance = {dist:.6g}")
        elif dim == 'b':
            x1 = read_float("x1: ");
            y1 = read_float("y1: ");
            z1 = read_float("z1: ")
            x2 = read_float("x2: ");
            y2 = read_float("y2: ");
            z2 = read_float("z2: ")
            if None in (x1, y1, z1, x2, y2, z2): return
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            print(f"Distance = {dist:.6g}")
        else:
            print("Invalid option.")
    except Exception as e:
        print(f"Error: {e}")


def vector_operations():
    """Calculates dot product, cross product, and vector projection."""
    print("\n--- VECTOR OPERATIONS ---")
    print("a) Dot Product\nb) Cross Product\nc) Vector Projection\n0) Back")
    choice = input("Choose: ").strip().lower()

    def read_vector3d(prompt):
        s = input(f"{prompt} (e.g., 1 2 3): ").strip().split()
        if len(s) != 3:
            print("Invalid input. Enter 3 numbers.")
            return None
        try:
            return [float(v) for v in s]
        except ValueError:
            print("Invalid input. Use only numbers.")
            return None

    try:
        if choice == 'a':
            v1 = read_vector3d("Vector 1");
            v2 = read_vector3d("Vector 2")
            if None in (v1, v2): return
            res = sum(x * y for x, y in zip(v1, v2))
            print(f"Dot Product = {res:.6g}")
        elif choice == 'b':
            v1 = read_vector3d("Vector 1");
            v2 = read_vector3d("Vector 2")
            if None in (v1, v2): return
            res = [v1[1] * v2[2] - v1[2] * v2[1],
                   v1[2] * v2[0] - v1[0] * v2[2],
                   v1[0] * v2[1] - v1[1] * v2[0]]
            print(f"Cross Product = ({res[0]:.6g}, {res[1]:.6g}, {res[2]:.6g})")
        elif choice == 'c':
            v1 = read_vector3d("Vector to project");
            v2 = read_vector3d("Projection vector")
            if None in (v1, v2): return
            dot_prod = sum(x * y for x, y in zip(v1, v2))
            mag_sq_v2 = sum(x ** 2 for x in v2)
            if mag_sq_v2 == 0:
                print("Projection vector cannot be zero.")
                return
            scalar = dot_prod / mag_sq_v2
            res = [scalar * x for x in v2]
            print(f"Projection = ({res[0]:.6g}, {res[1]:.6g}, {res[2]:.6g})")
        elif choice == '0':
            return
        else:
            print("Invalid option.")
    except Exception as e:
        print(f"Error: {e}")


def vector_rotation():
    """Rotates a vector in 2D or 3D."""
    print("a) 2D Rotation\nb) 3D Rotation (X, Y, Z Axes)\n0) Back")
    choice = input("Choose: ").strip().lower()

    try:
        if choice == 'a':
            x, y = read_float("x: "), read_float("y: ")
            angle_deg = read_float("Angle (degrees): ")
            if None in (x, y, angle_deg): return
            angle_rad = math.radians(angle_deg)
            x_new = x * math.cos(angle_rad) - y * math.sin(angle_rad)
            y_new = x * math.sin(angle_rad) + y * math.cos(angle_rad)
            print(f"Rotated vector: ({x_new:.6g}, {y_new:.6g})")
        elif choice == 'b':
            x, y, z = read_float("x: "), read_float("y: "), read_float("z: ")
            angle_deg = read_float("Angle (degrees): ")
            axis = input("Rotation axis (x, y, z): ").strip().lower()
            if None in (x, y, z, angle_deg) or axis not in ['x', 'y', 'z']: return
            angle_rad = math.radians(angle_deg)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

            x_new, y_new, z_new = x, y, z
            if axis == 'x':
                y_new = y * cos_a - z * sin_a
                z_new = y * sin_a + z * cos_a
            elif axis == 'y':
                x_new = x * cos_a + z * sin_a
                z_new = -x * sin_a + z * cos_a
            elif axis == 'z':
                x_new = x * cos_a - y * sin_a
                y_new = x * sin_a + y * cos_a

            print(f"Rotated vector: ({x_new:.6g}, {y_new:.6g}, {z_new:.6g})")
        elif choice == '0':
            return
        else:
            print("Invalid option.")
    except Exception as e:
        print(f"Error: {e}")


# --------------------
# Basic Physics and Engineering
# --------------------

def uniform_motion(s0: float | None = None, v: float | None = None, t: float | None = None):
    """Uniform Motion: s = s0 + v*t"""
    s0 = s0 if s0 is not None else read_float("Initial position s0: ")
    v = v if v is not None else read_float("Velocity v: ")
    t = t if t is not None else read_float("Time t: ")
    if None in (s0, v, t): print("Missing parameter(s)."); return
    s = s0 + v * t
    print(f"Final position: s = {s}")


def uniform_accelerated_motion(s0: float | None = None, v0: float | None = None, a: float | None = None,
                               t: float | None = None):
    """Uniform Accelerated Motion: s = s0 + v0*t + 0.5*a*t^2 ; v = v0 + a*t"""
    s0 = s0 if s0 is not None else read_float("Initial position s0: ")
    v0 = v0 if v0 is not None else read_float("Initial velocity v0: ")
    a = a if a is not None else read_float("Acceleration a: ")
    t = t if t is not None else read_float("Time t: ")
    if None in (s0, v0, a, t): print("Missing parameter(s)."); return
    s = s0 + v0 * t + 0.5 * a * t * t
    v = v0 + a * t
    print(f"Final position: s = {s}")
    print(f"Final velocity: v = {v}")


def oblique_launch(v0: float | None = None, angle_deg: float | None = None, g: float = 9.80665):
    """Given v0 and angle, calculates range, flight time, and max height."""
    v0 = v0 if v0 is not None else read_float("Initial velocity v0 (m/s): ")
    angle_deg = angle_deg if angle_deg is not None else read_float("Launch angle (degrees): ")
    if None in (v0, angle_deg): print("Missing parameter(s)."); return
    th = math.radians(angle_deg)
    flight_time = 2 * v0 * math.sin(th) / g
    launch_range = v0 * math.cos(th) * flight_time
    max_height = (v0 ** 2) * (math.sin(th) ** 2) / (2 * g)
    print(f"Flight time: {flight_time:.6g} s")
    print(f"Horizontal range: {launch_range:.6g} m")
    print(f"Maximum height: {max_height:.6g} m")


def ohm_law():
    print("Which variable do you want to calculate?")
    print("a) Voltage (V)\nb) Current (I)\nc) Resistance (R)")
    choice = input("Choose: ").strip().lower()
    try:
        if choice == 'a':
            i = read_float("Current (I): ");
            r = read_float("Resistance (R): ")
            print(f"V = {i * r:.6g} V")
        elif choice == 'b':
            v = read_float("Voltage (V): ");
            r = read_float("Resistance (R): ")
            if r == 0:
                print("Resistance cannot be zero.")
            else:
                print(f"I = {v / r:.6g} A")
        elif choice == 'c':
            v = read_float("Voltage (V): ");
            i = read_float("Current (I): ")
            if i == 0:
                print("Current cannot be zero.")
            else:
                print(f"R = {v / i:.6g} Ω")
        else:
            print("Invalid option.")
    except TypeError:
        print("Invalid input.")


def kinetic_energy():
    try:
        m = read_float("Mass (m): ");
        v = read_float("Velocity (v): ")
        if None in (m, v): return
        ke = 0.5 * m * v ** 2
        print(f"Kinetic Energy = {ke:.6g} J")
    except TypeError:
        print("Invalid input.")


def ideal_gas_law():
    print("Ideal Gas Law: PV = nRT")
    print("Which variable do you want to calculate?")
    print("a) Pressure (P)\nb) Volume (V)\nc) Amount of substance (n)\nd) Temperature (T)")
    choice = input("Choose: ").strip().lower()
    R = 8.314462618
    try:
        if choice == 'a':
            v = read_float("Volume (V, m^3): ");
            n = read_float("Amount (n, mol): ");
            t = read_float("Temperature (T, K): ")
            if v == 0:
                print("Volume cannot be zero.")
            else:
                print(f"P = {(n * R * t) / v:.6g} Pa")
        elif choice == 'b':
            p = read_float("Pressure (P, Pa): ");
            n = read_float("Amount (n, mol): ");
            t = read_float("Temperature (T, K): ")
            if p == 0:
                print("Pressure cannot be zero.")
            else:
                print(f"V = {(n * R * t) / p:.6g} m^3")
        elif choice == 'c':
            p = read_float("Pressure (P, Pa): ");
            v = read_float("Volume (V, m^3): ");
            t = read_float("Temperature (T, K): ")
            if t == 0:
                print("Temperature cannot be zero.")
            else:
                print(f"n = {(p * v) / (R * t):.6g} mol")
        elif choice == 'd':
            p = read_float("Pressure (P, Pa): ");
            v = read_float("Volume (V, m^3): ");
            n = read_int("Amount (n, mol): ")
            if n == 0:
                print("Amount of substance cannot be zero.")
            else:
                print(f"T = {(p * v) / (R * n):.6g} K")
        else:
            print("Invalid option.")
    except TypeError:
        print("Invalid input.")


def universal_gravitation():
    G = 6.67430e-11
    try:
        m1 = read_float("Mass of body 1 (kg): ");
        m2 = read_float("Mass of body 2 (kg): ")
        r = read_float("Distance between centers (m): ")
        if None in (m1, m2, r): return
        if r == 0:
            print("Distance cannot be zero.")
        else:
            f = (G * m1 * m2) / r ** 2
            print(f"Gravitational force (F) = {f:.6g} N")
    except TypeError:
        print("Invalid input.")


# --------------------
# Scientific conversions
# --------------------

CONSTANTS = {
    'pi': math.pi, 'e': math.e, 'phi': (1 + math.sqrt(5)) / 2, 'c': 299792458,
    'G': 6.67430e-11, 'h': 6.62607015e-34, 'kB': 1.380649e-23, 'NA': 6.02214076e23
}


def temperature_conversion():
    print("a) C -> F\nb) F -> C\nc) C -> K\nd) K -> C")
    choice = input("Choose: ").strip().lower()
    if choice == 'a':
        c = read_float("°C: ");
        print(f"{c} °C = {c * 9 / 5 + 32} °F")
    elif choice == 'b':
        f = read_float("°F: ");
        print(f"{f} °F = {(f - 32) * 5 / 9} °C")
    elif choice == 'c':
        c = read_float("°C: ");
        print(f"{c} °C = {c + 273.15} K")
    elif choice == 'd':
        k = read_float("K: ");
        print(f"{k} K = {k - 273.15} °C")
    else:
        print("Invalid option.")


def speed_conversion():
    print("a) m/s -> km/h\nb) km/h -> m/s\nc) m/s -> mph\nd) mph -> m/s")
    choice = input("Choose: ").strip().lower()
    if choice == 'a':
        v = read_float("m/s: ");
        print(f"{v} m/s = {v * 3.6} km/h")
    elif choice == 'b':
        v = read_float("km/h: ");
        print(f"{v} km/h = {v / 3.6} m/s")
    elif choice == 'c':
        v = read_float("m/s: ");
        print(f"{v} m/s = {v * 2.2369362920544} mph")
    elif choice == 'd':
        v = read_float("mph: ");
        print(f"{v} mph = {v / 2.2369362920544} m/s")
    else:
        print("Invalid option.")


def roman_to_arabic():
    roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    s = input("Enter a Roman numeral: ").strip().upper()
    total, prev_val = 0, 0
    try:
        for c in reversed(s):
            val = roman_map[c]
            total += val if val >= prev_val else -val
            prev_val = val
        print(f"The Arabic numeral is: {total}")
    except KeyError:
        print("Invalid input. Use only I, V, X, L, C, D, M.")


def arabic_to_roman():
    val = read_int("Enter an Arabic number (1 to 3999): ")
    if val is None or not (1 <= val <= 3999):
        print("Invalid number.");
        return
    romans = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'),
              (90, 'XC'), (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'),
              (5, 'V'), (4, 'IV'), (1, 'I')]
    result = ""
    for arabic, roman in romans:
        while val >= arabic:
            result += roman
            val -= arabic
    print(f"The Roman numeral is: {result}")


def base64_converter():
    print("a) Encode\nb) Decode")
    choice = input("Choose: ").strip().lower()
    if choice == 'a':
        s = input("Enter text to encode: ").encode('utf-8')
        encoded = base64.b64encode(s)
        print("Result:", encoded.decode('utf-8'))
    elif choice == 'b':
        s = input("Enter text to decode: ").encode('utf-8')
        try:
            decoded = base64.b64decode(s)
            print("Result:", decoded.decode('utf-8'))
        except Exception as e:
            print(f"Error decoding: {e}")
    else:
        print("Invalid option.")


# --------------------
# Statistics and probability
# --------------------

def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    u = (x - mu) / sigma
    return (1.0 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-u * u / 2)


def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def binomial_pmf(k: int, n: int, p: float) -> float:
    from math import comb
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def simple_linear_regression():
    print("Simple Linear Regression: y = ax + b")
    print("Enter X values separated by space:")
    try:
        x_vals = list(map(float, input().split()))
        print("Enter Y values separated by space:")
        y_vals = list(map(float, input().split()))
        if len(x_vals) != len(y_vals) or len(x_vals) < 2:
            print("The lists must have the same size and at least 2 points.");
            return

        n = len(x_vals)
        sum_x = sum(x_vals);
        sum_y = sum(y_vals)
        sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
        sum_x2 = sum(x ** 2 for x in x_vals)

        a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        b = (sum_y / n) - a * (sum_x / n)
        y_mean = sum_y / n
        ss_total = sum((y - y_mean) ** 2 for y in y_vals)
        ss_residual = sum((y - (a * x + b)) ** 2 for x, y in zip(x_vals, y_vals))
        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 1

        print(f"Equation: y = {a:.6g}x + {b:.6g}")
        print(f"Coefficient of determination (R^2): {r_squared:.6g}")
    except Exception as e:
        print(f"Error: {e}")


def correlation_coefficient():
    print("Pearson's Correlation Coefficient Calculation")
    print("Enter X values separated by space:")
    try:
        x_vals = list(map(float, input().split()))
        print("Enter Y values separated by space:")
        y_vals = list(map(float, input().split()))
        if len(x_vals) != len(y_vals) or len(x_vals) < 2:
            print("The lists must have the same size and at least 2 points.");
            return

        if numpy_available:
            corr = np.corrcoef(x_vals, y_vals)[0, 1]
            print(f"Correlation coefficient (r): {corr:.6g}")
        else:
            n = len(x_vals);
            mean_x = statistics.mean(x_vals);
            mean_y = statistics.mean(y_vals)
            cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals))
            stdev_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_vals))
            stdev_y = math.sqrt(sum((y - mean_y) ** 2 for y in y_vals))
            if stdev_x * stdev_y == 0:
                print("Could not calculate correlation coefficient (zero standard deviation).")
            else:
                corr = cov / (stdev_x * stdev_y)
                print(f"Correlation coefficient (r): {corr:.6g}")
    except Exception as e:
        print(f"Error: {e}")


# --------------------
# Programming utilities
# --------------------

class Stack:
    """Basic implementation of a stack (LIFO)."""

    def __init__(self): self._list: List[Any] = []

    def push(self, item: Any): self._list.append(item)

    def pop(self) -> Any | None: return self._list.pop() if self._list else None

    def peek(self) -> Any | None: return self._list[-1] if self._list else None

    def is_empty(self) -> bool: return not self._list

    def __repr__(self) -> str: return f"Stack: {self._list}"


class Queue:
    """Basic implementation of a queue (FIFO)."""

    def __init__(self): self._deque: deque[Any] = deque()

    def enqueue(self, item: Any): self._deque.append(item)

    def dequeue(self) -> Any | None: return self._deque.popleft() if self._deque else None

    def peek(self) -> Any | None: return self._deque[0] if self._deque else None

    def is_empty(self) -> bool: return not self._deque

    def __repr__(self) -> str: return f"Queue: {list(self._deque)}"


def data_structures_menu():
    stack = Stack();
    queue = Queue()
    print("Choose the structure: a) Stack b) Queue")
    choice = input().strip().lower()
    ds = stack if choice == 'a' else queue
    while True:
        print(f"\n{ds}")
        print("1. Add 2. Remove 3. View top/front 4. Back")
        sub_choice = input("Choose: ").strip()
        if sub_choice == '1':
            item = input("Item to add: ")
            ds.push(item) if choice == 'a' else ds.enqueue(item)
        elif sub_choice == '2':
            item = ds.pop() if choice == 'a' else ds.dequeue()
            print(f"Removed: {item}")
        elif sub_choice == '3':
            item = ds.peek()
            print(f"Top/Front: {item}")
        elif sub_choice == '4':
            break
        else:
            print("Invalid option.")


def text_to_ascii():
    s = input("Enter the text: ")
    codes = [str(ord(c)) for c in s]
    print("ASCII / Unicode codes:", " ".join(codes))


def base_converter():
    n = input("Number (integer) to convert: ").strip()
    try:
        base_from = int(input("Input base (2-36): "))
        base_to = int(input("Output base (2-36): "))
    except ValueError:
        print("Invalid base."); return
    try:
        value = int(n, base_from)
    except ValueError:
        print("Invalid number in the given base."); return
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if value == 0: print("0"); return
    out = ""
    v = value
    while v > 0:
        out = digits[v % base_to] + out
        v //= base_to
    print(f"{n} (base {base_from}) = {out} (base {base_to})")


def generate_hash():
    s = input("Text to hash: ").encode('utf-8')
    print("MD5:", hashlib.md5(s).hexdigest())
    print("SHA1:", hashlib.sha1(s).hexdigest())
    print("SHA256:", hashlib.sha256(s).hexdigest())


def caesar_cipher():
    txt = input("Text: ")
    k = read_int("Shift (e.g., 3): ")
    if k is None: return
    res = []
    for ch in txt:
        if ch.isalpha():
            base = 'A' if ch.isupper() else 'a'
            res.append(chr((ord(ch) - ord(base) + k) % 26 + ord(base)))
        else:
            res.append(ch)
    print("Result:", "".join(res))


def vigenere_cipher():
    txt = input("Text (letters only): ").upper().replace(" ", "")
    key = input("Key (letters only): ").upper().replace(" ", "")
    if not txt or not key or not txt.isalpha() or not key.isalpha():
        print("Invalid input. Use letters only.");
        return
    op = input("a) Encrypt b) Decrypt: ").strip().lower()
    res = []
    key_len = len(key)
    for i, char in enumerate(txt):
        key_char = key[i % key_len]
        key_shift = ord(key_char) - ord('A')
        if op == 'b': key_shift = -key_shift
        new_char_code = (ord(char) - ord('A') + key_shift) % 26 + ord('A')
        res.append(chr(new_char_code))
    print("Result:", "".join(res))


def rle_converter():
    print("a) Encode\nb) Decode")
    choice = input("Choose: ").strip().lower()
    if choice == 'a':
        s = input("Enter the string to compress: ")
        if not s: print("Empty string."); return
        result = [];
        count = 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                count += 1
            else:
                result.append(s[i - 1] + str(count));
                count = 1
        result.append(s[-1] + str(count))
        print("Compressed:", "".join(result))
    elif choice == 'b':
        s = input("Enter the RLE string to decompress: ")
        if not s: print("Empty string."); return
        try:
            result = ""
            for char, count_str in re.findall(r'(\D)(\d+)', s):
                result += char * int(count_str)
            print("Decompressed:", result)
        except Exception as e:
            print(f"Invalid RLE format: {e}")
    else:
        print("Invalid option.")


def stopwatch():
    input("Press ENTER to start...")
    start_time = time.time();
    paused_time = 0.0;
    is_paused = False
    while True:
        elapsed = time.time() - start_time - paused_time
        print(f"\rElapsed time: {elapsed:.2f} s", end="")
        try:
            user_input = input().strip().lower()
            if user_input == 'p' and not is_paused:
                paused_start = time.time();
                is_paused = True
                print("\nStopwatch paused. Press 'c' to continue.")
            elif user_input == 'c' and is_paused:
                paused_time += time.time() - paused_start;
                is_paused = False
                print("Stopwatch resuming...")
            elif user_input == 's':
                print("\nStopwatch stopped.");
                break
            elif not user_input:
                pass
        except KeyboardInterrupt:
            print("\nStopwatch stopped via CTRL+C.");
            break


def dice_simulation():
    num_dice = read_int("How many dice (1-100): ")
    if num_dice is None or not (1 <= num_dice <= 100):
        print("Invalid number.");
        return
    num_sides = read_int("How many sides per die (>=2): ")
    if num_sides is None or num_sides < 2:
        print("Invalid number.");
        return

    results = [random.randint(1, num_sides) for _ in range(num_dice)]
    print("\nIndividual results:", results)

    counts = Counter(results)
    print("\nSummary:")
    for side in range(1, num_sides + 1):
        freq = counts.get(side, 0)
        print(f"  Side {side}: {freq} ({freq / num_dice * 100:.2f}%)")


# --------------------
# Artificial Intelligence and Optimization
# --------------------

def fuzzy_tipping():
    print("Fuzzy Logic Tip Calculator")
    try:
        service_quality = read_float("Service quality (0-10): ")
        food_quality = read_float("Food quality (0-10): ")
        if None in (service_quality, food_quality): return

        # Membership functions (triangular/trapezoidal)
        def poor_service(x):
            """Membership function for poor service (descending ramp)."""
            if 0 <= x <= 5:
                return max(0, (5 - x) / 5)
            return 0

        def good_service(x):
            """Membership function for good service (triangular)."""
            if 2.5 <= x <= 5:
                return (x - 2.5) / 2.5
            if 5 < x <= 7.5:
                return (7.5 - x) / 2.5
            return 0

        def excellent_service(x):
            """Membership function for excellent service (ascending ramp)."""
            if 5 <= x <= 10:
                return (x - 5) / 5
            return 0

        def poor_food(x):
            """Membership function for poor food (descending ramp)."""
            if 0 <= x <= 5:
                return (5 - x) / 5
            return 0

        def good_food(x):
            """Membership function for good food (ascending ramp)."""
            if 5 <= x <= 10:
                return (x - 5) / 5
            return 0

        # Fuzzification
        poor_serv_val = poor_service(service_quality)
        good_serv_val = good_service(service_quality)
        excellent_serv_val = excellent_service(service_quality)
        poor_food_val = poor_food(food_quality)
        good_food_val = good_food(food_quality)

        # Inference rules (AND = min, OR = max)
        low_tip = max(poor_serv_val, poor_food_val)
        medium_tip = good_serv_val
        high_tip = excellent_serv_val * good_food_val

        # Defuzzification (Center of Gravity) - Simplified
        outputs = {
            "low": low_tip,
            "medium": medium_tip,
            "high": high_tip
        }
        max_output = max(outputs, key=outputs.get)
        print(f"\nDegree of membership for each tip level:")
        for k, v in outputs.items(): print(f"  {k}: {v:.2f}")
        print(f"\nInference result: '{max_output}' level tip")
    except Exception as e:
        print("Error during fuzzy logic:", e)


def gradient_descent():
    print("Gradient Descent Optimization for f(x)")
    print("Enter f(x) (e.g., x**2 + 5*x + 6)")
    if not sympy_available:
        print("SymPy not available. This function requires the sympy library.")
        return
    try:
        expr = input("f(x) = ")
        x = sp.symbols('x')
        f_sym = sp.sympify(expr)
        df_sym = sp.diff(f_sym, x)

        f = sp.lambdify(x, f_sym, 'math')
        df = sp.lambdify(x, df_sym, 'math')

        x_init = read_float("Initial guess for x: ")
        learning_rate = read_float("Learning rate (e.g., 0.1): ") or 0.1
        epochs = read_int("Number of iterations (e.g., 100): ") or 100

        if None in (x_init, learning_rate, epochs): return

        x_current = x_init
        for i in range(epochs):
            x_current = x_current - learning_rate * df(x_current)

        print(f"Approximate minimum of f(x) found at x = {x_current:.6g}")
        print(f"Function value at this point: f(x) = {f(x_current):.6g}")
    except Exception as e:
        print("An error occurred during optimization:", e)


# --------------------
# Fractals and ASCII graphics
# --------------------

def mandelbrot_ascii(width: int = 80, height: int = 40, max_iter: int = 80, x_center: float = -0.5,
                     y_center: float = 0.0, scale: float = 1.5):
    """Draws an ASCII version of the Mandelbrot set."""
    chars = " .:-=+*#%@"
    for j in range(height):
        y = y_center + (j - height / 2) * (2 * scale / height)
        row = ""
        for i in range(width):
            x = x_center + (i - width / 2) * (2 * scale / width)
            zx, zy = 0.0, 0.0
            it = 0
            while zx * zx + zy * zy <= 4.0 and it < max_iter:
                zx, zy = zx * zx - zy * zy + x, 2 * zx * zy + y
                it += 1
            row += chars[int(it / max_iter * (len(chars) - 1))]
        print(row)


# --------------------
# Menu Modules
# --------------------

def advanced_math_menu():
    while True:
        print("\n--- ADVANCED MATHEMATICS ---")
        print("1. Numerical Analysis (Bisection, Newton, Simpson, Derivative)")
        print("2. Prime Functions and Number Theory")
        print("3. Polynomial Roots")
        print("0. Back")
        choice = input("Choose: ").strip()
        if choice == '1':
            numerical_analysis_menu()
        elif choice == '2':
            number_theory_menu()
        elif choice == '3':
            polynomial_roots()
        elif choice == '0':
            break
        else:
            print("Invalid option.")


def number_theory_menu():
    print("\n--- NUMBER THEORY ---")
    print(
        "a) Prime Factorization\nb) Euler's Totient Function\nc) Modular Exponentiation\nd) GCD (Euclidean Algorithm)\ne) GCD and Bézout's coefficients")
    choice = input("Choose: ").strip().lower()
    if choice == 'a':
        n = read_int("Number to factor: ")
        if n is not None: print("Factorization:", factorization(n))
    elif choice == 'b':
        n = read_int("Number for phi(n): ")
        if n is not None:
            try:
                print(f"phi({n}) = {euler_totient(n)}")
            except ValueError as e:
                print(e)
    elif choice == 'c':
        base = read_int("Base: ");
        exp = read_int("Exponent: ");
        mod = read_int("Modulus: ")
        if None not in (base, exp, mod): print(f"({base}^{exp}) mod {mod} = {modular_exponentiation(base, exp, mod)}")
    elif choice == 'd':
        a = read_int("a: ");
        b = read_int("b: ")
        if None not in (a, b): print(f"GCD({a}, {b}) = {gcd(a, b)}")
    elif choice == 'e':
        a = read_int("a: ");
        b = read_int("b: ")
        if None not in (a, b):
            d, x, y = extended_gcd(a, b)
            print(f"GCD({a}, {b}) = {d}")
            print(f"Bézout's coefficients: {a}*{x} + {b}*{y} = {d}")
    else:
        print("Invalid option.")


def numerical_analysis_menu():
    print("\n-- Numerical Analysis --")
    print("a) Bisection\nb) Newton-Raphson\nc) Integral (Trapezoidal/Simpson)\nd) Numerical Derivative")
    choice = input("Choose: ").strip().lower()
    if choice == 'a':
        print("Define f(x) (requires sympy). Ex: x**3 - x - 2")
        if not sympy_available: print("SymPy not available."); return
        expr = input("f(x) = ")
        x = sp.symbols('x');
        f_sym = sp.lambdify(x, sp.sympify(expr), 'math')
        a = read_float("a: ");
        b = read_float("b: ")
        try:
            root, it = bisection_method(f_sym, a, b); print(f"Approx root: {root} (iter: {it})")
        except Exception as e:
            print("Error:", e)
    elif choice == 'b':
        print("Newton-Raphson (requires sympy)")
        if not sympy_available: print("SymPy not available."); return
        expr = input("f(x) = ")
        x = sp.symbols('x');
        f_sym = sp.sympify(expr);
        df_sym = sp.diff(f_sym, x)
        f = sp.lambdify(x, f_sym, 'math');
        df = sp.lambdify(x, df_sym, 'math')
        x0 = read_float("Initial x0: ")
        try:
            root, it = newton_method(f, df, x0); print(f"Root: {root} (iter: {it})")
        except Exception as e:
            print("Error:", e)
    elif choice == 'c':
        print("Numerical integration (requires sympy)")
        if not sympy_available: print("SymPy not available."); return
        expr = input("f(x) = ")
        x = sp.symbols('x');
        f_sym = sp.lambdify(x, sp.sympify(expr), 'math')
        a = read_float("a: ");
        b = read_float("b: ")
        if None in (a, b): return
        print("1) Trapezoidal  2) Simpson");
        sub_choice = input("Choose: ").strip()
        n = read_int("Number of subdivisions (e.g., 1000): ") or 1000
        if sub_choice == '1':
            print("Result (Trapezoidal):", trapezoidal_integral(f_sym, a, b, n))
        else:
            print("Result (Simpson):", simpson_integral(f_sym, a, b, n))
    elif choice == 'd':
        print("Numerical derivative (requires sympy)")
        if not sympy_available: print("SymPy not available."); return
        expr = input("f(x) = ")
        x = sp.symbols('x');
        f_sym = sp.lambdify(x, sp.sympify(expr), 'math')
        x_val = read_float("x: ");
        h_val = read_float("h (step, e.g., 1e-6): ") or 1e-6
        print("Derivative at", x_val, "is approx:", numerical_derivative(f_sym, x_val, h_val))
    else:
        print("Invalid option.")


def linear_algebra_menu():
    print("\n-- LINEAR ALGEBRA --")
    print(
        "1. Matrix Addition\n2. Matrix Multiplication\n3. Determinant\n4. Inverse\n5. Matrix Transpose\n6. Eigenvalues/Eigenvectors\n7. Solve Linear System (Ax=b)\n8. LU Decomposition\n9. QR Decomposition\n10. Covariance Matrix\n11. Vandermonde Matrix\n0. Back")
    choice = input("Choose: ").strip().lower()
    if choice == '1':
        print("Matrix A:");
        A = matrix_from_input()
        print("Matrix B:");
        B = matrix_from_input()
        if A is None or B is None: return
        try:
            print("Result:\n", add_matrices(A, B))
        except Exception as e:
            print(f"Error: {e}")
    elif choice == '2':
        print("Matrix A:");
        A = matrix_from_input()
        print("Matrix B:");
        B = matrix_from_input()
        if A is None or B is None: return
        try:
            print("Result:\n", multiply_matrices(A, B))
        except Exception as e:
            print(f"Error: {e}")
    elif choice == '3':
        A = matrix_from_input()
        if A is None: return
        try:
            print("Determinant =", determinant(A))
        except Exception as e:
            print("Error:", e)
    elif choice == '4':
        A = matrix_from_input()
        if A is None: return
        try:
            print("Inverse:\n", inverse_matrix(A))
        except Exception as e:
            print("Error:", e)
    elif choice == '5':
        A = matrix_from_input()
        if A is None: return
        try:
            print("Transpose:\n", transpose_matrix(A))
        except Exception as e:
            print(f"Error: {e}")
    elif choice == '6':
        A = matrix_from_input()
        if A is None: return
        eigen_values_vectors(A)
    elif choice == '7':
        A = matrix_from_input();
        b = vector_from_input()
        if A is None or b is None: return
        solve_linear_system(A, b)
    elif choice == '8':
        A = matrix_from_input()
        if A is None: return
        try:
            L, U = lu_decomposition(A)
        except Exception as e:
            print(f"Error: {e}"); return
        print("L Matrix:\n", L)
        print("\nU Matrix:\n", U)
    elif choice == '9':
        A = matrix_from_input()
        if A is None: return
        try:
            Q, R = qr_decomposition(A)
        except Exception as e:
            print(f"Error: {e}"); return
        print("Q Matrix:\n", Q)
        print("\nR Matrix:\n", R)
    elif choice == '10':
        print("Data matrix (samples per row):")
        data = matrix_from_input()
        if data is None: return
        try:
            cov = covariance_matrix(data)
            print("Covariance Matrix:\n", cov)
        except Exception as e:
            print(f"Error: {e}")
    elif choice == '11':
        print("Input vector for Vandermonde matrix:")
        v = vector_from_input()
        if v is None: return
        n = read_int("Matrix order (optional, default = len(v)): ")
        try:
            vand = vandermonde_matrix(v, n)
            print("Vandermonde Matrix:\n", vand)
        except Exception as e:
            print(f"Error: {e}")
    elif choice == '0':
        return
    else:
        print("Invalid option.")


def physics_menu():
    print("\n-- PHYSICS AND ENGINEERING --")
    print(
        "1) Uniform Motion\n2) Uniform Accelerated Motion\n3) Oblique Launch\n4) Ohm's Law\n5) Kinetic Energy\n6) Ideal Gas Law\n7) Universal Gravitation\n8) Solids Geometry\n9) Distance between Points\n10) Vector Operations\n11) Vector Rotation\n0) Back")
    choice = input("Choose: ").strip()
    if choice == '1':
        uniform_motion()
    elif choice == '2':
        uniform_accelerated_motion()
    elif choice == '3':
        oblique_launch()
    elif choice == '4':
        ohm_law()
    elif choice == '5':
        kinetic_energy()
    elif choice == '6':
        ideal_gas_law()
    elif choice == '7':
        universal_gravitation()
    elif choice == '8':
        geometry_solids()
    elif choice == '9':
        distance_between_points()
    elif choice == '10':
        vector_operations()
    elif choice == '11':
        vector_rotation()
    elif choice == '0':
        return
    else:
        print("Invalid option.")


def conversions_menu():
    print("\n-- SCIENTIFIC AND BASE CONVERSIONS --")
    print("1) Temperature\n2) Speed\n3) Roman <-> Arabic\n4) Base64 Encode/Decode\n0) Back")
    choice = input("Choose: ").strip()
    if choice == '1':
        temperature_conversion()
    elif choice == '2':
        speed_conversion()
    elif choice == '3':
        print("a) Roman -> Arabic\nb) Arabic -> Roman")
        sub_choice = input("Choose: ").strip().lower()
        if sub_choice == 'a':
            roman_to_arabic()
        elif sub_choice == 'b':
            arabic_to_roman()
        else:
            print("Invalid option.")
    elif choice == '4':
        base64_converter()
    elif choice == '0':
        return
    else:
        print("Invalid option.")


def stats_menu():
    print("\n-- STATISTICS AND DATA ANALYSIS --")
    print(
        "1) Basic Statistics (Mean, Median, Std Dev)\n2) Normal Distribution (PDF/CDF)\n3) Binomial PMF\n4) Poisson PMF\n5) Simple Linear Regression\n6) Correlation Coefficient\n0) Back")
    choice = input("Choose: ").strip()
    if choice == '1':
        basic_statistics()
    elif choice == '2':
        x = read_float("x: ");
        mu = read_float("mu (default 0): ") or 0;
        sigma = read_float("sigma (default 1): ") or 1
        print("PDF =", normal_pdf(x, mu, sigma));
        print("CDF =", normal_cdf(x, mu, sigma))
    elif choice == '3':
        n = read_int("n: ");
        p = read_float("p (0-1): ");
        k = read_int("k: ")
        if None in (n, p, k): return
        print("Binomial PMF =", binomial_pmf(k, n, p))
    elif choice == '4':
        lam = read_float("lambda: ");
        k = read_int("k: ")
        if None in (lam, k): return
        print("Poisson PMF =", poisson_pmf(k, lam))
    elif choice == '5':
        simple_linear_regression()
    elif choice == '6':
        correlation_coefficient()
    else:
        print("Invalid option.")


def utilities_menu():
    print("\n-- UTILITIES --")
    print(
        "1) Text -> ASCII\n2) Base converter\n3) Hash (MD5/SHA1/SHA256)\n4) Caesar Cipher\n5) Vigenère Cipher\n6) Stopwatch\n7) Text Compression (RLE)\n8) Data Structures (Stack, Queue)\n9) Dice Simulation")
    choice = input("Choose: ").strip()
    if choice == '1':
        text_to_ascii()
    elif choice == '2':
        base_converter()
    elif choice == '3':
        generate_hash()
    elif choice == '4':
        caesar_cipher()
    elif choice == '5':
        vigenere_cipher()
    elif choice == '6':
        stopwatch()
    elif choice == '7':
        rle_converter()
    elif choice == '8':
        data_structures_menu()
    elif choice == '9':
        dice_simulation()
    else:
        print("Invalid option.")


def ai_optimization_menu():
    print("\n--- AI AND OPTIMIZATION ---")
    print("1) Fuzzy Logic Tip\n2) Optimization (Gradient Descent)\n0) Back")
    choice = input("Choose: ").strip()
    if choice == '1':
        fuzzy_tipping()
    elif choice == '2':
        gradient_descent()
    elif choice == '0':
        return
    else:
        print("Invalid option.")


def extreme_menu():
    while True:
        print("\n=== ADVANCED MODULE ===")
        print("1. Advanced Mathematics (Numbers, Polynomials)")
        print("2. Linear Algebra (Matrices)")
        print("3. Physics and Engineering")
        print("4. Scientific and Base Conversions")
        print("5. Statistics and Data Analysis")
        print("6. Utilities (Hash, Crypto, Stopwatch...)")
        print("7. AI and Optimization")
        print("8. Fractals: Mandelbrot ASCII")
        print("9. Scientific constants")
        print("0. Back")
        choice = input("Choose: ").strip()
        if choice == '1':
            advanced_math_menu()
        elif choice == '2':
            linear_algebra_menu()
        elif choice == '3':
            physics_menu()
        elif choice == '4':
            conversions_menu()
        elif choice == '5':
            stats_menu()
        elif choice == '6':
            utilities_menu()
        elif choice == '7':
            ai_optimization_menu()
        elif choice == '8':
            try:
                mandelbrot_ascii(width=80, height=32, max_iter=120)
            except Exception as e:
                print("Error generating Mandelbrot:", e)
        elif choice == '9':
            for k, v in CONSTANTS.items(): print(f"{k} = {v}")
        elif choice == '0':
            break
        else:
            print("Invalid option.")


# Original functions
def basic_operations():
    print("\n-- Basic Operations --")
    print("a) +  b) -  c) * d) /  e) ^  f) sqrt  g) n-root")
    choice = input("Choose: ").strip().lower()
    try:
        if choice in ['a', 'b', 'c', 'd', 'e']:
            x = float(input("x: "));
            y = float(input("y: "))
            if choice == 'a': print(x + y)
            if choice == 'b': print(x - y)
            if choice == 'c': print(x * y)
            if choice == 'd': print(x / y) if y != 0 else print("Error: division by zero")
            if choice == 'e': print(x ** y)
        elif choice == 'f':
            x = float(input("x: "));
            print(cmath.sqrt(x) if x < 0 else math.sqrt(x))
        elif choice == 'g':
            x = float(input("x: "));
            n = float(input("n: "))
            if n == 0: print("Invalid index"); return
            is_int_n = abs(n - round(n)) < 1e-12
            if x < 0 and is_int_n and int(round(n)) % 2 == 0:
                print("Even root of negative: complex", complex(x) ** (1.0 / n))
            else:
                print((complex(x) if x < 0 else x) ** (1.0 / n))
        else:
            print("Invalid option")
    except Exception as e:
        print("Error:", e)


def trigonometric_functions():
    print("\n-- Trigonometry --")
    mode = input("Degrees (d) or radians (r)? ").strip().lower()
    choice = input("a) sin b) cos c) tan d) asin e) acos f) atan g) atan2\nChoose: ").strip().lower()
    try:
        if choice in ['a', 'b', 'c']:
            x = float(input("Angle: "));
            x = math.radians(x) if mode == 'd' else x
            if choice == 'a': print(math.sin(x))
            if choice == 'b': print(math.cos(x))
            if choice == 'c':
                cosx = math.cos(x)
                if abs(cosx) < 1e-12: print("Warning: point near asymptote")
                print(math.tan(x))
        elif choice in ['d', 'e', 'f']:
            x = float(input("Value (ratio): "))
            if choice in ['d', 'e'] and abs(x) > 1: print("Out of domain"); return
            r = math.asin(x) if choice == 'd' else math.acos(x) if choice == 'e' else math.atan(x)
            r = math.degrees(r) if mode == 'd' else r
            print(r)
        elif choice == 'g':
            y = float(input("y: "));
            x = float(input("x: "))
            r = math.atan2(y, x);
            r = math.degrees(r) if mode == 'd' else r
            print(r)
    except Exception as e:
        print("Error:", e)


def hyperbolic_functions():
    print("\n-- Hyperbolics --")
    choice = input("a)sinh b)cosh c)tanh d)asinh e)acosh f)atanh\nChoose: ").strip().lower()
    try:
        x = float(input("x: "))
        if choice == 'a':
            print(math.sinh(x))
        elif choice == 'b':
            print(math.cosh(x))
        elif choice == 'c':
            print(math.tanh(x))
        elif choice == 'd':
            print(math.asinh(x))
        elif choice == 'e':
            if x < 1: print("Out of domain"); return
            print(math.acosh(x))
        elif choice == 'f':
            if x <= -1 or x >= 1: print("Out of domain"); return
            print(math.atanh(x))
    except Exception as e:
        print("Error:", e)


def logs_and_exponentials():
    print("\n-- Log/Exp --")
    choice = input("a)log10 b)ln c)exp d)log base e)dec->scientific f)scientific->dec\nChoose: ").strip().lower()
    try:
        if choice == 'a':
            x = float(input("x (>0): "));
            print(math.log10(x))
        elif choice == 'b':
            x = float(input("x (>0): "));
            print(math.log(x))
        elif choice == 'c':
            x = float(input("exponent: "));
            print(math.exp(x))
        elif choice == 'd':
            x = float(input("x>0: "));
            base = float(input("base>0 !=1: "));
            print(math.log(x, base))
        elif choice == 'e':
            s = input("decimal: ");
            print(format(float(s), '.6e'))
        elif choice == 'f':
            s = input("scientific notation (e.g., 1.23e4): ");
            print(float(s))
    except Exception as e:
        print("Error:", e)


def factorial_permutation_combination():
    print("\n-- Factorial/Perm/Comb/Gamma --")
    choice = input("a)! b)P(n,r) c)C(n,r) d)Gamma\nChoose: ").strip().lower()
    try:
        if choice == 'a':
            n = int(input("n (>=0): "));
            print(math.factorial(n))
        elif choice in ['b', 'c']:
            n = int(input("n: "));
            r = int(input("r: "))
            if r > n: print("r>n"); return
            if choice == 'b':
                print(math.factorial(n) // math.factorial(n - r))
            else:
                print(math.comb(n, r))
        elif choice == 'd':
            x = float(input("x: "));
            print(math.gamma(x))
    except Exception as e:
        print("Error:", e)


def basic_conversions():
    print("\n-- Number Conversions --")
    print("a)deg->rad b)rad->deg c)dec->bin d)bin->dec e)dec->oct f)oct->dec g)dec->hex h)hex->dec")
    choice = input("Choose: ").strip().lower()
    try:
        if choice == 'a':
            g = float(input("degrees: "));
            print(math.radians(g))
        elif choice == 'b':
            r = float(input("radians: "));
            print(math.degrees(r))
        elif choice == 'c':
            dec = int(input("decimal: "));
            print(bin(dec)[2:])
        elif choice == 'd':
            b = input("binary: ");
            print(int(b, 2))
        elif choice == 'e':
            dec = int(input("decimal: "));
            print(oct(dec)[2:])
        elif choice == 'f':
            o = input("octal: ");
            print(int(o, 8))
        elif choice == 'g':
            dec = int(input("decimal: "));
            print(hex(dec)[2:])
        elif choice == 'h':
            h = input("hex: ");
            print(int(h, 16))
    except Exception as e:
        print("Error:", e)


def basic_statistics():
    print("\n-- Basic Statistics --")
    print("Enter numbers separated by space:")
    try:
        data = list(map(float, input().split()))
        if not data: print("Empty list"); return
        print("Mean:", statistics.mean(data))
        print("Median:", statistics.median(data))
        if len(data) > 1:
            print("Variance:", statistics.variance(data))
            print("Std Dev:", statistics.stdev(data))
    except Exception as e:
        print("Error:", e)


def complex_numbers():
    print("\n-- Complex Numbers --")
    try:
        z1 = complex(input("z1 (e.g., 3+4j): ").strip());
        z2 = complex(input("z2: ").strip())
        print("Sum:", z1 + z2);
        print("Subtraction:", z1 - z2);
        print("Multiplication:", z1 * z2)
        try:
            print("Division:", z1 / z2)
        except ZeroDivisionError:
            print("Division by zero!")
        print("Modulus:", abs(z1));
        print("Argument (rad):", cmath.phase(z1))
        print("Conjugate:", z1.conjugate())
    except Exception as e:
        print("Error:", e)


def additional_functions():
    print("\n-- Additional Functions --")
    print("a)Is prime? b)Next n primes c)LCM d)GCD e)Bit ops f)Prime factors")
    choice = input("Choose: ").strip().lower()

    def is_prime(num):
        if num <= 1: return False
        if num <= 3: return True
        if num % 2 == 0 or num % 3 == 0: return False
        i = 5
        while i * i <= num:
            if num % i == 0 or num % (i + 2) == 0: return False
            i += 6
        return True

    if choice == 'a':
        n = read_int("n: ");
        print(is_prime(n))
    elif choice == 'b':
        n = read_int("How many primes: ");
        count = 0;
        num = 2;
        primes = []
        while count < n:
            if is_prime(num): primes.append(num); count += 1
            num += 1
        print(primes)
    elif choice == 'c':
        a = read_int("a: ");
        b = read_int("b: ")
        if None not in (a, b): print("LCM:", abs(a * b) // gcd(a, b) if a != 0 and b != 0 else 0)
    elif choice == 'd':
        a = read_int("a: ");
        b = read_int("b: ")
        if None not in (a, b): print("GCD:", gcd(a, b))
    elif choice == 'e':
        x = read_int("x: ");
        y = read_int("y: ")
        if None not in (x, y): print("AND:", x & y, "OR:", x | y, "XOR:", x ^ y)
    elif choice == 'f':
        n = read_int("n: ")
        if n is not None: print("Factorization:", factorization(n))
    else:
        print("Invalid option.")


def ascii_plot(xs, ys, title="Plot", xlabel="x", ylabel="y", width=None, height=20):
    pts = [(x, y) for x, y in zip(xs, ys) if
           y is not None and not (isinstance(y, float) and (math.isnan(y) or math.isinf(y)))]
    if not pts: print("No valid points."); return
    xs_valid = [p[0] for p in pts];
    ys_valid = [p[1] for p in pts]
    xmin, xmax = min(xs_valid), max(xs_valid);
    ymin, ymax = min(ys_valid), max(ys_valid)
    if abs(xmax - xmin) < 1e-12: xmin -= 1; xmax += 1
    if abs(ymax - ymin) < 1e-12: ymin -= 1; ymax += 1
    try:
        cols = shutil.get_terminal_size().columns
    except Exception:
        cols = 80
    if width is None: width = min(80, max(40, cols - 10))
    height = max(10, int(height));
    grid = [[" " for _ in range(width)] for _ in range(height)]

    def map_point(x, y):
        col = int((x - xmin) / (xmax - xmin) * (width - 1))
        row = int((ymax - y) / (ymax - ymin) * (height - 1))
        return max(0, min(height - 1, row)), max(0, min(width - 1, col))

    for x, y in pts:
        if isinstance(y, float) and (math.isinf(y) or math.isnan(y) or abs(y) > 1e6): continue
        r, c = map_point(x, y);
        grid[r][c] = "*"
    if xmin <= 0 <= xmax:
        x0_col = int((0 - xmin) / (xmax - xmin) * (width - 1))
        for r in range(height):
            if grid[r][x0_col] == " ": grid[r][x0_col] = "|"
    if ymin <= 0 <= ymax:
        y0_row = int((ymax - 0) / (ymax - ymin) * (height - 1))
        for c in range(width):
            if grid[y0_row][c] == " ": grid[y0_row][c] = "-"
    print(f"\n{title}")
    for row in grid: print("".join(row))
    print(f"{xlabel}: [{xmin:.4g}, {xmax:.4g}]    {ylabel}: [{ymin:.4g}, {ymax:.4g}]")
    print("('*' points, '|' y-axis, '-' x-axis)")


def plots():
    print("\n-- Plots --")
    print("a) sin b) cos c) tan d) exp e) ln f) polynomial")
    option = input("Choose: ").strip().lower()
    try:
        x_start = float(input("x start: "));
        x_end = float(input("x end: "))
        steps = int(input("points (e.g., 200): "))
        if steps <= 1 or x_end <= x_start: print("Invalid interval."); return
        xs = [x_start + i * (x_end - x_start) / (steps - 1) for i in range(steps)]
        ys = [];
        title = ""
        if option == 'a':
            ys = [math.sin(x) for x in xs]; title = "sin(x)"
        elif option == 'b':
            ys = [math.cos(x) for x in xs]; title = "cos(x)"
        elif option == 'c':
            ys_tmp = []
            for val in xs:
                try:
                    y = math.tan(val)
                except ValueError:
                    y = None
                ys_tmp.append(y if abs(y) < 1e6 else None)
            ys = ys_tmp;
            title = "tan(x)"
        elif option == 'd':
            ys = [];
            for x in xs:
                try:
                    y = math.exp(x)
                except OverflowError:
                    y = None
                ys.append(y if y is not None and abs(y) < 1e12 else None)
            title = "exp(x)"
        elif option == 'e':
            if any(x <= 0 for x in xs): print("ln(x): interval must be > 0"); return
            ys = [math.log(x) for x in xs];
            title = "ln(x)"
        elif option == 'f':
            a = float(input("a: "));
            b = float(input("b: "));
            c = float(input("c: "))
            ys = [a * x ** 2 + b * x + c for x in xs];
            title = f"{a}x^2+{b}x+{c}"
        else:
            print("Invalid"); return
        if plotext_available:
            try:
                plt.clear_figure();
                plt.title(title)
                seg_x, seg_y = [], []
                for x, y in zip(xs, ys):
                    if y is None or math.isnan(y) or math.isinf(y):
                        if seg_x: plt.plot(seg_x, seg_y); seg_x, seg_y = [], []
                    else:
                        seg_x.append(x); seg_y.append(y)
                if seg_x: plt.plot(seg_x, seg_y)
                plt.show()
                return
            except Exception:
                print("plotext failed; fallback to ASCII.")
        ys_for_ascii = [y if y is not None else float('nan') for y in ys]
        ys_mapped = [None if (isinstance(y, float) and (math.isnan(y) or math.isinf(y))) else y for y in ys_for_ascii]
        ascii_plot(xs, ys_mapped, title=title)
    except Exception as e:
        print("Error:", e)


# --------------------
# Main program
# --------------------

def main_menu():
    print("\n=== EXTREME SCIENTIFIC CALCULATOR ===")
    print("1. Basic Operations")
    print("2. Trigonometric Functions")
    print("3. Hyperbolic Functions")
    print("4. Logarithms, Exponentials, and Powers")
    print("5. Factorial, Permutation, Combination, and Gamma")
    print("6. Number Conversions")
    print("7. Basic Statistics")
    print("8. Complex Numbers")
    print("9. Additional Functions")
    print("10. Console Plots (ASCII or plotext)")
    print("11. Advanced Module (numerical methods, linear algebra, physics...)")
    print("0. Exit")
    return input("Choose an option: ").strip()


def main():
    while True:
        choice = main_menu()
        if choice == '1':
            basic_operations()
        elif choice == '2':
            trigonometric_functions()
        elif choice == '3':
            hyperbolic_functions()
        elif choice == '4':
            logs_and_exponentials()
        elif choice == '5':
            factorial_permutation_combination()
        elif choice == '6':
            basic_conversions()
        elif choice == '7':
            basic_statistics()
        elif choice == '8':
            complex_numbers()
        elif choice == '9':
            additional_functions()
        elif choice == '10':
            plots()
        elif choice == '11':
            extreme_menu()
        elif choice == '0':
            print("Exiting. Goodbye!"); sys.exit(0)
        else:
            print("Invalid option. Try again.")


if __name__ == "__main__":
    main()