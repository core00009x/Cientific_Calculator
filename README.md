# Extreme Scientific Calculator

This Python script is a robust and multifaceted calculator designed to go beyond basic arithmetic. It offers an extensive array of functionalities, including advanced mathematics, linear algebra, physics, statistics, scientific conversions, and even programming and cryptography tools. The project is organized to be user-friendly, presenting a menu-driven interface that guides the user through different categories of calculations.

## Requirements

The script is built to function with standard Python libraries. However, to unlock its full potential and access the more advanced features, you will need to install some optional dependencies.

### Optional Dependencies

For a complete experience, it is highly recommended to install the following libraries:

  -> **NumPy**: This is essential for all linear algebra functions, such as matrix operations, eigenvalue/eigenvector calculations, and solving systems of linear equations.
  -> **SymPy**: This library is used for symbolic mathematics features, if required.
  -> **plotext**: This is utilized for plotting graphs directly within the terminal, offering a visual representation of certain mathematical functions.

You can easily install these dependencies using `pip`:

bash
  > pip install numpy sympy plotext

# How to Use
To use the calculator, simply save the code as script.py. From your command-line interface, navigate to the directory where you saved the file and execute it with the following command:

Bash
  > python script.py
Upon execution, the script will display a main menu of options. From there, you can follow the prompts to select a calculation category and then the specific function you wish to perform. The interactive interface makes it easy to input values and receive results.

# Functionalities
The calculator's features are organized into several distinct categories to make navigation intuitive.

# 1. Advanced Mathematics
This section is dedicated to complex mathematical operations. It includes numerical calculus functionalities like calculating derivatives and integrals (using the Trapezoidal and Simpson's methods), and finding roots of equations with the Bisection and Newton-Raphson methods. For arithmetic, it provides tools for prime factorization, greatest common divisor (GCD), Euler's Totient function, modular exponentiation, and more. It also includes functions for sequences, such as calculating the n-th Fibonacci number.

# 2. Linear Algebra (Requires NumPy)
This is a powerful suite of tools for linear algebra. You can perform standard matrix operations like addition, multiplication, transposition, and find determinants and inverses. The script can also solve systems of linear equations, calculate eigenvalues and eigenvectors, and perform matrix decompositions such as LU and QR. Additionally, it offers features for creating special matrices like the Covariance and Vandermonde matrices.

# 3. Geometry and Vector Calculus
This category covers calculations related to both geometry and vectors. For geometry, you can compute the surface area and volume of various solids including spheres, cubes, cylinders, and cones. It also provides tools to find the distance between two points in 2D and 3D space. The vector calculus section allows for dot and cross products, vector projection in 3D, and vector rotation in both 2D and 3D.

# 4. Physics and Engineering
This section applies mathematical principles to practical physics and engineering problems. It includes calculations for motion, covering Uniform Motion (MU), Uniformly Accelerated Motion (MUA), and Oblique Launch. For electricity, you can apply Ohm's Law, while in thermodynamics, the Ideal Gas Law is available. Mechanics problems are also addressed with functions for calculating kinetic energy and applying the Universal Law of Gravitation.

# 5. Scientific Conversions
This category is a versatile tool for various scientific conversions. It includes temperature conversions between Celsius, Fahrenheit, and Kelvin, and speed conversions between m/s, km/h, and mph. You can also convert numbers between Arabic and Roman numerals, and between different number bases (2-36). For data encoding, it offers text-to-ASCII conversion and Base64 encoding/decoding.

# 6. Statistics and Probability
This section provides a set of tools for statistical analysis. You can work with probability distributions, specifically calculating the Probability Density Function (PDF) and Cumulative Distribution Function (CDF) for the normal distribution. It also includes functionalities for regression and correlation, such as calculating the Pearson correlation coefficient and performing simple linear regression.

# 7. Programming Utilities
Finally, this category is designed with programmers in mind. It provides basic data structure implementations like Stacks and Queues. For security and data handling, it includes various cryptographic tools such as the Caesar and Vigenère ciphers, Run-Length Encoding (RLE), and a hash generator for MD5, SHA1, and SHA256.
