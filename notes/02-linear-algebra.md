# 02 — Linear Algebra for Computing

## Vectors and inner products

For vectors `x,y` in R^n, the Euclidean inner product is `x^T y`. The induced norm is `||x||_2 = sqrt(x^T x)`.

## Matrices

A matrix represents a linear map. Composition corresponds to matrix multiplication, which is associative but generally not commutative.

## Eigenvalues

A non-zero vector `v` is an eigenvector of `A` with eigenvalue `lambda` when `Av = lambda v`. Eigenstructure is central to spectral graph methods, dimensionality reduction, stability analysis, and iterative algorithms.

## Numerical connection

In ML, gradient descent repeatedly applies vector and matrix operations. In distributed systems, linear algebra also appears in coding schemes and randomized sketching.
