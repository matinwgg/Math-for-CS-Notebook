# Mathematics for AI and Computer Science — Deep Study Guide

This chapter expands the notebook from a topic list into a rigorous study reference. The emphasis is not memorizing formulas: it is understanding what the objects mean, why the formulas are true, what assumptions they require, and how the mathematics appears in AI, security, distributed systems, and algorithms.

## 0. Prerequisites and notation

Assume familiarity with algebra, functions, logarithms, basic trigonometry, and single-variable calculus. For graduate-level work, become comfortable with summation notation, proof by induction, basic set notation, and elementary probability.

Notation used throughout:
- Scalars: `a, b, x ∈ R`.
- Vectors: `x ∈ R^n`; transpose `x^T`.
- Matrices: `A ∈ R^{m×n}`.
- Identity: `I_n`.
- Euclidean norm: `||x||_2 = sqrt(x^T x)`.
- Expectation: `E[X]`; variance: `Var(X)`; covariance: `Cov(X,Y)`.
- Probability of event A: `P(A)`; conditional probability: `P(A|B)`.
- Gradient of scalar f: `∇f`; Hessian: `∇²f`.
- Asymptotic notation: `O`, `Ω`, `Θ`.

A useful meta-principle is that AI repeatedly turns a computational problem into one of four mathematical forms: a linear-algebra operation, an estimation problem, an optimization problem, or a probabilistic inference problem. Discrete mathematics and algorithms determine how those computations can be represented and executed efficiently.

# 1. Linear Algebra

Linear algebra is the language of representations and transformations. In ML, data points, parameters, activations, embeddings, gradients, covariance matrices, and attention operations are all naturally expressed using vectors and matrices.

## 1.1 Vectors

A vector in `R^n` is an ordered n-tuple. It can represent a point, displacement, feature vector, parameter vector, probability vector, or direction.

For `x=(x_1,...,x_n)` and `y=(y_1,...,y_n)`, vector addition and scalar multiplication are componentwise:

`x+y=(x_1+y_1,...,x_n+y_n)` and `αx=(αx_1,...,αx_n)`.

The vector-space axioms matter because they guarantee that linear combinations are well-defined. A linear combination is `Σ_i α_i v_i`. Span, linear independence, basis, and dimension answer whether a collection of features contains redundant information and how many degrees of freedom a representation has.

### AI connection
A dataset with d numerical features is often represented as a matrix `X ∈ R^{n×d}`. Each row can be an example and each column a feature. A learned embedding is a vector in a high-dimensional representation space. Similarity search often uses dot products or cosine similarity.

## 1.2 Matrices

A matrix is a rectangular array representing either data or a linear map. For `A ∈ R^{m×n}`, multiplication by x maps `R^n → R^m`.

Matrix multiplication is defined by:

`(AB)_{ij}=Σ_k A_{ik}B_{kj}`.

The order matters: generally `AB ≠ BA`. Matrix multiplication is associative and distributive but not commutative.

A matrix is invertible exactly when its linear transformation has an inverse. Equivalent conditions for a square matrix include nonzero determinant, full rank, trivial nullspace, and nonzero eigenvalues.

### Rank and nullspace
The rank is the dimension of the column space. The nullspace is `{x: Ax=0}`. The rank-nullity theorem states:

`rank(A) + nullity(A) = number of columns of A`.

This theorem is fundamental for understanding underdetermined models, identifiability, compression, and degeneracy.

## 1.3 Tensors

A tensor generalizes vectors and matrices to multiple indices. A scalar is order 0, vector order 1, matrix order 2, and an image can be represented as a rank-3 tensor such as height × width × channels.

In deep learning, tensor operations are usually compositions of linear maps, elementwise nonlinearities, reductions, and reshaping/permutation operations. Always distinguish tensor **rank/order** from matrix rank; they are different concepts.

## 1.4 Dot products and geometry

The Euclidean inner product is:

`x·y = x^T y = Σ_i x_i y_i`.

The Cauchy–Schwarz inequality says:

`|x·y| ≤ ||x||_2 ||y||_2`.

It leads to the angle formula:

`cos θ = (x·y)/(||x|| ||y||)`.

Cosine similarity is therefore a geometric normalization of the dot product, not merely an arbitrary ML heuristic.

Orthogonality means `x·y=0`. Orthogonal vectors have zero inner-product interaction and form the basis of least-squares projection and orthogonal decompositions.

## 1.5 Norms

Common norms include:
- `L1`: `||x||_1=Σ|x_i|`.
- `L2`: `||x||_2=sqrt(Σx_i²)`.
- `L∞`: `||x||_∞=max_i |x_i|`.
- Frobenius norm: `||A||_F=sqrt(Σ_{ij}A_{ij}²)`.

A norm must satisfy positivity, homogeneity, and the triangle inequality.

In ML, L1 regularization promotes sparse parameters while L2 regularization penalizes large parameter magnitude. In adversarial ML, the chosen norm defines the threat model: an `L∞` perturbation limits every coordinate, whereas an `L2` perturbation limits total Euclidean energy.

## 1.6 Projections and least squares

Projection of x onto a nonzero vector v is:

`proj_v(x) = ((x·v)/(v·v))v`.

For a matrix A, least squares minimizes:

`min_x ||Ax-b||_2²`.

Differentiating gives the normal equations:

`A^T A x = A^T b`.

The geometry says the residual `r=b-Ax` is orthogonal to every column of A. In numerical software, explicitly forming `A^T A` can worsen conditioning; QR or SVD methods are usually more stable.

## 1.7 Eigenvalues and eigenvectors

A nonzero vector v is an eigenvector of A if:

`Av=λv`.

The scalar λ is its eigenvalue. It represents a direction preserved by the transformation, with scaling λ.

Eigenvalues satisfy the characteristic equation:

`det(A-λI)=0`.

For symmetric real matrices, eigenvalues are real and eigenvectors can be chosen orthonormal. This spectral theorem is central to PCA, covariance analysis, graph Laplacians, and stability analysis.

### PCA
Given centered data X, the covariance matrix is:

`Σ = (1/(n-1)) X^T X`.

The first principal component is the unit vector maximizing projected variance:

`max_{||v||=1} v^TΣv`.

Using a Lagrange multiplier gives `Σv=λv`; therefore the principal directions are eigenvectors and the explained variance is associated with eigenvalues.

## 1.8 SVD

Every real matrix has a singular value decomposition:

`A=UΣV^T`.

The singular values are nonnegative and ordered. SVD works for rectangular and rank-deficient matrices and gives the best low-rank approximation in important matrix norms.

The Eckart–Young result implies that truncating SVD after k singular values gives the best rank-k approximation under the standard spectral/Frobenius settings.

Applications include PCA, compression, recommender systems, latent semantic analysis, denoising, pseudoinverses, and numerical least squares.

## 1.9 Matrix decompositions

Important decompositions:
- LU: solves linear systems efficiently and supports determinant-related calculations.
- QR: stable least-squares and orthogonalization.
- Cholesky: `A=LL^T` for symmetric positive-definite A.
- Eigendecomposition: `A=QΛQ^{-1}` when diagonalizable.
- SVD: `A=UΣV^T`, always available over real matrices.

Understand when each decomposition exists, its numerical stability, and its computational cost rather than treating them as interchangeable.

# 2. Calculus for AI

Calculus describes local change. Optimization uses derivatives to decide how parameters should move to reduce an objective.

## 2.1 Derivatives

For scalar f(x), the derivative is:

`f'(x)=lim_{h→0}(f(x+h)-f(x))/h`.

It is the instantaneous rate of change and the slope of the tangent line.

The first-order approximation is:

`f(x+Δx) ≈ f(x)+f'(x)Δx`.

The second-order approximation adds curvature:

`f(x+Δx) ≈ f(x)+f'(x)Δx + 1/2 f''(x)(Δx)^2`.

## 2.2 Partial derivatives and gradients

For `f:R^n→R`, the partial derivative `∂f/∂x_i` measures sensitivity while holding other coordinates fixed.

The gradient is:

`∇f=[∂f/∂x_1,...,∂f/∂x_n]^T`.

The gradient points in the direction of steepest local increase under the Euclidean metric. Consequently, `-∇f` is the steepest local descent direction.

For a vector-valued function `f:R^n→R^m`, the Jacobian is the m×n matrix of partial derivatives.

## 2.3 Chain rule

For `y=f(g(x))`:

`dy/dx = f'(g(x))g'(x)`.

For multivariable functions, Jacobians multiply:

`J_{f∘g}(x)=J_f(g(x))J_g(x)`.

Backpropagation is repeated application of this rule through a computational graph. The apparent complexity of neural-network differentiation comes from applying a simple local rule across many composed operations.

## 2.4 Hessians and curvature

The Hessian is:

`H=∇²f`.

Its entries are second partial derivatives. Near a point:

`f(x+Δ)≈f(x)+∇f(x)^TΔ+1/2 Δ^THΔ`.

At a stationary point, positive-definite Hessian implies a strict local minimum; negative-definite implies a strict local maximum; an indefinite Hessian indicates saddle behavior.

Eigenvalues of the Hessian quantify curvature directions. Large condition numbers indicate elongated optimization landscapes and can make gradient methods difficult.

## 2.5 Optimization intuition

A differentiable local minimum must satisfy `∇f(x*)=0`, but the converse is false: stationary points can be maxima or saddles.

Gradient descent uses:

`x_{t+1}=x_t-η∇f(x_t)`.

For a quadratic `f(x)=1/2 x^TQx-b^Tx`, convergence depends on the eigenvalues of Q and the learning rate. For positive-definite Q, a sufficiently small step size produces convergence; poor conditioning causes slow progress along low-curvature directions.

# 3. Probability and Statistics

Probability provides a mathematical language for uncertainty. Statistics uses observed data to infer unknown quantities.

## 3.1 Random variables and distributions

A random variable maps outcomes to numerical values. A discrete variable has a probability mass function; a continuous variable has a density.

Core distributions include Bernoulli, Binomial, Categorical, Geometric, Poisson, Uniform, Gaussian, Exponential, Gamma, Beta, and Multivariate Gaussian.

The Gaussian density is:

`p(x)=1/(σ√(2π)) exp(-(x-μ)^2/(2σ²))`.

The Central Limit Theorem explains why normalized sums of many weakly dependent contributions often approach a Gaussian distribution under appropriate assumptions.

## 3.2 Expectation and variance

Expectation is the probability-weighted average:

`E[X]=Σ_x xp(x)` for discrete X and `E[X]=∫xp(x)dx` for continuous X.

Variance is:

`Var(X)=E[(X-E[X])²]=E[X²]-(E[X])²`.

Standard deviation is `sqrt(Var(X))`.

For constants a,b:

`E[aX+b]=aE[X]+b` and `Var(aX+b)=a²Var(X)`.

## 3.3 Covariance and correlation

`Cov(X,Y)=E[(X-E[X])(Y-E[Y])]`.

Correlation normalizes covariance:

`ρ=Cov(X,Y)/(σ_Xσ_Y)`.

Covariance measures linear co-movement; zero covariance does not generally imply independence.

In ML, covariance matrices encode feature dependence and appear in Gaussian models, PCA, whitening, Kalman filtering, and uncertainty estimation.

## 3.4 Conditional probability and Bayes

`P(A|B)=P(A∩B)/P(B)`.

Bayes' rule follows:

`P(A|B)=P(B|A)P(A)/P(B)`.

For a model with parameter θ and data D:

`p(θ|D) ∝ p(D|θ)p(θ)`.

Likelihood describes how compatible data are with a parameter; prior describes beliefs before seeing the data; posterior combines both.

A crucial distinction: Bayesian posterior probability `P(H|D)` is not the same object as the frequentist p-value `P(D|H)`.

## 3.5 Maximum likelihood and MAP

Maximum likelihood chooses:

`θ_ML=argmax_θ p(D|θ)`.

Because logarithm is monotonic:

`argmax p(D|θ)=argmax log p(D|θ)`.

MAP estimation maximizes:

`θ_MAP=argmax p(D|θ)p(θ)`.

Taking negative logs turns MAP into an optimization problem containing a data-fit term plus a prior-induced penalty. This is the mathematical bridge between Bayesian inference and regularization.

## 3.6 Hypothesis testing

A null hypothesis H0 and alternative H1 define competing claims. A test statistic is compared with a reference distribution under H0.

Type I error: reject true H0. Probability is α.

Type II error: fail to reject false H0. Probability is β; power is `1-β`.

A p-value is the probability, under H0, of observing a test statistic at least as extreme as the one observed. It is **not** the probability that H0 is true.

Multiple testing changes the false-positive problem dramatically. Bonferroni controls family-wise error conservatively; false-discovery-rate procedures target a different error criterion.

## 3.7 Confidence intervals

A 95% confidence interval is a procedure with 95% long-run coverage under repeated sampling assumptions. It does not mean there is a 95% probability that a fixed frequentist parameter lies inside the realized interval.

For a known-variance normal mean:

`x̄ ± z_{α/2} σ/√n`.

Unknown variance commonly leads to a t distribution rather than a z distribution.

# 4. Discrete Mathematics

Discrete mathematics formalizes finite, countable, symbolic, and combinatorial structures that computers manipulate.

## 4.1 Logic

Propositional logic uses statements with truth values. Key operators include AND, OR, NOT, implication, and equivalence.

`P→Q` is logically equivalent to `¬P∨Q`.

Predicate logic introduces quantifiers:

`∀x P(x)` and `∃x P(x)`.

In software verification, logical specifications describe preconditions, postconditions, invariants, and safety properties.

## 4.2 Sets, functions, relations

Sets support union, intersection, difference, Cartesian products, and complements.

A function `f:A→B` assigns each element of A exactly one element of B. Injectivity means distinct inputs have distinct outputs; surjectivity means every element of B is reached; bijectivity gives a reversible correspondence.

A relation R on A is a subset of `A×A`. Important properties include reflexivity, symmetry, antisymmetry, and transitivity.

Equivalence relations partition a set into equivalence classes. Partial orders model dependencies and scheduling constraints.

## 4.3 Graphs and trees

A graph `G=(V,E)` consists of vertices and edges. Directed graphs encode asymmetric relationships; weighted graphs attach costs or capacities.

A path is a sequence of adjacent vertices. A cycle begins and ends at the same vertex. Connectivity determines whether one state can reach another.

Trees are connected acyclic graphs. A tree with n vertices has exactly n−1 edges.

### Graph matrices
The adjacency matrix A encodes edges. The degree matrix D is diagonal. The graph Laplacian is:

`L=D-A`.

For an undirected graph, L is positive semidefinite and has a zero eigenvalue for each connected component. Spectral graph theory connects eigenvalues to connectivity, clustering, diffusion, and graph neural networks.

## 4.4 Combinatorics

The product rule counts sequential choices; the sum rule counts disjoint alternatives.

Permutations of n distinct objects: `n!`.

k-combinations from n objects:

`C(n,k)=n!/(k!(n-k)!)`.

The binomial theorem:

`(x+y)^n=Σ_{k=0}^n C(n,k)x^{n-k}y^k`.

The pigeonhole principle is simple but powerful: mapping more than n objects into n boxes forces a collision. It underlies hashing arguments and impossibility proofs.

## 4.5 Proof techniques

Direct proof: assume premises and derive the conclusion.

Contrapositive: prove `¬Q→¬P` instead of `P→Q`.

Contradiction: assume the negation and derive impossibility.

Induction: prove a base case, then prove `P(k)→P(k+1)`.

Strong induction allows the inductive step to use all previous cases.

Invariants prove that a property remains true throughout an algorithm. Loop invariants are central to algorithm correctness.

# 5. Optimization

Optimization formalizes choosing the best feasible object according to an objective.

General form:

`min_x f(x) subject to g_i(x)≤0 and h_j(x)=0`.

## 5.1 Gradient descent

`x_{t+1}=x_t-η_t∇f(x_t)`.

Learning-rate selection is a mathematical stability problem. Too large a step can diverge; too small a step can make convergence impractically slow.

Stochastic gradient descent replaces the full gradient with an unbiased or approximately unbiased mini-batch estimate. Its noise can help optimization but also makes convergence stochastic.

Momentum maintains a velocity-like state. Adaptive methods such as Adam normalize updates using estimates of first and second moments. Their behavior should be understood in terms of stochastic optimization rather than as magic optimizers.

## 5.2 Convex optimization

A function f is convex if:

`f(θx+(1-θ)y) ≤ θf(x)+(1-θ)f(y)` for `0≤θ≤1`.

For differentiable f, an equivalent first-order condition is:

`f(y) ≥ f(x)+∇f(x)^T(y-x)`.

For twice-differentiable f, positive semidefinite Hessian is a sufficient characterization on convex domains.

The major benefit is global structure: every local minimum is a global minimum.

## 5.3 Regularization

Regularization modifies the objective:

`min_w L(w)+λR(w)`.

L2 regularization uses `R(w)=||w||²`; L1 uses `R(w)=||w||_1`.

Regularization controls model complexity and can improve generalization. In Bayesian terms, penalties can correspond to negative log-priors under suitable assumptions.

Regularization is not identical to solving overfitting. Generalization also depends on data distribution, model class, optimization dynamics, and effective capacity.

## 5.4 Constraint optimization

Equality constraints use Lagrange multipliers:

`L(x,λ)=f(x)+λ^T h(x)`.

At regular constrained optima, stationarity requires the gradient of the Lagrangian to vanish.

Inequality constraints lead to Karush–Kuhn–Tucker conditions: primal feasibility, dual feasibility, complementary slackness, and stationarity.

The KKT framework is foundational for constrained ML, resource allocation, control, adversarial robustness, and operations research.

# 6. Mathematical Foundations of Data Structures and Algorithms

Data structures and algorithms are not primarily mathematics subjects, but their behavior is governed by mathematical structures. The goal here is to understand those foundations.

## 6.1 Arrays, lists, stacks, queues

An array provides indexed access because the address of element i is computed using a base address plus an offset. This is an arithmetic property of contiguous memory.

A linked list represents a sequence through pointers. Its traversal cost is linear because reaching the ith element requires following a chain of links.

A stack implements LIFO semantics; a queue implements FIFO semantics. Their correctness can be specified with invariants about the order of elements.

## 6.2 Hash maps

A hash function maps keys into a finite bucket space. With m buckets and n inserted keys, the load factor is:

`α=n/m`.

Under a suitable approximately uniform hashing model, expected chain length is related to α. Hash collisions are unavoidable when the key space exceeds the bucket space by the pigeonhole principle.

Cryptographic hashes additionally aim for preimage resistance, second-preimage resistance, and collision resistance; these are security properties rather than merely good distribution properties.

## 6.3 Heaps and priority queues

A binary heap maintains a partial order rather than a complete sorted order. The height of a balanced binary heap is `Θ(log n)`, yielding logarithmic insertion and extraction.

The mathematical invariant is more important than memorizing implementation details: every parent satisfies the heap-order relation with its children.

## 6.4 Trees

A binary search tree relies on an ordering invariant: keys in the left subtree are less than the node's key and keys in the right subtree are greater, subject to the duplicate policy.

Balanced trees maintain height `O(log n)`, making search, insertion, and deletion logarithmic.

A tree's height controls recursive algorithm depth, while its branching factor controls the number of nodes explored.

## 6.5 Graph algorithms

Breadth-first search explores by distance layers in an unweighted graph. The shortest-path guarantee follows because vertices are first reached in nondecreasing path length.

Dijkstra's algorithm relies on nonnegative edge weights. Its greedy step is justified by the fact that once the smallest tentative distance is selected, no later nonnegative extension can improve it.

Bellman–Ford uses repeated relaxation and can detect reachable negative cycles.

Minimum spanning trees rely on cut and cycle properties. Kruskal's algorithm uses the fact that the lightest edge crossing a suitable cut is safe to add.

## 6.6 Sorting and searching

Comparison sorting has a lower bound of `Ω(n log n)` comparisons in the worst case because a comparison decision tree with n! possible permutations needs height at least `log_2(n!)`, which is `Θ(n log n)`.

Binary search is logarithmic because each comparison reduces the remaining ordered search interval by approximately half:

`T(n)=T(n/2)+O(1)=O(log n)`.

## 6.7 Dynamic programming

Dynamic programming is based on two mathematical properties:

1. optimal substructure;
2. overlapping subproblems.

A recurrence describes the optimal value of a larger instance using smaller instances.

For Fibonacci:

`F(n)=F(n-1)+F(n-2)`.

Naive recursion repeats exponentially many subproblems; memoization reduces the number of distinct states to O(n), with O(1) work per transition.

The key skill is not memorizing DP templates. Define the state precisely, derive the recurrence, prove its correctness, establish base cases, then analyze the state-space size and transition cost.

## 6.8 Complexity analysis

Big-O gives an asymptotic upper bound; Big-Ω gives an asymptotic lower bound; Big-Θ gives a tight asymptotic bound.

For functions f and g:

`f(n)=O(g(n))` means there exist c,n0 such that `0≤f(n)≤cg(n)` for all `n≥n0`.

Common growth hierarchy:

`1 < log n < n < n log n < n² < n³ < 2^n < n!`.

Amortized analysis studies the average cost over a sequence of operations without requiring randomness. The accounting and potential methods are mathematical tools for proving bounds such as O(1) amortized dynamic-array append.

For randomized algorithms, distinguish worst-case runtime from expected runtime and from high-probability bounds.

## 6.9 Algorithmic problem solving as mathematics

A disciplined solution process is:

1. Define the mathematical input and output.
2. Identify invariants and constraints.
3. Find the relevant structure: ordering, graph, recurrence, probability distribution, algebraic relation, or optimization problem.
4. Derive a candidate algorithm.
5. Prove correctness.
6. Derive time and space complexity.
7. Test boundary cases and adversarial inputs.
8. Determine whether the assumptions actually hold.

# 7. Cross-connections to AI and security

## 7.1 Neural networks

A layer is often:

`z=Wx+b`, followed by `a=σ(z)`.

Linear algebra represents the transformation; calculus supplies gradients; probability supplies loss interpretations; optimization trains the parameters.

## 7.2 Attention

Scaled dot-product attention can be written:

`Attention(Q,K,V)=softmax(QK^T/√d_k)V`.

The mathematics combines matrix multiplication, inner products, normalization, exponentials, probability-like weights, and differentiable optimization.

## 7.3 Cryptography

Modern cryptography depends heavily on modular arithmetic, finite groups, finite fields, probability, entropy, combinatorics, and complexity assumptions.

For a prime p, arithmetic modulo p forms the finite field `F_p`. Polynomial arithmetic over finite fields underlies many cryptographic constructions and error-correcting codes.

Security parameters are quantitative: an attack with probability approximately `2^-128` is qualitatively different from one with probability `2^-40`.

## 7.4 Differential privacy

A randomized mechanism M is ε-differentially private when neighboring datasets D,D' satisfy:

`P[M(D)∈S] ≤ exp(ε)P[M(D')∈S]`

for every measurable output set S, with approximate variants adding δ.

This is a mathematical stability property: an individual's participation should not substantially change the distribution of released outputs.

## 7.5 Distributed systems

Graphs represent network topology. Probability models failures. Queueing theory studies workloads and waiting time. Linear algebra can describe consensus and network processes. Markov chains model stochastic state transitions.

The mathematical question is often not merely whether a system works, but under what assumptions it works and with what probability or bound.

# 8. Exercises and solution sketches

### Linear algebra
1. Prove Cauchy–Schwarz using nonnegativity of `||x-ty||²`.
2. Derive the normal equations for least squares.
3. Show that covariance matrices are positive semidefinite.
4. Explain why PCA eigenvectors maximize projected variance.

### Calculus
1. Derive the gradient of `f(x)=1/2||Ax-b||²`.
2. Use the chain rule to derive the gradient through a two-layer linear network.
3. Classify stationary points using the Hessian.

### Probability/statistics
1. Derive `Var(X)=E[X²]-E[X]²`.
2. Prove Bayes' theorem from conditional probability.
3. Explain why zero covariance does not imply independence.
4. Construct a hypothesis test and identify Type I/II errors.

### Discrete mathematics
1. Prove that a tree with n vertices has n−1 edges.
2. Prove correctness of binary search using a loop invariant.
3. Derive the comparison-sorting lower bound from decision trees.

### Optimization
1. Derive the gradient update for L2-regularized least squares.
2. Show why a convex differentiable stationary point is globally optimal.
3. Derive KKT conditions for a simple constrained quadratic problem.

### Algorithms
1. Derive the recurrence for merge sort and solve it using a recursion tree or Master theorem.
2. Prove BFS shortest-path correctness in unweighted graphs.
3. Give an amortized proof for dynamic-array append.
4. Formulate a shortest-path problem as a graph optimization problem.

# 9. Recommended progression

**Stage 1 — Foundations:** algebra, functions, vectors, sets, logic, basic probability, derivatives.

**Stage 2 — Core CS mathematics:** matrices, proof techniques, combinatorics, graph theory, recurrences, asymptotic analysis.

**Stage 3 — ML mathematics:** multivariable calculus, probability distributions, estimation, linear regression, PCA, optimization, convexity.

**Stage 4 — Advanced AI:** Jacobians/Hessians, numerical stability, stochastic optimization, probabilistic graphical models, information theory, high-dimensional geometry.

**Stage 5 — Security/research:** finite fields, number theory, probability bounds, entropy, information-theoretic security, differential privacy, adversarial optimization, randomized algorithms.

At every stage, implement selected concepts in Python or Rust and verify the mathematics experimentally. Numerical experiments are useful for intuition, but they do not replace proofs.

# 10. References for deeper study

- Gilbert Strang, *Introduction to Linear Algebra*.
- Sheldon Axler, *Linear Algebra Done Right*.
- Tom M. Apostol, *Calculus*.
- Stephen Boyd and Lieven Vandenberghe, *Convex Optimization*.
- Dimitri Bertsekas, *Nonlinear Programming*.
- Larry Wasserman, *All of Statistics*.
- Christopher Bishop, *Pattern Recognition and Machine Learning*.
- Kevin Murphy, *Probabilistic Machine Learning*.
- Trevor Hastie, Robert Tibshirani, Jerome Friedman, *The Elements of Statistical Learning*.
- Thomas Cormen et al., *Introduction to Algorithms*.
- Michael Sipser, *Introduction to the Theory of Computation*.
- Kenneth Rosen, *Discrete Mathematics and Its Applications*.
- Jonathan Katz and Yehuda Lindell, *Introduction to Modern Cryptography*.
- Dwork and Roth, *The Algorithmic Foundations of Differential Privacy*.

The notebook should treat these references as starting points. For research work, verify theorem statements, assumptions, and current terminology against primary papers and authoritative graduate texts.
