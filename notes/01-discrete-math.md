# 01 — Discrete Mathematics

## Proof patterns

### Direct proof
To prove `P => Q`, assume `P` and derive `Q` using definitions and previously established results.

### Contrapositive
`P => Q` is equivalent to `not Q => not P`. This is often useful when the negation of the conclusion has a simpler structure.

### Contradiction
Assume the proposition is false and derive an impossibility. In algorithmic proofs, identify exactly which invariant or assumption is contradicted.

## Sets and relations

For finite sets, inclusion-exclusion gives:

`|A union B| = |A| + |B| - |A intersection B|`.

A relation `R` is an equivalence relation when it is reflexive, symmetric, and transitive. Equivalence classes partition the underlying set.

## Combinatorics

The number of ordered selections of `k` objects from `n` distinct objects is:

`P(n,k) = n! / (n-k)!`.

The number of unordered selections is:

`C(n,k) = n! / (k!(n-k)!)`.

## Computer-science connection

Counting arguments appear directly in complexity bounds, probability of randomized algorithms, cryptographic key spaces, and state-space analysis.
