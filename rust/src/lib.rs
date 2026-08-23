//! Small exact arithmetic primitives used by the notebook.

/// Greatest common divisor using Euclid's algorithm.
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Extended Euclidean algorithm. Returns `(g, x, y)` such that `ax + by = g`.
pub fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        return (a.abs(), a.signum(), 0);
    }
    let (g, x1, y1) = extended_gcd(b, a % b);
    (g, y1, x1 - (a / b) * y1)
}

/// Modular inverse when it exists.
pub fn mod_inverse(a: i64, modulus: i64) -> Option<i64> {
    if modulus <= 1 {
        return None;
    }
    let (g, x, _) = extended_gcd(a.rem_euclid(modulus), modulus);
    if g != 1 { None } else { Some(x.rem_euclid(modulus)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gcd_works() { assert_eq!(gcd(84, 30), 6); }

    #[test]
    fn modular_inverse_works() { assert_eq!(mod_inverse(3, 11), Some(4)); }

    #[test]
    fn non_invertible_value_is_rejected() { assert_eq!(mod_inverse(6, 15), None); }
}
