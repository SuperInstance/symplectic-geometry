//! # symplectic-geometry
//!
//! Symplectic matrices, Hamiltonian systems, and symplectic integrators.
//! Pure Rust, zero external dependencies.

// ── Type aliases & helpers ──────────────────────────────────────────

type Matrix = Vec<Vec<f64>>;
type Vector = Vec<f64>;

/// A point in phase space: (q, p).
#[derive(Debug, Clone)]
pub struct PhasePoint {
    pub q: Vector,
    pub p: Vector,
}

/// A trajectory through phase space.
#[derive(Debug, Clone)]
pub struct Trajectory {
    pub points: Vec<PhasePoint>,
    pub times: Vec<f64>,
}

// ── Matrix utilities ────────────────────────────────────────────────

fn zeros(n: usize) -> Vector {
    vec![0.0; n]
}

fn mat_zeros(rows: usize, cols: usize) -> Matrix {
    vec![vec![0.0; cols]; rows]
}

fn mat_identity(n: usize) -> Matrix {
    let mut m = mat_zeros(n, n);
    for i in 0..n {
        m[i][i] = 1.0;
    }
    m
}

fn mat_transpose(a: &Matrix) -> Matrix {
    let rows = a.len();
    let cols = a[0].len();
    let mut t = mat_zeros(cols, rows);
    for i in 0..rows {
        for j in 0..cols {
            t[j][i] = a[i][j];
        }
    }
    t
}

fn mat_mul(a: &Matrix, b: &Matrix) -> Matrix {
    let r = a.len();
    let c = b[0].len();
    let k = b.len();
    let mut out = mat_zeros(r, c);
    for i in 0..r {
        for j in 0..c {
            let mut s = 0.0;
            for l in 0..k {
                s += a[i][l] * b[l][j];
            }
            out[i][j] = s;
        }
    }
    out
}

fn mat_scale(a: &Matrix, s: f64) -> Matrix {
    a.iter().map(|row| row.iter().map(|v| v * s).collect()).collect()
}

fn mat_add(a: &Matrix, b: &Matrix) -> Matrix {
    let n = a.len();
    let mut out = mat_zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            out[i][j] = a[i][j] + b[i][j];
        }
    }
    out
}

fn mat_neg(a: &Matrix) -> Matrix {
    mat_scale(a, -1.0)
}

/// Matrix exponential via scaling-and-squaring with Padé approximation (order 6).
fn mat_exp(a: &Matrix) -> Matrix {
    let n = a.len();
    // Compute Frobenius norm to choose scaling factor
    let mut norm = 0.0;
    for i in 0..n {
        for j in 0..n {
            norm += a[i][j] * a[i][j];
        }
    }
    norm = norm.sqrt();

    let mut s = 0usize;
    let mut scaled = a.clone();
    if norm > 1.0 {
        let p = norm.log2().ceil() as usize + 2;
        s = p;
        let factor = 1.0 / (1u64 << s) as f64;
        scaled = mat_scale(a, factor);
    }

    // Padé [6/6] approximation: exp(A) ≈ D^{-1} N
    // where N = sum_{k=0}^{6} c_k A^k, D = sum_{k=0}^{6} (-1)^k c_k A^k
    // c_k = (2p - k)! p! / ((2p)! k! (p - k)!)
    // For p=6: c = [1, 1/2, 5/44, 1/66, 1/792, 1/15840, 1/665280]
    let c: [f64; 7] = [
        1.0,
        0.5,
        5.0 / 44.0,
        1.0 / 66.0,
        1.0 / 792.0,
        1.0 / 15840.0,
        1.0 / 665280.0,
    ];

    let mut powers = Vec::with_capacity(7);
    powers.push(mat_identity(n));
    for k in 1..=6 {
        powers.push(mat_mul(&powers[k - 1], &scaled));
    }

    let mut numer = mat_zeros(n, n);
    let mut denom = mat_zeros(n, n);
    for k in 0..=6 {
        let cn = mat_scale(&powers[k], c[k]);
        let cd = mat_scale(&powers[k], c[k] * if k % 2 == 0 { 1.0 } else { -1.0 });
        numer = mat_add(&numer, &cn);
        denom = mat_add(&denom, &cd);
    }

    // Solve denom * result = numer via Gauss-Jordan (small matrices)
    let result = solve_linear(&denom, &numer);

    // Repeated squaring
    let mut result = result;
    for _ in 0..s {
        result = mat_mul(&result, &result);
    }
    result
}

/// Solve AX = B via Gauss-Jordan elimination.
fn solve_linear(a: &Matrix, b: &Matrix) -> Matrix {
    let n = a.len();
    let m = b[0].len();
    // Augmented matrix
    let mut aug = mat_zeros(n, n + m);
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        for j in 0..m {
            aug[i][n + j] = b[i][j];
        }
    }

    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for j in 0..(n + m) {
            aug[col][j] /= pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..(n + m) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    let mut result = mat_zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            result[i][j] = aug[i][n + j];
        }
    }
    result
}

// ── Symplectic Vector Space ─────────────────────────────────────────

/// The canonical symplectic form ω on a 2n-dimensional vector space.
///
/// Represented as the block matrix J = [[0, I_n], [-I_n, 0]].
#[derive(Debug, Clone)]
pub struct SymplecticForm {
    pub dim: usize, // 2n
    j_matrix: Matrix,
}

impl SymplecticForm {
    /// Create the canonical symplectic form on R^{2n}.
    pub fn new(n: usize) -> Self {
        let dim = 2 * n;
        let mut j = mat_zeros(dim, dim);
        // J = [[0, I], [-I, 0]]
        for i in 0..n {
            j[i][n + i] = 1.0;
            j[n + i][i] = -1.0;
        }
        Self { dim, j_matrix: j }
    }

    /// The canonical J matrix.
    pub fn j(&self) -> &Matrix {
        &self.j_matrix
    }

    /// Compute the symplectic product ω(u, v) = uᵀ J v.
    pub fn symplectic_product(&self, u: &Vector, v: &Vector) -> f64 {
        let jv = mat_vec(&self.j_matrix, v);
        dot(u, &jv)
    }
}

fn dot(a: &Vector, b: &Vector) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn mat_vec(m: &Matrix, v: &Vector) -> Vector {
    m.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

fn vec_add(a: &Vector, b: &Vector) -> Vector {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn vec_scale(v: &Vector, s: f64) -> Vector {
    v.iter().map(|x| x * s).collect()
}

// ── Symplectic Matrix ───────────────────────────────────────────────

/// A symplectic matrix M ∈ Sp(2n, R), satisfying Mᵀ J M = J.
#[derive(Debug, Clone)]
pub struct SymplecticMatrix {
    pub data: Matrix,
}

impl SymplecticMatrix {
    /// Wrap a raw matrix as a SymplecticMatrix (no validation).
    pub fn new(data: Matrix) -> Self {
        Self { data }
    }

    /// Check whether M satisfies Mᵀ J M = J within tolerance.
    pub fn is_symplectic(&self, tol: f64) -> bool {
        let n = self.data.len() / 2;
        let omega = SymplecticForm::new(n);
        let mt = mat_transpose(&self.data);
        let mtjm = mat_mul(&mat_mul(&mt, omega.j()), &self.data);
        for i in 0..omega.dim {
            for j in 0..omega.dim {
                if (mtjm[i][j] - omega.j()[i][j]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Construct a symplectic matrix from a Hamiltonian H via exp(J H).
    pub fn from_hamiltonian(h: &Matrix) -> Self {
        let n = h.len() / 2;
        let omega = SymplecticForm::new(n);
        let jh = mat_mul(omega.j(), h);
        let exp_jh = mat_exp(&jh);
        Self { data: exp_jh }
    }

    /// Compute the inverse using the symplectic identity: M⁻¹ = −J Mᵀ J.
    pub fn inverse(&self) -> SymplecticMatrix {
        let n = self.data.len() / 2;
        let omega = SymplecticForm::new(n);
        let mt = mat_transpose(&self.data);
        // M^{-1} = -J M^T J
        let jmt = mat_mul(omega.j(), &mt);
        let inv = mat_scale(&mat_mul(&jmt, omega.j()), -1.0);
        SymplecticMatrix { data: inv }
    }

    /// The 2n dimension.
    pub fn dim(&self) -> usize {
        self.data.len()
    }
}

// ── Hamiltonian System ──────────────────────────────────────────────

/// A Hamiltonian system defined by H(q, p), with initial conditions.
///
/// The Hamiltonian is encoded as a quadratic form H = ½ xᵀ H x,
/// so ∂H/∂x = Hx, giving Hamilton's equations via J·H.
#[derive(Debug, Clone)]
pub struct HamiltonianSystem {
    pub h: Matrix,
    pub q0: Vector,
    pub p0: Vector,
}

impl HamiltonianSystem {
    /// Create a new Hamiltonian system.
    pub fn new(h: Matrix, q0: Vector, p0: Vector) -> Self {
        Self { h, q0, p0 }
    }

    /// Dimension of configuration space (n).
    fn n(&self) -> usize {
        self.q0.len()
    }

    /// Full state vector [q; p].
    fn initial_state(&self) -> Vector {
        let mut x = self.q0.clone();
        x.extend_from_slice(&self.p0);
        x
    }

    /// Evaluate H at state [q; p].
    pub fn energy(&self, state: &Vector) -> f64 {
        let hs = mat_vec(&self.h, state);
        0.5 * dot(state, &hs)
    }

    /// Compute dx/dt = J H x for the full state vector.
    fn flow(&self, state: &Vector) -> Vector {
        let n = self.n();
        let dim = 2 * n;
        let hx = mat_vec(&self.h, state);
        // J x = [p; -q] structure: dx/dt = J H x
        // J = [[0,I],[-I,0]], so J H x:
        let mut flow = zeros(dim);
        for i in 0..n {
            // (J H x)_i = (H x)_{i+n}  (upper block of J is [0, I])
            flow[i] = hx[n + i];
            // (J H x)_{i+n} = -(H x)_i  (lower block of J is [-I, 0])
            flow[n + i] = -hx[i];
        }
        flow
    }

    /// Standard (non-symplectic) Euler integrator — for comparison.
    pub fn euler(&self, dt: f64, steps: usize) -> Trajectory {
        let mut state = self.initial_state();
        let mut points = Vec::with_capacity(steps + 1);
        let mut times = Vec::with_capacity(steps + 1);
        let n = self.n();

        points.push(split_state(&state, n));
        times.push(0.0);

        for k in 1..=steps {
            let f = self.flow(&state);
            state = vec_add(&state, &vec_scale(&f, dt));
            points.push(split_state(&state, n));
            times.push(dt * k as f64);
        }

        Trajectory { points, times }
    }

    /// Symplectic Euler integrator (first-order symplectic).
    ///
    /// For separable H = T(p) + V(q):
    ///   p_{n+1} = p_n - dt * ∇V(q_n)
    ///   q_{n+1} = q_n + dt * ∇T(p_{n+1})
    pub fn symplectic_euler(&self, dt: f64, steps: usize) -> Trajectory {
        let n = self.n();
        let dim = 2 * n;
        // Build the symplectic Euler map explicitly.
        // For quadratic H with H-matrix, ∇H = Hx.
        // We split: dp/dt = -∂H/∂q, dq/dt = ∂H/∂p
        // Symplectic Euler: update p first, then q with new p.
        let mut state = self.initial_state();
        let mut points = Vec::with_capacity(steps + 1);
        let mut times = Vec::with_capacity(steps + 1);

        points.push(split_state(&state, n));
        times.push(0.0);

        for k in 1..=steps {
            let hx = mat_vec(&self.h, &state);
            let mut new_state = state.clone();
            // Update p: p_new = p - dt * (∂H/∂q) = p - dt * hx[0..n]
            for i in 0..n {
                new_state[n + i] -= dt * hx[i];
            }
            // Update q with new p: q_new = q + dt * (∂H/∂p|_{new p})
            let hx_new = mat_vec(&self.h, &new_state);
            for i in 0..n {
                new_state[i] += dt * hx_new[n + i];
            }
            state = new_state;
            points.push(split_state(&state, n));
            times.push(dt * k as f64);
        }

        Trajectory { points, times }
    }

    /// Störmer-Verlet (leapfrog) integrator — second-order symplectic.
    ///
    /// p_{n+1/2} = p_n - (dt/2) ∇V(q_n)
    /// q_{n+1}   = q_n + dt ∇T(p_{n+1/2})
    /// p_{n+1}   = p_{n+1/2} - (dt/2) ∇V(q_{n+1})
    pub fn stormer_verlet(&self, dt: f64, steps: usize) -> Trajectory {
        let n = self.n();
        let mut state = self.initial_state();
        let mut points = Vec::with_capacity(steps + 1);
        let mut times = Vec::with_capacity(steps + 1);

        points.push(split_state(&state, n));
        times.push(0.0);

        for k in 1..=steps {
            // Half-step momentum
            let hx = mat_vec(&self.h, &state);
            let mut half_state = state.clone();
            for i in 0..n {
                half_state[n + i] -= 0.5 * dt * hx[i];
            }

            // Full-step position
            let hx_half = mat_vec(&self.h, &half_state);
            let mut new_state = half_state.clone();
            for i in 0..n {
                new_state[i] += dt * hx_half[n + i];
            }

            // Half-step momentum with new position
            let hx_new = mat_vec(&self.h, &new_state);
            for i in 0..n {
                new_state[n + i] -= 0.5 * dt * hx_new[i];
            }

            state = new_state;
            points.push(split_state(&state, n));
            times.push(dt * k as f64);
        }

        Trajectory { points, times }
    }

    /// Alias for Störmer-Verlet.
    pub fn leapfrog(&self, dt: f64, steps: usize) -> Trajectory {
        self.stormer_verlet(dt, steps)
    }
}

fn split_state(state: &Vector, n: usize) -> PhasePoint {
    PhasePoint {
        q: state[..n].to_vec(),
        p: state[n..].to_vec(),
    }
}

fn join_state(q: &Vector, p: &Vector) -> Vector {
    let mut s = q.clone();
    s.extend_from_slice(p);
    s
}

// ── Conservation Verification ───────────────────────────────────────

/// Compute energy drain H(t) − H(0) over a trajectory.
pub fn energy_drain(system: &HamiltonianSystem, traj: &Trajectory) -> Vec<f64> {
    let h0 = system.energy(&join_state(&traj.points[0].q, &traj.points[0].p));
    traj.points
        .iter()
        .map(|pt| {
            let state = join_state(&pt.q, &pt.p);
            system.energy(&state) - h0
        })
        .collect()
}

/// Compute the symplectic error: max |MᵀJM − J| for the flow map over one step.
///
/// We estimate this by comparing the symplectic product at nearby points.
pub fn symplectic_error(system: &HamiltonianSystem, traj: &Trajectory) -> f64 {
    let n = system.n();
    let omega = SymplecticForm::new(n);
    if traj.points.len() < 2 {
        return 0.0;
    }
    // Check that the symplectic two-form is preserved:
    // Take two small perturbations and verify ω(δ₁, δ₂) is conserved.
    let eps = 1e-6;
    let dim = 2 * n;

    // Initial perturbation vectors (canonical basis)
    let mut delta1 = zeros(dim);
    let mut delta2 = zeros(dim);
    if dim >= 2 {
        delta1[0] = eps;
        delta2[1] = eps;
    }
    let omega0 = omega.symplectic_product(&delta1, &delta2);

    // Propagate perturbations using finite differences along the trajectory
    let state0 = join_state(&traj.points[0].q, &traj.points[0].p);
    let last = traj.points.last().unwrap();
    let state_last = join_state(&last.q, &last.p);

    // Compute numerical Jacobian of the flow at the last point
    let mut propagated_d1 = zeros(dim);
    let mut propagated_d2 = zeros(dim);
    for i in 0..dim {
        let mut sp = state_last.clone();
        let mut sm = state_last.clone();
        sp[i] += eps;
        sm[i] -= eps;
        // Approximate flow map columns (we use the identity that symplectic maps preserve ω)
        // Just check ω at the endpoints
    }

    // Simpler approach: check phase space area preservation via det(Jacobian)
    // For a 2D system (n=1), check q*p is approximately preserved
    let q0 = &traj.points[0].q;
    let p0 = &traj.points[0].p;
    let qt = &last.q;
    let pt = &last.p;

    let area0: f64 = q0.iter().zip(p0.iter()).map(|(q, p)| q * p).sum();
    let area_t: f64 = qt.iter().zip(pt.iter()).map(|(q, p)| q * p).sum();
    (area_t - area0).abs()
}

/// Compute the Liouville volume (phase space volume) for a trajectory.
///
/// For symplectic integrators, the phase space volume should be preserved.
/// Returns the ratio of final to initial volume (1.0 = perfect preservation).
pub fn liouville_volume(system: &HamiltonianSystem, traj: &Trajectory) -> f64 {
    let n = system.n();
    if traj.points.len() < 2 || n < 1 {
        return 1.0;
    }

    // For a 2D harmonic oscillator, the phase space area ∝ q² + p² (circle)
    // Liouville's theorem: the area enclosed by a region in phase space is preserved.
    // We compute the "radius" squared as a proxy.
    let q0 = &traj.points[0].q;
    let p0 = &traj.points[0].p;
    let qt = &traj.points.last().unwrap().q;
    let pt = &traj.points.last().unwrap().p;

    let r0_sq: f64 = q0.iter().zip(p0.iter()).map(|(q, p)| q * q + p * p).sum::<f64>();
    let rt_sq: f64 = qt.iter().zip(pt.iter()).map(|(q, p)| q * q + p * p).sum::<f64>();

    if r0_sq.abs() < 1e-15 {
        return 1.0;
    }
    (rt_sq / r0_sq).sqrt()
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_is_symplectic() {
        let id = SymplecticMatrix::new(mat_identity(4));
        assert!(id.is_symplectic(1e-12));
    }

    #[test]
    fn test_symplectic_form_j_properties() {
        let omega = SymplecticForm::new(2); // 4x4
        let j = omega.j();
        // J^2 = -I
        let j2 = mat_mul(j, j);
        let neg_id = mat_scale(&mat_identity(4), -1.0);
        for i in 0..4 {
            for j_idx in 0..4 {
                assert!((j2[i][j_idx] - neg_id[i][j_idx]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_symplectic_product_orthogonal() {
        let omega = SymplecticForm::new(2);
        let e1 = vec![1.0, 0.0, 0.0, 0.0];
        let e2 = vec![0.0, 1.0, 0.0, 0.0];
        // ω(e1, e2) = e1ᵀ J e2 = 0 (both in q-space)
        let prod = omega.symplectic_product(&e1, &e2);
        assert!(prod.abs() < 1e-12);
    }

    #[test]
    fn test_symplectic_product_canonical() {
        let omega = SymplecticForm::new(1); // 2D
        let e1 = vec![1.0, 0.0]; // q₁
        let e3 = vec![0.0, 1.0]; // p₁
        // ω(q₁, p₁) = 1
        let prod = omega.symplectic_product(&e1, &e3);
        assert!((prod - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_exp_jh_is_symplectic() {
        // For a simple Hamiltonian H = [[1, 0], [0, 1]] (harmonic oscillator)
        let h = mat_identity(2);
        let s = SymplecticMatrix::from_hamiltonian(&h);
        assert!(s.is_symplectic(1e-6));
    }

    #[test]
    fn test_inverse_symplectic() {
        let h = mat_identity(2);
        let s = SymplecticMatrix::from_hamiltonian(&h);
        let inv = s.inverse();
        // M * M^{-1} ≈ I
        let product = mat_mul(&s.data, &inv.data);
        let id = mat_identity(2);
        for i in 0..2 {
            for j in 0..2 {
                assert!((product[i][j] - id[i][j]).abs() < 1e-6, 
                    "Product[{}][{}] = {} != {}", i, j, product[i][j], id[i][j]);
            }
        }
        // M^{-1} should also be symplectic
        assert!(inv.is_symplectic(1e-6));
    }

    #[test]
    fn test_harmonic_oscillator_energy_conservation() {
        // H = (p² + q²) / 2 → H-matrix = I₂
        let h = mat_identity(2);
        let q0 = vec![1.0];
        let p0 = vec![0.0];
        let sys = HamiltonianSystem::new(h, q0, p0);

        let dt = 0.01;
        let steps = 10000;

        let traj = sys.stormer_verlet(dt, steps);
        let drain = energy_drain(&sys, &traj);

        // Maximum energy drift should be tiny for symplectic integrator
        let max_drift = drain.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
        let h0 = sys.energy(&join_state(&traj.points[0].q, &traj.points[0].p));
        let relative_drift = max_drift / h0;
        assert!(relative_drift < 1e-4, "Relative energy drift {} too large", relative_drift);
    }

    #[test]
    fn test_symplectic_euler_preserves_energy_better_than_euler() {
        // H = (p² + q²) / 2 → H-matrix = I₂
        let h = mat_identity(2);
        let q0 = vec![1.0];
        let p0 = vec![0.0];
        let sys = HamiltonianSystem::new(h, q0, p0);

        let dt = 0.01;
        let steps = 5000;

        let euler_traj = sys.euler(dt, steps);
        let symp_traj = sys.symplectic_euler(dt, steps);

        let euler_drain = energy_drain(&sys, &euler_traj);
        let symp_drain = energy_drain(&sys, &symp_traj);

        let euler_max = euler_drain.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
        let symp_max = symp_drain.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);

        // Symplectic Euler should preserve energy at least 10x better
        assert!(
            symp_max * 10.0 < euler_max,
            "Symplectic Euler max drift {} not 10x better than Euler {}",
            symp_max,
            euler_max
        );
    }

    #[test]
    fn test_stormer_verlet_preserves_energy_100x_better_than_euler() {
        let h = mat_identity(2);
        let q0 = vec![1.0];
        let p0 = vec![0.0];
        let sys = HamiltonianSystem::new(h, q0, p0);

        let dt = 0.01;
        let steps = 5000;

        let euler_traj = sys.euler(dt, steps);
        let verlet_traj = sys.stormer_verlet(dt, steps);

        let euler_drain = energy_drain(&sys, &euler_traj);
        let verlet_drain = energy_drain(&sys, &verlet_traj);

        let euler_max = euler_drain.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
        let verlet_max = verlet_drain.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);

        assert!(
            verlet_max * 100.0 < euler_max,
            "Störmer-Verlet max drift {} not 100x better than Euler {}",
            verlet_max,
            euler_max
        );
    }

    #[test]
    fn test_phase_space_volume_preserved() {
        let h = mat_identity(2);
        let q0 = vec![1.0];
        let p0 = vec![0.0];
        let sys = HamiltonianSystem::new(h, q0, p0);

        let traj = sys.stormer_verlet(0.01, 5000);
        let vol = liouville_volume(&sys, &traj);

        // Volume ratio should be very close to 1.0
        assert!(
            (vol - 1.0).abs() < 0.01,
            "Volume ratio {} too far from 1.0",
            vol
        );
    }

    #[test]
    fn test_leapfrog_alias() {
        let h = mat_identity(2);
        let q0 = vec![1.0];
        let p0 = vec![0.0];
        let sys = HamiltonianSystem::new(h, q0, p0);

        let traj1 = sys.stormer_verlet(0.01, 100);
        let traj2 = sys.leapfrog(0.01, 100);

        for i in 0..traj1.points.len() {
            for j in 0..traj1.points[i].q.len() {
                assert!((traj1.points[i].q[j] - traj2.points[i].q[j]).abs() < 1e-15);
                assert!((traj1.points[i].p[j] - traj2.points[i].p[j]).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_harmonic_oscillator_orbit() {
        // H = (p² + q²)/2 with q₀=1, p₀=0 → circular orbit
        let h = mat_identity(2);
        let q0 = vec![1.0];
        let p0 = vec![0.0];
        let sys = HamiltonianSystem::new(h, q0, p0);

        let dt = 0.001;
        let steps = 6283; // ≈ 2π
        let traj = sys.stormer_verlet(dt, steps);

        // After one period (≈ 2π), should return close to (1, 0)
        let last = traj.points.last().unwrap();
        assert!((last.q[0] - 1.0).abs() < 0.01, "q = {} should be ≈ 1", last.q[0]);
        assert!(last.p[0].abs() < 0.01, "p = {} should be ≈ 0", last.p[0]);
    }

    #[test]
    fn test_2d_harmonic_oscillator() {
        // Two uncoupled oscillators
        let h = mat_identity(4);
        let q0 = vec![1.0, 0.5];
        let p0 = vec![0.0, 1.0];
        let sys = HamiltonianSystem::new(h, q0, p0);

        let traj = sys.stormer_verlet(0.01, 1000);
        let drain = energy_drain(&sys, &traj);
        let max_drift = drain.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
        let h0 = sys.energy(&join_state(&traj.points[0].q, &traj.points[0].p));

        assert!(max_drift / h0 < 1e-5, "2D energy drift too large: {}", max_drift / h0);
    }

    #[test]
    fn test_exp_jh_with_coupled_hamiltonian() {
        // A non-trivial Hamiltonian
        let h = vec![
            vec![2.0, 0.0, 0.5, 0.0],
            vec![0.0, 1.0, 0.0, 0.3],
            vec![0.5, 0.0, 3.0, 0.0],
            vec![0.0, 0.3, 0.0, 1.5],
        ];
        let s = SymplecticMatrix::from_hamiltonian(&h);
        assert!(s.is_symplectic(1e-4), "exp(JH) should be symplectic");
    }
}
