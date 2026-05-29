# symplectic-geometry

**Symplectic matrices, Hamiltonian systems, and symplectic integrators — pure Rust, zero dependencies.**

## The Big Idea: Symplectic = Rotation in Phase Space

Imagine you're tracking a pendulum. At every moment, you know two things: where it is (position *q*) and how fast it's moving (momentum *p*). These two numbers define a point in **phase space** — a 2D map of every possible state.

Now run time forward. The point traces out a path. Here's the key question: **does the path distort the map, or just rotate on it?**

```
Standard Euler integration:          Symplectic integration:

    p                                   p
    |   ·  ·                            |    · · · ·
    |  ·    ·  ← spirals outward       |   ·       ·  ← stays on circle
    | ·      ·                          |  ·         ·
    |·        ·                         | ·           ·
    +----------→ q                      +-------------→ q
    Energy leaks → orbit grows          Energy conserved → orbit preserved
```

**Translation** moves points but doesn't preserve the shape of phase space. **Rotation** preserves it. A symplectic integrator is one that acts as a rotation in phase space — and rotations preserve the metric. *That's* why they conserve energy.

More precisely: a symplectic map *M* satisfies *MᵀJM = J* where *J* is the canonical symplectic form. This condition means the map preserves areas in phase space (Liouville's theorem). If your integrator is symplectic, energy doesn't drift — it oscillates around the true value forever.

## Why This Matters

If you simulate a Hamiltonian system (anything from orbital mechanics to molecular dynamics to quantum spin chains) with a standard integrator, energy drifts. Slowly at first, then catastrophically. The orbit spirals out. After 10,000 steps you're simulating a different system.

Symplectic integrators don't have this problem. They preserve the geometric structure of phase space, so energy stays bounded *forever* — not because of clever error correction, but because the math literally won't let it escape.

## Quick Start

```rust
use symplectic_geometry::{HamiltonianSystem, energy_drain};

fn main() {
    // Harmonic oscillator: H = (p² + q²) / 2
    // H-matrix is the 2×2 identity
    let h = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let q0 = vec![1.0];  // start at q=1
    let p0 = vec![0.0];  // zero momentum

    let sys = HamiltonianSystem::new(h, q0, p0);

    // Compare three integrators over 50 time units (5000 steps of dt=0.01)
    let euler_traj = sys.euler(0.01, 5000);
    let symp_traj  = sys.symplectic_euler(0.01, 5000);
    let verlet_traj = sys.stormer_verlet(0.01, 5000);

    let euler_drain = energy_drain(&sys, &euler_traj);
    let symp_drain  = energy_drain(&sys, &symp_traj);
    let verlet_drain = energy_drain(&sys, &verlet_traj);

    let euler_max = euler_drain.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
    let symp_max  = symp_drain.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
    let verlet_max = verlet_drain.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);

    println!("Standard Euler max drift:  {:.6}", euler_max);   // ~0.25 (catastrophic)
    println!("Symplectic Euler max drift: {:.6}", symp_max);    // ~0.005 (bounded!)
    println!("Störmer-Verlet max drift:   {:.6}", verlet_max);  // ~0.00001 (excellent!)
}
```

## The Three Integrators

### Standard Euler — The Cautionary Tale

```
x_{n+1} = x_n + dt · f(x_n)
```

First-order, cheap, and **dangerous** for Hamiltonian systems. Each step translates the state in phase space. Translation is *not* area-preserving. Energy drifts monotonically — it only goes in one direction (usually up). After enough steps, your simulation is fiction.

Use it only as a baseline to see how badly things can go wrong.

### Symplectic Euler — The First Guardrail

```
p_{n+1} = p_n - dt · ∇V(q_n)       (update momentum first)
q_{n+1} = q_n + dt · ∇T(p_{n+1})   (update position with NEW momentum)
```

Still first-order, but the asymmetric update creates a symplectic map. Energy oscillates around the true value with amplitude ~O(dt) — it never escapes. The orbit stays on a slightly deformed energy surface.

The trick: by using the *updated* momentum to advance position, the combined map becomes a canonical transformation. Each step is area-preserving.

### Störmer-Verlet (Leapfrog) — The Gold Standard

```
p_{n+1/2} = p_n         - (dt/2) · ∇V(q_n)      half-kick
q_{n+1}   = q_n         +  dt     · ∇T(p_{n+1/2}) drift
p_{n+1}   = p_{n+1/2}   - (dt/2) · ∇V(q_{n+1})   half-kick
```

Second-order, symplectic, **time-reversible**. This is the workhorse of computational physics. The half-step-at-the-beginning-and-end structure means each full step is the composition of two symplectic maps, which is automatically symplectic.

Energy oscillation is O(dt²) — an order of magnitude better than symplectic Euler at the same step size. And it's time-reversible: run it backwards and you retrace your steps exactly. Standard Euler can't do this.

The Verlet integrator is *the* reason molecular dynamics simulations can run for billions of steps without blowing up.

## The Symplectic Form

The mathematical heart of the library:

```
J = [[ 0,  I],
     [-I,  0]]
```

This 2n×2n matrix defines the symplectic structure on phase space. Key properties:

- **J² = -I**: applying the symplectic form twice gives a 90° rotation
- **ω(u, v) = uᵀJv**: the symplectic product of two phase-space vectors
- **A matrix M is symplectic iff MᵀJM = J**: it preserves the symplectic form

```rust
use symplectic_geometry::{SymplecticForm, SymplecticMatrix};

// The canonical symplectic form on R⁴ (2 position + 2 momentum)
let omega = SymplecticForm::new(2);  // 4-dimensional phase space

// Symplectic product of canonical basis vectors
let q1 = vec![1.0, 0.0, 0.0, 0.0];  // first position
let p1 = vec![0.0, 0.0, 1.0, 0.0];  // first momentum
let prod = omega.symplectic_product(&q1, &p1);  // = 1.0 (canonical pair)

// Generate a symplectic matrix from a Hamiltonian
let h = vec![
    vec![2.0, 0.0, 0.5, 0.0],
    vec![0.0, 1.0, 0.0, 0.3],
    vec![0.5, 0.0, 3.0, 0.0],
    vec![0.0, 0.3, 0.0, 1.5],
];
let s = SymplecticMatrix::from_hamiltonian(&h);
assert!(s.is_symplectic(1e-4));  // MᵀJM = J ✓

// Symplectic inverse: M⁻¹ = -JMᵀJ (no matrix inversion needed!)
let inv = s.inverse();
```

The `from_hamiltonian` constructor computes exp(JH) using a Padé [6/6] approximation with scaling-and-squaring — the same approach used by scipy and MATLAB. The symplectic inverse formula `-JMᵀJ` means you never need to solve a linear system; it's an algebraic identity.

## Energy Tracking

```rust
use symplectic_geometry::{HamiltonianSystem, energy_drain, liouville_volume};

let sys = HamiltonianSystem::new(/* ... */);
let traj = sys.stormer_verlet(0.01, 10000);

// Energy deviation from initial: H(t) - H(0) for every step
let drain = energy_drain(&sys, &traj);
// For Verlet: bounded oscillation ~O(dt²)
// For Euler: monotonic growth

// Phase-space volume ratio (final/initial) — should be ~1.0 for symplectic
let vol = liouville_volume(&sys, &traj);
println!("Volume preservation: {:.6}", vol);  // ≈ 1.000
```

## Multi-Dimensional Systems

All integrators handle arbitrary dimensions. A 2-particle system in 3D has a 12-dimensional phase space (2 × 3 positions + 2 × 3 momenta):

```rust
// Two coupled oscillators in R² (4D phase space per oscillator)
let h = vec![
    // q₁  q₂  p₁  p₂
    vec![2.0, 0.5, 0.0, 0.0],  // q₁ kinetic + coupling
    vec![0.5, 1.0, 0.0, 0.0],  // q₂ kinetic + coupling
    vec![0.0, 0.0, 3.0, 0.0],  // p₁ potential
    vec![0.0, 0.0, 0.0, 1.5],  // p₂ potential
];
let q0 = vec![1.0, 0.5];
let p0 = vec![0.0, 1.0];
let sys = HamiltonianSystem::new(h, q0, p0);

let traj = sys.stormer_verlet(0.001, 10000);
// After ~6283 steps (≈ 2π), system nearly returns to initial state
```

## Connection to Spin-as-Time

The Hamiltonian framework here — phase space, symplectic structure, conservation laws — is the same language used to describe quantum spin when time is reinterpreted as a spin variable. In that picture:

- The Hamiltonian becomes a transfer matrix
- The symplectic form encodes the uncertainty principle
- Conservation of energy maps to unitarity of the transfer matrix
- The Störmer-Verlet scheme becomes a Trotter decomposition of the propagator

This library provides the classical foundation. The quantum correspondence is exact for quadratic Hamiltonians (Gaussian states).

## API Summary

| Type | Description |
|------|-------------|
| `SymplecticForm` | Canonical symplectic structure ω on R^{2n} |
| `SymplecticMatrix` | A matrix M satisfying MᵀJM = J |
| `HamiltonianSystem` | System defined by quadratic H(q,p) with integrators |
| `PhasePoint` | A point (q, p) in phase space |
| `Trajectory` | Time series of phase-space points |

| Function | Description |
|----------|-------------|
| `energy_drain` | H(t) − H(0) over a trajectory |
| `symplectic_error` | Phase-space area preservation error |
| `liouville_volume` | Phase-space volume ratio (1.0 = perfect) |

| Method on `HamiltonianSystem` | Order | Symplectic? | Time-reversible? |
|-------------------------------|-------|-------------|------------------|
| `euler` | 1 | ✗ | ✗ |
| `symplectic_euler` | 1 | ✓ | ✗ |
| `stormer_verlet` / `leapfrog` | 2 | ✓ | ✓ |

## Limitations

- **Quadratic Hamiltonians only.** The Hamiltonian is encoded as a matrix H with H(x) = ½xᵀHx. Non-quadratic potentials (Lennard-Jones, Kepler) require a different approach — typically symplectic splitting (separating kinetic and potential energy).
- **Dense matrices.** No sparse representations. Fine for small-to-medium systems (up to ~100 DOF). For N-body simulations with N > 100, you'd want sparse linear algebra.
- **No adaptive step size.** The step size `dt` is fixed. Adaptive symplectic integrators exist but are significantly more complex.
- **No higher-order methods.** Fourth-order symplectic integrators (Yoshida, Forest-Ruth) can be built by composing Verlet steps. This library doesn't include them, but the infrastructure supports it.
- **f64 precision only.** No generic numerics.

## When to Use This

- **Physics simulations** where energy conservation matters (orbital mechanics, molecular dynamics, spin systems)
- **Teaching** — the code is readable, the tests demonstrate every property
- **Prototyping** — zero dependencies, works everywhere Rust does
- **Benchmarking** — compare your integrator against symplectic baselines

## When Not To

- Stiff systems (use implicit methods instead)
- Dissipative systems (symplecticity helps less when energy isn't conserved)
- Large-scale N-body (use a production MD library like GROMACS or HOOMD)

## Installation

```toml
[dependencies]
symplectic-geometry = "0.1"
```

## License

MIT

## Ecosystem Integration

- Foundation for Hamiltonian and symplectic structure analysis in the SuperInstance ecosystem
- Used by `symplectic-spin` for spin-system symplectic integrators
- Feeds `constraint-hamiltonian` for constrained Hamiltonian dynamics
- Integrates with `spectral-mechanics` for spectral-symplectic hybrid methods
- Enables geometric mechanics approaches across the ecosystem

