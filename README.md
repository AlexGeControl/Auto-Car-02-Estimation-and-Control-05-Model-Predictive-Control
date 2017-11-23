## README

### Technical Report for Lane Keeping through Model Predictive Control(MPC)

---

<img src="writeup_images/demo.gif" width="100%" alt="Lane Keeping through MPC"/>

The goals of this project are the following:

* Implement the **Model Predictive Controller(MPC)**.
* Tune controller's parameters so as to keep the car in the center of lane.

---

### Implementation

---

#### Model Implementation

In this project **global kinematic model** is used for system modeling.

*State variables* are defined as follows:

| Variable |                  Description                 |
|:--------:|:--------------------------------------------:|
|     x    |                  location x                  |
|     y    |                  location y                  |
|    psi   |                    heading                   |
|     v    |              velocity magnitude              |
|    CTE   |               cross track error              |
|    epsi  | heading error between actual and target pose |

*Control variables* are defined as follows:

| Variable |                               Description                               |
|:--------:|:-----------------------------------------------------------------------:|
|    str   |                        steering angle, in radians                       |
|    acc   | throttle and brake, >0.0 for acceleration and  <0.0 for de-acceleration |

*Update equations* are implemented as in the following code snippet for constraints of MPC model:

```cpp
  // Constraints from global kinematic model:
  fg[   1 + x_start + i] = x_curr - (x_prev + v_prev*CppAD::cos(psi_prev)*dt);
  fg[   1 + y_start + i] = y_curr - (y_prev + v_prev*CppAD::sin(psi_prev)*dt);
  fg[ 1 + psi_start + i] = psi_curr - (psi_prev + v_prev/Lf*str_prev*dt);
  fg[   1 + v_start + i] = v_curr - (v_prev + acc_prev*dt);
  fg[ 1 + cte_start + i] = cte_curr - (ref_y_prev - y_prev + v_prev*CppAD::sin(epsi_prev)*dt);
  fg[1 + epsi_start + i] = epsi_curr - (epsi_prev + v_prev/Lf*str_prev*dt - ref_psi_prev);
```

#### Timestep Length & Prediction Horizon

My strategy for timestep length & prediction horizon selection can be summarized as:

The higher the speed, the shorter both timestep length and prediction horizon.

1. Timestep length should be short. The system model is non-linear in essence. So the global kinematic model, as an approximated linearization, could only be valid within the vincinity of given state. Finer timestep could thus bring reduced approximation error and facilitate the generation of better control strategy.
2. Prediction horizon should also be short. First, short horizon means reduced real-time computing workload, which holds the key to efficient real-time application. Besides, since the global kinematic model is only an approximation, the longer the horizon, the larger the discrepancy between simulated future and actual future. Optimal decision based on this simulated future could turned out to be a disaster for real driving scenario. So it helps to keep prediction horizon short.

I tried the following three timestep & prediction horizon combination:

| Horizon |  Timestep |             Optimal Trajectory             |
|:-------:|:---------:|:------------------------------------------:|
|  5 secs | 0.100 sec | Oscillates when gets away from the vehicle |
|  1 secs | 0.100 sec |   Acceptable when without actuation delay  |
|  1 secs | 0.025 sec |    Acceptable when with actuation delay    |

#### Telemetry Preprocessing

First, the reference trajectory is transformed from global frame to vehicle frame:

```cpp
  /**
   * Transform reference trajectory into vehicle frame:
   */
  const size_t N = ptsx.size();

  Eigen::VectorXd trajX(N);
  Eigen::VectorXd trajY(N);

  // Rotation matrix from vehicle pose:
  const double R[2][2] = {
    {+cos(psi), +sin(psi)},
    {-sin(psi), +cos(psi)}
  };
  
  // Transform reference trajectory into vehicle frame:
  for (size_t i = 0; i < N; ++i) {
    trajX(i) = R[0][0]*(ptsx[i] - px) + R[0][1]*(ptsy[i] - py);
    trajY(i) = R[1][0]*(ptsx[i] - px) + R[1][1]*(ptsy[i] - py);
  }
```

After that, polynomial is fitted:

```cpp
  // Polynomial coefficients of target trajectory, in vehicle frame:
  Eigen::VectorXd coeffs = polyfit(
    trajX,
    trajY
  );
  // Polynomial coefficients of derivate of target trajectory, in vehicle frame:
  Eigen::VectorXd coeffsDiff(coeffs.size() - 1);
  for (int i = 0; i < coeffsDiff.size(); ++i) {
    coeffsDiff(i) = (i + 1)*coeffs(i + 1);
  }
```

Here I modified the implementation of polymonial utilities:

```cpp
/**
 * Evaluate a polynomial using Horner's method
 * @param coeffs: Polynomial coefficients
 * @param x: Position x of evaluation
 */
double polyeval(
  const Eigen::VectorXd &coeffs,
  const double x
) {
  const unsigned int N = coeffs.size();

  double result = 0.0;

  for (int i = N - 1; i >= 0; --i) {
    result *= x;
    result += coeffs(i);
  }

  return result;
}

/**
 * Fit a polynomial.
 * Adapted from https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
 *
 * @param xvals: Position Xs of target trajectory
 * @param yvals: Position Ys of target trajectory
 * @param order: Polynomial degree of target trajectory, default value is 3
 */
Eigen::VectorXd polyfit(
  Eigen::VectorXd xvals,
  Eigen::VectorXd yvals,
  int order = 3
) {
  // Pre-assumption check:
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);

  // Initialize polynomial matrix A:
  Eigen::MatrixXd A(xvals.size(), order + 1);

  // Build polynomial matrix A:
  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }
  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  // Solve coefficients:
  auto Q = A.householderQr();
  auto result = Q.solve(yvals);

  return result;
}
```

#### Actuation Latency

Here I counterbalanced actuation latency through refined extrapolation:

```cpp
  // Initial starting state:
  double xInit = state(0);
  double yInit = state(1);
  double psiInit = state(2);
  double vInit = state(3);
  double cteInit = state(4);
  double epsiInit = state(5);
  double strInit = state(6);
  double accInit = state(7);
  // Initial end state:
  double x = 0.0, y = 0.0, psi = 0.0, v = 0.0, cte = 0.0, epsi = 0.0;

  // Refined timestep for reduced approximation error:
  const int num_extrapolation  = 2*int(delay / dt);
  delay /= num_extrapolation;
  
  // Iterative extrapolation:
  for (int i = 0; i < num_extrapolation; ++i) {
    x = xInit + vInit*cos(psiInit)*delay;
    y = yInit + vInit*sin(psiInit)*delay;
    psi = psiInit + vInit/Lf*strInit*delay;
    v = vInit + accInit*delay;
    cte = cteInit + vInit*sin(epsiInit)*delay;
    epsi = epsiInit + vInit/Lf*strInit*delay;

    xInit = x;
    yInit = y;
    psiInit = psi;
    vInit = v;
    cteInit = cte;
    epsi = epsi;
}
```

For the given start state, it is first extrapolated to reach its actual value after actuation delay time.

The extrapolated state is used for MPC initial value.

Here half-sized timestep is used to achieve reduced approximation error.

---

### Cost Function Heuristics

---

I use the following heuristic, which is summarized from my own test drive experience: **Heavily penalize steering action and only use reasonable throttle & brake actions**. 

In reality this holds the key to successful test drive :) since it helps to reduce the uncomfortable jerks.

```cpp
/**
 * MPC implementation
 * @param fg: Vector containing the cost and constraints.
 * @param vars: Vector containing the variable values (state & actuators).
 */
void operator()(
  ADvector& fg,
  const ADvector& vars
) {
  // The cost is stored is the first element of `fg`.
  // Any additions to the cost should be added to `fg[0]`.
  fg[0] = 0.0;

  // a. Tracking performance:
  for (size_t i = 0; i < N; ++i) {
    // Tracking precision:
    fg[0] += CppAD::pow(vars[cte_start + i], 2);
    fg[0] += CppAD::pow(vars[epsi_start + i], 2);
    // Speed steadiness:
    fg[0] += CppAD::pow(vars[v_start + i] - ref_v, 2);
  }

  // b. Control strength:
  for (size_t i = 0; i < N - 1; ++i) {
    fg[0] += 1200*CppAD::pow(vars[delta_start + i], 2);
    fg[0] += 60*CppAD::pow(vars[a_start + i], 2);
  }

  // c. Control smoothness:
  for (size_t i = 0; i < N - 2; ++i) {
    fg[0] += 800*CppAD::pow(
      vars[delta_start + i + 1] - vars[delta_start + i],
      2
    );
    fg[0] += 40*CppAD::pow(
      vars[a_start + i + 1] - vars[a_start + i],
      2
    );
}
```

For implementation, the above driving strategy could be achieved through tuning of weights of multi-objective cost function.

The above weights are determined through offline analysis of proportion of each objective's cost in total cost.
