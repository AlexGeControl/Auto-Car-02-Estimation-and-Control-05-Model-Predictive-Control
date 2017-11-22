#include "MPC.h"

#include <limits>
#include <cmath>
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"

using CppAD::AD;

// Set reference velocity:
double ref_v = 40;

// Set the timestep length and duration
size_t  N = 40;
double dt = 0.025;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// The solver takes all the state variables and actuator variables in a singular vector.
// Thus, we should to establish when one variable starts and another ends to make our lifes easier.
size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
size_t a_start = delta_start + N - 1;

/**
 * Evaluate a polynomial using Horner's method
 * @param coeffs: Polynomial coefficients
 * @param x: Position x of evaluation
 */
AD<double> polyeval(
  const Eigen::VectorXd coeffs,
  const AD<double> &x
) {
  AD<double> result = 0.0;

  for (int i = 0; i < coeffs.size(); ++i) {
    result += coeffs(i)*CppAD::pow(x, i);
  }

  return result;
}

class FG_eval {
public:
  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;

  // Constructor:
  FG_eval(Eigen::VectorXd coeffs) {
    // Polynomial coefficients:
    (*this).coeffs = coeffs;
    // Polynomial derivative coefficients:
    coeffsDiff = Eigen::VectorXd(coeffs.size() - 1);
    for (int i = 0; i < coeffsDiff.size(); ++i) {
      coeffsDiff(i) = (i + 1)*coeffs(i + 1);
    }
  }

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

    //
    // Setup Constraints
    //
    // a. Initial state:
    fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + psi_start] = vars[psi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];
    // b. Future states through global kinematic model:
    for (size_t i = 1; i < N; ++i) {
      // Current state:
      AD<double>    x_curr = vars[   x_start + i];
      AD<double>    y_curr = vars[   y_start + i];
      AD<double>  psi_curr = vars[ psi_start + i];
      AD<double>    v_curr = vars[   v_start + i];
      AD<double>  cte_curr = vars[ cte_start + i];
      AD<double> epsi_curr = vars[epsi_start + i];

      // Previous state:
      AD<double>    x_prev = vars[   x_start + i - 1];
      AD<double>    y_prev = vars[   y_start + i - 1];
      AD<double>  psi_prev = vars[ psi_start + i - 1];
      AD<double>  cte_prev = vars[ cte_start + i - 1];
      AD<double>    v_prev = vars[   v_start + i - 1];
      AD<double> epsi_prev = vars[epsi_start + i - 1];

      // Previous control:
      AD<double> str_prev = vars[delta_start + i - 1];
      AD<double> acc_prev = vars[    a_start + i - 1];

      // Previous ref:
      AD<double> ref_y_prev = polyeval(coeffs, x_prev);
      AD<double> ref_psi_prev = CppAD::atan(polyeval(coeffsDiff, x_prev));

      // Constraints from global kinematic model:
      fg[   1 + x_start + i] = x_curr - (x_prev + v_prev*CppAD::cos(psi_prev)*dt);
      fg[   1 + y_start + i] = y_curr - (y_prev + v_prev*CppAD::sin(psi_prev)*dt);
      fg[ 1 + psi_start + i] = psi_curr - (psi_prev + v_prev/Lf*str_prev*dt);
      fg[   1 + v_start + i] = v_curr - (v_prev + acc_prev*dt);
      fg[ 1 + cte_start + i] = cte_curr - (ref_y_prev - y_prev + v_prev*CppAD::sin(epsi_prev)*dt);
      fg[1 + epsi_start + i] = epsi_curr - (epsi_prev + v_prev/Lf*str_prev*dt - ref_psi_prev);
    }
  }

private:
  // Polynomial coefficients
  Eigen::VectorXd coeffs;
  // Polynomial derivative coefficients:
  Eigen::VectorXd coeffsDiff;
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(
  Eigen::VectorXd state,
  Eigen::VectorXd coeffs,
  double delay
) {
  typedef CPPAD_TESTVECTOR(double) Dvector;

  // number of independent variables
  // N timesteps == N - 1 actuations
  size_t n_vars = N * 6 + (N - 1) * 2;
  // Number of constraints
  size_t n_constraints = N * 6;

  // Variables.
  Dvector vars(n_vars);
  // a. Future states & controls:
  for (size_t i = 0; i < n_vars; i++) {
    vars[i] = 0.0;
  }
  // b. Initial state:
  double xInit = state(0);
  double yInit = state(1);
  double psiInit = state(2);
  double vInit = state(3);
  double cteInit = state(4);
  double epsiInit = state(5);
  double strInit = state(6);
  double accInit = state(7);
  // c. Initial state after delay:
  double x = 0.0, y = 0.0, psi = 0.0, v = 0.0, cte = 0.0, epsi = 0.0;

  const int num_extrapolation  = 2*int(delay / dt);
  delay /= num_extrapolation;

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

  vars[x_start] = x;
  vars[y_start] = y;
  vars[psi_start] = psi;
  vars[v_start] = v;
  vars[cte_start] = cte;
  vars[epsi_start] = epsi;

  // Lower and upper bounds of variables:
  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  // a. Free variables:
  for (size_t i = 0; i < delta_start; i++) {
    vars_lowerbound[i] = -std::numeric_limits<double>::max();
    vars_upperbound[i] = +std::numeric_limits<double>::max();
  }
  // b. Steerings:
  // The upper and lower limits of delta are set to -25 and 25 degrees (values in radians).
  for (size_t i = delta_start; i < a_start; i++) {
    vars_lowerbound[i] = -0.436332;
    vars_upperbound[i] = +0.436332;
  }

  // c. Throttle & brake:
  // The upper and lower limits are set to -1.0 and +1.0.
  for (size_t i = a_start; i < n_vars; i++) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] = +1.0;
  }

  // Lower and upper bounds of constraints
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  // a. Equal constraits--global kinematic model
  for (size_t i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }
  // b. Initial state:
  constraints_lowerbound[x_start] = x;
  constraints_upperbound[x_start] = x;

  constraints_lowerbound[y_start] = y;
  constraints_upperbound[y_start] = y;

  constraints_lowerbound[psi_start] = psi;
  constraints_upperbound[psi_start] = psi;

  constraints_lowerbound[v_start] = v;
  constraints_upperbound[v_start] = v;

  constraints_lowerbound[cte_start] = cte;
  constraints_upperbound[cte_start] = cte;

  constraints_lowerbound[epsi_start] = epsi;
  constraints_upperbound[epsi_start] = epsi;

  // Object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // Options for IPOPT solver
  //
  //
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
    options,
    vars, vars_lowerbound, vars_upperbound,
    constraints_lowerbound, constraints_upperbound,
    fg_eval,
    solution
  );

  //
  // Check some of the solution values
  //
  bool ok = true;
  ok &= (solution.status == CppAD::ipopt::solve_result<Dvector>::success);

  // auto cost = solution.obj_value;
  // std::cout << "Cost " << cost << std::endl;

  vector<double> result;

  // a. Control:
  result.push_back(solution.x[delta_start]);
  result.push_back(solution.x[a_start]);
  // b. Predicted state:
  for (size_t i = 0; i < N; ++i) {
    result.push_back(solution.x[x_start + i]);
    result.push_back(solution.x[y_start + i]);
  }

  return result;
}
