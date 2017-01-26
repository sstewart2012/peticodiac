#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <omp.h>
#include <string.h>

#include "AbstractSolver.h"
#include "random.h"
#include "timing.h"

using solver::Solver;
using solver::SolverType;
#ifdef PROFILER_ENABLED
using solver::SolverProfiler;
#endif

void print_solution(const Solver &s) {
  const std::vector<float> solution = s.solution();
  for (int i = 0; i < solution.size(); ++i) {
    printf("x%d=%.3f%s", i, solution[i], i + 1 < solution.size() ? " " : "\n");
  }
}

void test_random(Solver* const solver, const int num_vars, const int num_constrs) {
  for (int i = 0; i < num_constrs; ++i) {
    std::vector<float> constr = ::random_instance(num_vars, 0, 11);
    solver->add_constraint(constr);
    const float* const bounds = ::random_float(2, 1, 11);
    const float low = std::min(bounds[0], bounds[1]);
    const float upp = std::max(bounds[0], bounds[1]);
    if (low == upp) {
      solver->set_bounds(i + num_vars, low, upp + 1);
    } else {
      solver->set_bounds(i + num_vars, low, upp);
    }
  }

#ifdef DEBUG
  solver->print_variables();
  printf("\n");
  solver->print_tableau();
  printf("\n");
#endif

  double solver_time = -cpu_second();
  bool solver_val = solver->solve();
  solver_time += cpu_second();
  int solver_steps = solver->get_step_count();

  printf("CPU time              : %.3f seconds\n", solver_time);
  printf("CPU result            : %s\n", solver_val ? "SAT" : "UNSAT");
  printf("CPU steps             : %d\n", solver_steps);
  printf("CPU steps per second  : %.1f\n", solver_steps / solver_time);
  printf("\n");
}

std::vector<std::string> split(const std::string &s, char delim = ' ') {
  std::vector<std::string> tokens;
  for (int i = 0; i < s.length(); ++i) {
    std::string current_string;
    while (s[i] != delim && s[i] != '\n' && i < s.length()) {
      current_string += s[i];
      i++;
    }
    tokens.push_back(current_string);
  }
  return tokens;
}

void test_solver(const SolverType type, const int num_vars, const int num_constrs) {
  Solver* solver;

  std::string line;
  std::ifstream peticodiac_file;
  peticodiac_file.open("temp.smt2.peticodiac");
  if (peticodiac_file.is_open()) {
    while (getline(peticodiac_file, line)) {
      if (strcmp(line.substr(0, 1).c_str(), "#") != 0 ) {
        std::vector<std::string> expression_line = split(line);

        if (expression_line[0].compare("p") == 0) {
          // This is the header, create the solver
          printf("Create solver with %s vars and %s bounds\n", expression_line[2].c_str(), expression_line[3].c_str());
          solver = Solver::create(type, std::stoi(expression_line[2]), std::stoi(expression_line[3]));
        } else if (expression_line[0].compare("c") == 0) {
          // This is the constraint, add constraint as a vector<float>
          printf("Add constraint %s\n", line.c_str());
          std::vector<float> coefficient;
          for (int i = 1; i < expression_line.size(); ++i) {
            coefficient.push_back(std::stof(expression_line[i]));
          }
          solver->add_constraint(coefficient);
        } else if (expression_line[0].compare("b") == 0) {
          // This is the bound, set_bounds with index, lower bound, and upper bound
          printf("Set bound %s\n", line.c_str());
          int index = std::stoi(expression_line[1]);
          std::vector<std::string> lower = split(expression_line[2], ':');
          float lower_bound = lower.size() == 1 ? NO_BOUND : std::stof(lower[1]);
          std::vector<std::string> upper = split(expression_line[3], ':');
          float upper_bound = upper.size() == 1 ? NO_BOUND : std::stof(upper[1]);
          solver->set_bounds(index, lower_bound, upper_bound);
        }
      }
    }
  }

  double time = -cpu_second();
  bool val = solver->solve();
  time += cpu_second();

  if (val) {
    printf("SAT (%.3f seconds)\n", time);
    print_solution(*solver);
  } else {
    printf("UNSAT (%.3f seconds)\n", time);
  }
  delete solver;
}

int main(const int argc, const char** argv) {
  const int num_vars = atoi(argv[1]);
  const int num_constrs = atoi(argv[2]);
  const int solver_type = atoi(argv[3]);

  Solver *solver = nullptr;
  if (solver_type == 1)
    solver = Solver::create(SolverType::CPU_EAGER, num_vars, num_constrs);
  else if (solver_type == 2)
    solver = Solver::create(SolverType::CPU_LAZY, num_vars, num_constrs);
  else if (solver_type == 3)
    solver = Solver::create(SolverType::CUDA, num_vars, num_constrs);

  if (solver != nullptr) {
    //test_random(solver, num_vars, num_constrs);
    test_solver(SolverType::CPU_EAGER, num_vars, num_constrs);
    delete solver;
  }

  exit(EXIT_SUCCESS);
}
