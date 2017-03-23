#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <omp.h>
#include <string.h>

#include "CpuSolver.h"
#include "CpuEagerSolver.h"

#ifdef CUDA_ENABLED
#include "CudaSolver.h"
#endif

#include "random.h"
#include "timing.h"

using solver::Solver;
using solver::SolverType;
#ifdef PROFILER_ENABLED
using solver::SolverProfiler;
#endif

template <typename T>
void print_solution(Solver<T> &s) {
  const std::vector<T> solution = s.solution();
  for (int i = 0; i < int(solution.size()); ++i) {
    // printf("x%d=%.3f%s", i, solution[i], i + 1 < int(solution.size()) ? " " : "\n");
    std::cout << "x" << i << "=" << solution[i];
    if ((i + 1) < int(solution.size())) {
      std::cout << "  ";
    } else {
      std::cout << endl;
    }
  }
}

std::vector<std::string> split(const std::string &s, char delim = ' ') {
  std::vector<std::string> tokens;
  for (int i = 0; i < int(s.length()); ++i) {
    std::string current_string;
    while (s[i] != delim && s[i] != '\n' && i < int(s.length())) {
      current_string += s[i];
      i++;
    }
    tokens.push_back(current_string);
  }
  return tokens;
}

template <typename T>
void execute(Solver<T>* const solver) {
printf("\n\n## Start Execute ##\n\n");
#ifdef DEBUG
  printf("#### Variables ####\n");
  solver->print_variables();
  printf("#### Tableau ####\n");
  solver->print_tableau();
  printf("\n");
#endif

  double solver_time = -cpu_second();
  bool solver_val = solver->solve();
  solver_time += cpu_second();
  bool solution_verified = solver->verify_solution();
  int solver_steps = solver->get_step_count();

  printf("\n################ Summary ################\n");
  printf("CPU time              : %.3f seconds\n", solver_time);
  printf("CPU result            : %s\n", solver_val ? "SAT" : "UNSAT");
  printf("CPU steps             : %d\n", solver_steps);
  printf("CPU steps per second  : %.1f\n", solver_steps / solver_time);
  printf("\n");
  if (solver_val) {
    if (solution_verified) {
      printf("SAT Solution:\n");
    } else {
      printf("Erroneous SAT Solution:\n");
    }
    print_solution<T>(*solver);
    printf("\n");
  }
}

template <typename T>
void start_solver_test(const SolverType type, const int num_var, const int num_constr) {
  printf("## Start Solver ##\n");
  Solver<T>* solver = nullptr;
  int num_vars = 2;
  int num_constrs = 2;
  solver = Solver<T>::create(type, num_vars, num_constrs);

  solver::Fraction constraint1(3);
  solver::Fraction constraint2(2);
  std::vector<T> constr = {constraint1, constraint2};
  solver->add_constraint(constr);
  solver::Fraction lower(10);
  solver::Fraction upper(40);
  const T low = lower;
  const T upp = upper;
  std::cout << "    The lower bound = " << low << " and upper bound = " << upp << endl;
  //printf("The lower bound = %f and upper bound = %f\n", low, upp);
  solver->set_bounds(2, low, upp);

  solver::Fraction constraint3(6);
  solver::Fraction constraint4(8);
  std::vector<T> constr2 = {constraint3, constraint4};
  solver->add_constraint(constr2);
  solver::Fraction lower2(NO_BOUND);
  solver::Fraction upper2(60);
  const T low2 = lower2;
  const T upp2 = upper2;
  std::cout << "    The lower bound = " << low2 << " and upper bound = " << upp2 << endl;
  //printf("The lower bound = %f and upper bound = %f\n", low, upp);
  solver->set_bounds(3, low2, upp2);

  execute<T>(solver);
  delete solver;
}

// template <typename T>
// void start_solver_test(const SolverType type, const int num_var, const int num_constr) {
//   printf("In start_solver");
//   Solver<T>* solver = nullptr;
//   int num_vars = 1;
//   int num_constrs = 1;
//   solver = Solver<T>::create(type, num_vars, num_constrs);
//
//   std::vector<T> constr = {5.0};
//   solver->add_constraint(constr);
//   const T low = 5.0f;
//   const T upp = 5.0f;
//   std::cout << "The lower bound = " << low << " and upper bound = " << upp << endl;
//   //printf("The lower bound = %f and upper bound = %f\n", low, upp);
//   solver->set_bounds(num_vars, low, upp);
//   execute<T>(solver);
//   delete solver;
// }

template <typename T>
void start_solver(const SolverType type, const int num_vars, const int num_constrs) {
  Solver<T>* solver = nullptr;
  solver = Solver<T>::create(type, num_vars, num_constrs);

  for (int i = 0; i < num_constrs; ++i) {
    std::vector<T> constr = ::random_instance(num_vars, 0, 11);
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

  execute<T>(solver);
  delete solver;
}

template <typename T>
void start_solver(const SolverType type, char const *input_file) {
  Solver<T>* solver = nullptr;

  std::string line;
  std::ifstream peticodiac_file;
  peticodiac_file.open(input_file);
  if (peticodiac_file.is_open()) {
    while (getline(peticodiac_file, line)) {
      if (line != "" && strcmp(line.substr(0, 1).c_str(), "#") != 0 ) {
        std::vector<std::string> expression_line = split(line);

        if (expression_line[0].compare("p") == 0) {
          // This is the header, create the solver
#ifdef DEBUG
          printf("#### Create solver with %s vars and %s bounds\n", expression_line[2].c_str(), expression_line[3].c_str());
#endif
          solver = Solver<T>::create(type, std::stoi(expression_line[2]), std::stoi(expression_line[3]));
        } else if (expression_line[0].compare("c") == 0) {
          // This is the constraint, add constraint as a vector<float>
#ifdef DEBUG
          printf("#### Add constraint %s\n", line.c_str());
#endif
          std::vector<T> coefficient;
          for (int i = 1; i < int(expression_line.size()); ++i) {
            coefficient.push_back(std::stof(expression_line[i]));
          }
          solver->add_constraint(coefficient);
        } else if (expression_line[0].compare("b") == 0) {
          // This is the bound, set_bounds with index, lower bound, and upper bound
          int index = std::stoi(expression_line[1]);
          std::vector<std::string> lower = split(expression_line[2], ':');
          float lower_bound = lower.size() == 1 ? NO_BOUND : std::stof(lower[1]);
          std::vector<std::string> upper = split(expression_line[3], ':');
          float upper_bound = upper.size() == 1 ? NO_BOUND : std::stof(upper[1]);
#ifdef DEBUG
          printf("#### Set bound %s\n", line.c_str());
          printf("The lower bound = %f and upper bound = %f\n", lower_bound, upper_bound);
#endif
          solver->set_bounds(index, lower_bound, upper_bound);
        } else if (line.compare("eoa") == 0) {
          // Start the solver
#ifdef DEBUG
          printf("#### End of assertion: start solver\n");
#endif
          execute<T>(solver);
          delete solver;
          solver = nullptr;
        }
      }
    }
  }

  if (solver != nullptr) {
    delete solver;
  }
}

int main(const int argc, const char** argv) {
  if (argc == 1 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
    char const *usage = "Usage: peticodiac --file input_file --solver SOLVER_NAME\n" \
        "       peticodiac --num_vars INT --num_constraints INT --solver SOLVER_NAME\n\n" \
        "       --file, -f              Specify the input expression file (in peticodiac format) for the solver to solve\n" \
        "       --num_constraints, -nc  Specify the number of initial basic variables to be created\n" \
        "       --num_variables, -nv    Specify the number of initial non-basic variables to be created\n" \
        "       --solver, -s            Specify the solver type\n" \
        "                               1: CPU_EAGER\n" \
        "                               2: CPU_LAZY\n" \
        "                               3: CUDA\n";
        printf(usage);
        exit(EXIT_SUCCESS);
  }

  int num_vars;
  int num_constrs;
  int solver_type;
  char const *input_file;
  SolverType type;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--file") == 0 || strcmp(argv[i], "-f") == 0) {
      input_file = argv[++i];
    } else if (strcmp(argv[i], "--num_constraints") == 0 || strcmp(argv[i], "-nc") == 0) {
      num_constrs = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--num_variables") == 0 || strcmp(argv[i], "-nv") == 0) {
      num_vars = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--solver") == 0 || strcmp(argv[i], "-s") == 0) {
      solver_type = atoi(argv[++i]);
    }
  }

  if (solver_type == 1) {
    type = SolverType::CPU_EAGER;
  } else if (solver_type == 2) {
    type = SolverType::CPU_LAZY;
  } else if (solver_type == 3) {
    type = SolverType::CUDA;
  }

  if (input_file) {
    start_solver<float>(type, input_file);
  } else {
    //start_solver(type, num_vars, num_constrs);
    //start_solver_test<float>(type, num_vars, num_constrs);
    start_solver_test<solver::Fraction>(type, num_vars, num_constrs);
  }

  exit(EXIT_SUCCESS);
}
