# SMTLIB2 Format Translation Design Document

## Peticodiac Input Format
NUM_VARS defines the number of non-basic variables, or columns in our tableau.
NUM_CONSTRS defines the number of basic variables, or rows in our tableau.

The input is a set of constraint vectors each of size NUM_VARS for a
NUM_VARS by NUM_CONSTRS size tableau coefficient matrix A and
a set of bounds input specified by index, lower bound, and upper bound of
size NUM_CONSTRS.

```
/* Example of constraint and bounds */

NUM_VARS is set to 2
NUM_CONSTRS is set to 1
The vector size for add_constraint() must match 2 since NUM_VARS is 2
The set_bound() must be invoked once since NUM_CONSTRS is 1

// 2x0 + 3x1
solver->add_constraint( { 2, 3 } );

// Set bounds, start with index 2 because we have 2 non-basic variables
// The lower bound is 2, and no upper bound
solver->set_bounds(2, 2, NO_BOUND);
```

The vector of decision variables X is constrained to be non-negative
real number.

## SMTLIB2 Language Spec Highlight
1. Solver initialization: `set-logic`
  - Initialize the solver with a specified logic.
  - If the solver supports the logic, it should return `success`.
  - Otherwise, it returns `unsupported` if the solver doesn't support the logic, or `error` if non-symbol is passed in as an argument.
  - Only one `set-logic` in a given execution.
  - Input form: `set-logic <symbol>`
  - In our case, we use the `QF_LRA` symbol (Quantifier-free Linear Real Arithmetic).

2. Termination: `exit`
  - Solver should always return `success` and terminates.
  - Only one `exit` in a given execution.

3. Defining new sort: `declare-sort` or `define-sort`
  - Declare a new sort.
  - Usage not found in the QF_LRA benchmarks.

4. Defining new function symbols and constants: `declare-fun` or `define-fun`
  - Declare a new symbol. All symbols must be declared before use.
  - Constant is a special case of function with no argument
  - Input form: `(declare-fun <symbol> ( <sort-expr>* ) <sort-expr>)`
  - Input form: `(define-fun <symbol> ( ( <symbol> <sort-expr>)* ) <sort-expr> <expr>)`
  - In our case, we can use `(declare-fun x0 () Real)` to declare a floating non-basic variable

5. Asserting logical statements: `assert`
  - Assert a formula is true
  - Command should not appear before a `set-logic` since the logic defines the meaning of sort
  - Input form: `(assert <expr>)`
  - The expression's sort is Bool
  - In our case, we can use `(assert (= (+ (* 2 x1) (* -1 x2)) s1))` to add constraint and `(assert (> s1 3))` to add bound

6. Check satisfiability: `check-sat`
  - Tell the solver to check for satisfiability.
  - May return `sat`, `unsat`, or `unknown`

7. Sat operations: `get-value` and `get-assignment`
  - When `check-sat` produces a `sat` response, we can ask for the value of a specific value or a list of assignment that made the problem satisfiable.
  - To use `get-value`, the `:produce-models` option will be set before `set-logic` in the script
  - To use `get-assignment`, the `:produce-assignments` option will be set before `set-logic` in the script
  - Usage not found in the QF_LRA benchmarks.

8. Unsat operations: `get-proof` and `get-unsat-core`
  - To use `get-proof`, the `:produce-proofs` option must be set before `set-logic` in the script.
  - To use `get-unsat-core`, the `:produce-unsat-cores` option must be set before `set-logic` in the script.
  - Usage not found in the QF_LRA benchmarks.

9. The formula given is in the linear arithmetic form `Φ`.
  - We want to convert it into a equisatisfiable formula following [1] for make the computation simpler

## SMTLIB2 vs. Peticodiac


## Design Proposal
### Input Format
- Create our own file-based input format similar to the DIMACS format for SAT solver
```
Example of a possible input format
#
# Comment sections
#
p cnf 2 1
c 2 3
b 2 2 NO_BOUND
```
  - The file can start with comments with lines beginning with `#`
  - The line `p cnf NUM_VARS NUM_CONSTRS` indicating that the instance is in conjunction form, with the given number of constraints and bounds
  - A line starts with `c` indicates a constraint coefficient input. The number of lines that contain the letter `c` should be equal to NUM_CONSTRS and the number of subsequent value count should equal to NUM_VARS
  - A line starts with `b` indicates a bound input. The number of lines that contain the letter `b` should be equal to NUM_CONSTRS

- Develop a preprocessor that converts SMT-LIB scripts into an intermediate file using our input format
- Execute the solver against our input file

### From SMTLIB2 to Peticodiac input format
- The jSMTLIB is an open source java-based parser for the SMTLIB2 Language
- It can act as the frontend for a non-SMTLIB2-native SMT solver by converting the SMTLIB2 script to the solver's native input using an adapter
- We can create a Java-based peticodiac adapter that conforms to the jSMTLIB Solver_test and ISolver interfaces
- Key functions to implement
  - declare_fun(): declares variables
  - assertExpr(): creates constraints and bounds
  - output(): outputs a file containing our SMT solver input format

## Pseudo Code

[1] Bruno Dutertre and Leonardo de Moura, A Fast Linear-Arithmetic Solver for DPLL(T)*
