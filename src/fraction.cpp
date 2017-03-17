#include "fraction.h"

namespace solver {
Fraction::Fraction() {}
Fraction::Fraction(int numerator) : numerator(numerator), denominator(1) {}
Fraction::Fraction(int numerator, int denominator) : numerator(numerator), denominator(denominator) {}
Fraction::~Fraction() {}

std::ostream& operator<<(std::ostream& os, const Fraction obj) {
  os << obj.numerator << "/" << obj.denominator;
  return os;
}

Fraction Fraction::operator*(const Fraction& operand) const {
  Fraction result;
  result.numerator = this->numerator * operand.numerator;
  result.denominator = this->denominator * operand.denominator;
  Fraction::reduce(result);
  return result;
}

Fraction Fraction::operator/(const Fraction& operand) const {
  Fraction result;
  result.numerator = this->numerator * operand.denominator;
  result.denominator = this->denominator * operand.numerator;
  Fraction::reduce(result);
  return result;
}

Fraction operator/(int operand1, const Fraction& operand2) {
  Fraction fractionOperand1(operand1);
  Fraction result = operand1/operand2;
  return result;
}

Fraction Fraction::operator+(const Fraction& operand) const {
  Fraction result;
  result.numerator = this->numerator * operand.denominator + operand.numerator * this->denominator;
  result.denominator = this->denominator * operand.denominator;
  Fraction::reduce(result);
  return result;
}

Fraction Fraction::operator-(const Fraction& operand) const {
  Fraction result;
  result.numerator = this->numerator * operand.denominator - operand.numerator * this->denominator;
  result.denominator = this->denominator * operand.denominator;
  Fraction::reduce(result);
  return result;
}

Fraction Fraction::operator-() const {
  Fraction result;
  result.numerator = -this->numerator;
  result.denominator = this->denominator;
  Fraction::reduce(result);
  return result;
}

Fraction Fraction::operator+=(const Fraction& operand) const {
  Fraction result;
  result.numerator = this->numerator * operand.denominator + operand.numerator * this->denominator;
  result.denominator = this->denominator * operand.denominator;
  Fraction::reduce(result);
  return result;
}

Fraction Fraction::operator-=(const Fraction& operand) const {
  Fraction result;
  result.numerator = this->numerator * operand.denominator - operand.numerator * this->denominator;
  result.denominator = this->denominator * operand.denominator;
  Fraction::reduce(result);
  return result;
}

bool Fraction::operator<(const Fraction& operand) const {
  int leftNumerator = this->numerator * operand.denominator;
  int rightNumerator = operand.numerator * this->denominator;
  return leftNumerator < rightNumerator;
}

bool Fraction::operator>(const Fraction& operand) const {
  int leftNumerator = this->numerator * operand.denominator;
  int rightNumerator = operand.numerator * this->denominator;
  return leftNumerator > rightNumerator;
}

bool Fraction::operator!=(const int operand) const {
  return this->numerator != this->denominator * operand;
}

bool Fraction::operator==(const int operand) const {
  return this->numerator == this->denominator * operand;
}

void Fraction::reduce(Fraction& fraction) const {
  int gcd = Fraction::gcd(fraction.numerator, fraction.denominator);
  fraction.numerator = fraction.numerator / gcd;
  fraction.denominator = fraction.denominator / gcd;
}

int Fraction::gcd(int num1, int num2) const {
  if (num1 < num2) {
    int tmp = num1;
    num2 = num1;
    num2 = tmp;
  }

  if (num2 == 0) {
    return num1;
  } else {
    return Fraction::gcd(num2, num1 % num2);
  }
}

}
