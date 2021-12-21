#include <iostream>

// factorial.hs:
//     fac 0 = 1
//     fac n = n * fac (n-1)
//     main = print (fac 6)
// ghc -o factorial factorial.hs
template<int n> struct
fact {
    static const int value = n * fact<n - 1>::value;
};

template<> struct
fact<0> { // specialization for n = 0
    static const int value = 1;
};

// predicate.hs:
//     is_zero 0 = True
//     is_zero x = False
// ghc -o predicate predicate.hs
template<typename T> struct
isPtr {
  static const int value = false;
};

template<typename U> struct
isPtr<U*> {
  static const int value = true;
};

template<typename U> struct
isPtr<U* const> {
    static const bool value = true;
};

template<typename U> struct
isPtr<const U*> {
    static const bool value = false;
};

// count_list.hs
//     l = [1,2,3]
//     count [] = 0
//     count (head:tail) = 1 + count tail
//     main = print (count l)
// ghc count_list.hs
// Just a declaration
template<typename... list> struct
count;

template<> struct
count<> {
  static const int value = 0;
};

template<typename head, typename... tail> struct
count<head, tail...> {
  // "typename... tail" defines a template parameter pack.
  // The expansion is done by following the name of the pack with three dots,
  // as in tail….
  static const int value = 1 + count<tail...>::value;
};

// or_combinator.hs
//     or_combinator f1 f2 = λ x -> (f1 x) || (f2 x)
//     (or_combinator is_zero is_one) 2
// ghc or_combinator.hs
template<template<typename> class f1, template<typename> class f2> struct
or_combinator {
    template<typename T> struct
    lambda {
        static const bool value = f1<T>::value || f2<T>::value;
    };
};

template<typename T> struct
isConst {
  static const int value = false;
};

template<typename U> struct
isConst<const U> {
  static const int value = true;
};

// all.hs
//     all pred [] = True
//     all pred (head:tail) = (pred head) && (all pred tail)
// ghs all.hs
template<template<typename> class predicate, typename... list> struct
all;

template<template<typename> class predicate> struct
all<predicate> {
  static const int value = true;
};

template<template<typename> class predicate, typename head, typename... tail> struct
all<predicate, head, tail...> {
  static const int value = predicate<head>::value && all<predicate, tail...>::value;
};

// foldr.hs
//     foldr f init [] init
//     foldr f init (head:tail) =
//         f head (foldr f init tail)
template<template<typename, int> typename, int, typename...> struct
fold_right;

template<template<typename, int> typename f, int init> struct
fold_right<f, init> {
  static const int value = init;
};

template<template<typename, int> typename f, int init, typename head, typename...tail> struct
fold_right<f, init, head, tail...> {
  static const int value = f<head, fold_right<f, init, tail...>::value>::value;
};

template<typename head, int right_value> struct
add_it {
  static const int value = right_value + 1;
};

// sum.hs
//   sum [] = 0
//   sum (head:tail) = head + (sum tail)
template<int...> struct
sum;

template<> struct
sum<> {
  static const int value = 0;
};

template<int head, int... tail> struct
sum<head, tail...> {
  static const int value = head + sum<tail...>::value;
};

// list_comp.hs
//  count lst = sum [1 | x <- lst]
template<typename T> struct
one {
  static const int value = 1;
};

template<typename... lst> struct
count {
  /*
   * E.g., lst = [int, char, void*], one<lst>::value... will be expanded to
   * [one<int>::value, one<char>::value, one<void*>::value]. The subsequent
   * call to sum would be made with those arguments. The ellipsis in C++
   * follows a pattern that contains a pack, it's not the pack that's
   * expanded, but the whole pattern is repeated for each element of the
   * pack.
   */
  static const int value = sum<one<lst>::value...>::value;
};

template<typename... lst> struct
countPtrs {
  static const int value = sum<isPtr<lst>::value ...>::value;
};

int main() {
  std::cout << "[factorial.hs] Factorial of 6 = " << fact<6>::value << "\n";
  std::cout << "[predicate.hs] isPtr<int> = " << isPtr<int>::value
            << "; isPtr<char *> = " << isPtr<char *>::value
            << "; isPtr<float * const> = " << isPtr<float * const>::value
            << "; isPtr<float * const> = " << isPtr<const float *>::value
            << "\n";
  std::cout << "[count_list.hs] len of the list of types [int, char, long] = "
            << count<int, char, long>::value << "\n";
  std::cout << "[or_combinator.hs] or_combinator<isPtr, isConst>::lambda<const int>::value = "
            << or_combinator<isPtr, isConst>::lambda<int>::value << ","
            << or_combinator<isPtr, isConst>::lambda<const int>::value << ","
            << or_combinator<isPtr, isConst>::lambda<int *>::value << ","
            << std::endl;
  std::cout << "[all.hs] all<isPtr, int*, char*, float*>::value = "
	    << all<isPtr, int*, char*, float*>::value << "; "
	    << all<isConst, const int*, const char*, float*>::value << std::endl;
  std::cout << "[foldr.hs] fold_right<f, 0, int, float, double, char, void>::value = "
	    << fold_right<add_it, 0, int, float, double, char, void>::value << std::endl;
  std::cout << "[foldr.hs] fold_right<f, 0, int, float, double, char, void>::value = "
	    << fold_right<add_it, 0, int, float, double, char, void>::value << std::endl;
  std::cout << "[sum.hs] sum<1,2,3,4,5,6,7,8,9>::value = "
	    << sum<1,2,3,4,5,6,7,8,9>::value << std::endl;
  std::cout << "[list_comp.hs] count<int, float, double, char, void>::value = "
	    << count<int, float, double, char, void>::value << std::endl;
  std::cout << "[list_comp.hs] countPtrs<int*, char*, void*, short*, float*, double*, bool>::value = "
	    << countPtrs<int*, char*, void*, short*, float*, double*, bool>::value << std::endl;
  return 0;
}
