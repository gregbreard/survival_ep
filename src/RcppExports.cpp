// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// survivalEM
List survivalEM(const arma::mat y, const arma::mat x, const int max_iter, bool async);
RcppExport SEXP survivalEP_survivalEM(SEXP ySEXP, SEXP xSEXP, SEXP max_iterSEXP, SEXP asyncSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const arma::mat >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< const int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< bool >::type async(asyncSEXP);
    __result = Rcpp::wrap(survivalEM(y, x, max_iter, async));
    return __result;
END_RCPP
}
