#include "test_autodiff_reverse.hpp"
#include <boost/math/special_functions/jacobi.hpp>
#include <cmath>
#include <vector>

BOOST_AUTO_TEST_SUITE(test_stl_supported_functions)

using namespace rdiff;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_acosh, T, all_float_types)
{
    RandomSample<T> rng{1, 10};
    T               x_v         = rng.next();
    T               test_func_v = acosh(x_v);

    rvar<T, 1>      x_rvar      = x_v;

    rvar<T, 1>      x_func      = acosh(x_rvar);

    BOOST_REQUIRE_CLOSE(x_func.item(), test_func_v, boost_close_tol<T>());

    gradient_tape<T, 1>& tape = get_active_tape<T, 1>();

    tape.zero_grad();
    rvar<T, 1> test_func_2    = acosh(x_rvar * x_rvar);
    T          expected_deriv = 2 * x_v / ((sqrt(x_v * x_v - 1) * sqrt(x_v * x_v + 1)));
    test_func_2.backward();

    BOOST_REQUIRE_CLOSE(x_rvar.adjoint(), expected_deriv, boost_close_tol<T>());

    tape.clear();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_asinh, T, all_float_types)
{
    RandomSample<T> rng{1, 10};
    T               x_v         = rng.next();
    T               test_func_v = asinh(x_v);

    rvar<T, 1>      x_rvar      = x_v;

    rvar<T, 1>      x_func      = asinh(x_rvar);

    BOOST_REQUIRE_CLOSE(x_func.item(), test_func_v, boost_close_tol<T>());

    gradient_tape<T, 1>& tape = get_active_tape<T, 1>();

    tape.zero_grad();
    rvar<T, 1> test_func_2    = asinh(x_rvar * x_rvar);
    T          expected_deriv = 2 * x_v / (sqrt(x_v * x_v * x_v * x_v + 1));
    test_func_2.backward();

    BOOST_REQUIRE_CLOSE(x_rvar.adjoint(), expected_deriv, boost_close_tol<T>());

    tape.clear();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_atanh, T, all_float_types)
{
    RandomSample<T> rng{-1, 1};
    T               x_v         = rng.next();
    T               test_func_v = atanh(x_v);

    rvar<T, 1>      x_rvar      = x_v;

    rvar<T, 1>      x_func      = atanh(x_rvar);

    BOOST_REQUIRE_CLOSE(x_func.item(), test_func_v, boost_close_tol<T>());

    gradient_tape<T, 1>& tape = get_active_tape<T, 1>();

    tape.zero_grad();
    rvar<T, 1> test_func_2    = atanh(x_rvar * x_rvar);
    T          expected_deriv = -2 * x_v / (x_v * x_v * x_v * x_v - 1);
    test_func_2.backward();

    BOOST_REQUIRE_CLOSE(x_rvar.adjoint(), expected_deriv, boost_close_tol<T>());

    tape.clear();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_tgamma, T, all_float_types)
{
    using boost::math::digamma;
    using boost::math::polygamma;
    using boost::math::tgamma;
    RandomSample<T> rng{-1, 1};
    T               x_v = rng.next();
    while (abs(x_v) < 0.001) {
        x_v = rng.next();
    }
    T          test_func_v = tgamma(x_v);

    rvar<T, 1> x_rvar      = x_v;

    rvar<T, 1> x_func      = tgamma(x_rvar);

    BOOST_REQUIRE_CLOSE(x_func.item(), test_func_v, boost_close_tol<T>());

    gradient_tape<T, 1>& tape = get_active_tape<T, 1>();

    tape.zero_grad();
    rvar<T, 1> test_func_2    = tgamma(x_rvar);
    T          expected_deriv = tgamma(x_v) * digamma(x_v);
    test_func_2.backward();

    BOOST_REQUIRE_CLOSE(x_rvar.adjoint(), expected_deriv, boost_close_tol<T>());

    tape.clear();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_digamma, T, all_float_types)
{
    using boost::math::digamma;
    using boost::math::polygamma;
    using boost::math::tgamma;
    RandomSample<T> rng{-1, 1};
    T               x_v = rng.next();
    while (abs(x_v) < 0.001) {
        x_v = rng.next();
    }
    T          test_func_v = digamma(x_v);

    rvar<T, 1> x_rvar      = x_v;

    rvar<T, 1> x_func      = digamma(x_rvar);

    BOOST_REQUIRE_CLOSE(x_func.item(), test_func_v, boost_close_tol<T>());

    gradient_tape<T, 1>& tape = get_active_tape<T, 1>();

    tape.zero_grad();
    rvar<T, 1> test_func_2    = digamma(x_rvar);
    T          expected_deriv = polygamma(1, x_v);
    test_func_2.backward();

    BOOST_REQUIRE_CLOSE(x_rvar.adjoint(), expected_deriv, boost_close_tol<T>());

    tape.clear();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_polygamma, T, all_float_types)
{
    using boost::math::digamma;
    using boost::math::tgamma;
    RandomSample<T> rng{-1, 1};
    T               x_v = rng.next();
    while (abs(x_v) < 0.001) {
        x_v = rng.next();
    }
    for (int i = 1; i < 20; i++) {
        T          test_func_v = boost::math::polygamma(i, x_v);

        rvar<T, 1> x_rvar      = x_v;

        rvar<T, 1> x_func      = polygamma(i, x_rvar);

        BOOST_REQUIRE_CLOSE(x_func.item(), test_func_v, boost_close_tol<T>());

        gradient_tape<T, 1>& tape = get_active_tape<T, 1>();

        tape.zero_grad();
        rvar<T, 1> test_func_2    = polygamma(i, x_rvar);
        T          expected_deriv = boost::math::polygamma(i + 1, x_v);
        test_func_2.backward();

        BOOST_REQUIRE_CLOSE(x_rvar.adjoint(), expected_deriv, boost_close_tol<T>());
    }
    //tape.clear();
}
BOOST_AUTO_TEST_SUITE_END()
