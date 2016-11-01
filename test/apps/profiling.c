/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#define HAVE_PROFILING 1
#include <ucs/debug/profile.h>

#include <stdio.h>


/* calc_pi() would be profiled */
UCS_PROFILE_FUNC(double, calc_pi, (count), int count) {
    double pi_d_4;
    int n;

    pi_d_4 = 0.0;

    /* Profile a block of code */
    UCS_PROFILE_CODE("leibnitz") {
        for (n = 0; n < count; ++n) {
            pi_d_4 += pow(-1.0, n) / (2 * n + 1);

            /* create a timestamp for each step */
            UCS_PROFILE_SAMPLE("step");
        }
    }

    return pi_d_4 * 4.0;
}

/* print_pi() would be profiled */
UCS_PROFILE_FUNC_VOID(print_pi, (pi), double pi) {
    /* Call printf() and profile it */
    UCS_PROFILE_CALL(printf, "PI estimation is %.10f\n", pi);
}

int main(int argc, char **argv)
{
    double pi = calc_pi(10);
    print_pi(pi);
    return 0;
}
