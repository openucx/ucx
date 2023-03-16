#include <stdio.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <getopt.h>
#include <unistd.h>
#include <inttypes.h>
#include <string.h>
#include <sys/time.h>

#include <shmem.h>

#define ROOT_PE 0
#define OSHM_LOOP_ATOMIC 500
#define FIELD_WIDTH 20
#define FLOAT_PRECISION 2
#define HEADER "# Atomic Fetch & Add Benchmark\n"


union data_types {
    int         int_type;
    long        long_type;
    long long   longlong_type;
    float       float_type;
    double      double_type;
} global_msg_buffer[OSHM_LOOP_ATOMIC];

double pwrk[_SHMEM_REDUCE_MIN_WRKDATA_SIZE];
long psync[_SHMEM_REDUCE_SYNC_SIZE];

double getMicrosecondTimeStamp();

#define TIME getMicrosecondTimeStamp

void init(int* rank, int* npes)
{
    int i;
    shmem_init();
    *rank = shmem_my_pe();
    *npes = shmem_n_pes();

    for (i = 0; i < _SHMEM_REDUCE_SYNC_SIZE; i += 1) {
        psync[i] = _SHMEM_SYNC_VALUE;
    }
}

void print_operation_rate (int myid, char * operation, double rate, double lat)
{
    if (myid == 0) {
        fprintf(stdout, "%-*s%*.*f%*.*f\n", 20, operation, FIELD_WIDTH,
                FLOAT_PRECISION, rate, FIELD_WIDTH, FLOAT_PRECISION, lat);
        fflush(stdout);
    }
}

double getMicrosecondTimeStamp()
{
    double retval;
    struct timeval tv;
    if (gettimeofday(&tv, NULL)) {
        perror("gettimeofday");
        abort();
    }
    retval = ((double)tv.tv_sec) * 1000000 + tv.tv_usec;
    return retval;
}



double benchmark_fadd (int my_rank, int npes, union data_types *buffer, int buffer_size,
                unsigned long iterations)
{
    double begin, end; 
    int i;
    static double rate = 0, sum_rate = 0, lat = 0, sum_lat = 0;

    /*
     * Touch memory
     */
    memset(buffer, CHAR_MAX * drand48(), buffer_size);

    shmem_barrier_all();

    if (my_rank != ROOT_PE) {
        int value = 1;
        int old_value;

        begin = TIME();
        for (i = 0; i < iterations; i++) { 
            old_value = shmem_int_fadd(&(buffer[i].int_type), value, ROOT_PE);
        }
        end = TIME();

        rate = ((double)iterations * 1e6) / (end - begin);
        // lat = (end - begin) / (double)iterations;
    }

    shmem_double_sum_to_all(&sum_rate, &rate, 1, 0, 0, npes, pwrk, psync);
    // shmem_double_sum_to_all(&sum_lat, &lat, 1, 0, 0, v.npes, pwrk2, psync2);    
    print_operation_rate(my_rank, "shmem_int_fadd", sum_rate/1e6, sum_lat/npes);
    return 0;
}

void print_header_local(int myid)
{
    if (myid == 0) {
        fprintf(stdout, HEADER);
        fprintf(stdout, "%-*s%*s%*s\n", 20, "# Operation", FIELD_WIDTH,
                "Million ops/s", FIELD_WIDTH, "Latency (us)");
        fflush(stdout);
    }
}

static void usage()
{
    printf("Usage:   shmem_fadd [options]\n");
    printf("\n");
    printf("Options are:\n");
    printf("  -n <iters>     Specify number of iterations to run (default: 10000).\n");
    printf("  -s <size>      Specify message size (default: 4 bytes).\n");
    printf("  -w             Wait for data using shmem_wait_until() (default: poll on memory).\n");
    printf("  -f             Send data and flag separately with shmem_fence() in-between.\n");
    printf("  -g             Use global data (default: heap).\n");
    printf("  -q             call shmem_quiet() after every shmem_put().\n");
    printf("\n");
}

int main(int argc, char **argv)
{
    size_t msg_size;
    long num_iters;
    union data_types *msg_buffer;
    int c;
    int my_rank, npes;
    init(&my_rank, &npes);

    if (npes < 2) {
        fprintf(stderr, "This test requires at least 2 processes\n");
        return -1;
    }

    num_iters  = 10000;
    msg_size   = 8;
    while ((c = getopt (argc, argv, "n:s:wqfh")) != -1) {
        switch (c) {
            break;
        case 'n':
            num_iters = atol(optarg);
            if (num_iters == 0) {
                num_iters = LONG_MAX;
            }
            break;
        case 's':
            msg_size = atol(optarg);
            break;
        case 'h':
        default:
            if (my_rank == 0) {
                usage();
            }
            return 0;
        }
    }
    
    msg_buffer = shmalloc(msg_size);

    memset(msg_buffer, 0xff, msg_size);

    print_header_local(my_rank);

    shmem_barrier_all();

    benchmark_fadd(my_rank, npes, msg_buffer, msg_size, num_iters);

    shmem_barrier_all();

    shfree(msg_buffer);

    shmem_finalize();

    return 0;
}
