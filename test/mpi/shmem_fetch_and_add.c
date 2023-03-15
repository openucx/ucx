
double benchmark_fadd (struct pe_vars v, union data_types *buffer,
                unsigned long iterations)
{
    double begin, end; 
    int i;
    static double rate = 0, sum_rate = 0, lat = 0, sum_lat = 0;

    /*
     * Touch memory
     */
    memset(buffer, CHAR_MAX * drand48(), sizeof(union data_types
                [OSHM_LOOP_ATOMIC]));

    shmem_barrier_all();

    if (v.me < v.pairs) {
        int value = 1;
        int old_value;

        begin = TIME();
        for (i = 0; i < iterations; i++) { 
            old_value = shmem_int_fadd(&(buffer[i].int_type), value, v.nxtpe);
        }
        end = TIME();

        rate = ((double)iterations * 1e6) / (end - begin);
        lat = (end - begin) / (double)iterations;
    }

    shmem_double_sum_to_all(&sum_rate, &rate, 1, 0, 0, v.npes, pwrk1, psync1);
    shmem_double_sum_to_all(&sum_lat, &lat, 1, 0, 0, v.npes, pwrk2, psync2);    
    print_operation_rate(v.me, "shmem_int_fadd", sum_rate/1e6, sum_lat/v.pairs);

    return 0;
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
    static char global_buffer[GLOBAL_DATA_SIZE];
    int use_wait, use_global, do_quiet, use_flag;
    size_t msg_size;
    long num_iters;
    int my_pe;
    char *mem;
    int c;

    start_pes(0);

    my_pe = shmem_my_pe();

    if (shmem_n_pes() != 2) {
        fprintf(stderr, "This test requires exactly 2 processes\n");
        return -1;
    }

    num_iters  = 10000;
    use_global = 0;
    use_wait   = 0;
    do_quiet   = 0;
    use_flag   = 0;
    msg_size   = 8;
    while ((c = getopt (argc, argv, "n:s:wgqfh")) != -1) {
        switch (c) {
            break;
        case 'n':
            num_iters = atol(optarg);
            if (num_iters == 0) {
                num_iters = LONG_MAX;
            }
            break;
        case 'w':
            use_wait = 1;
            break;
        case 'g':
            use_global = 1;
            break;
        case 'q':
            do_quiet = 1;
            break;
        case 'f':
            use_flag = 1;
            break;
        case 's':
            msg_size = atol(optarg);
            break;
        case 'h':
        default:
            if (my_pe == 0) {
                usage();
            }
            return 0;
        }
    }

    if (msg_size < sizeof(int)) {
        fprintf(stderr, "message size must be at least %lu\n", sizeof(int));
        return -1;
    }

    if (use_global) {
        if (msg_size <= GLOBAL_DATA_SIZE) {
            mem = global_buffer;
        } else {
            fprintf(stderr, "global data can be used only up to %lu bytes\n",
                    (size_t)GLOBAL_DATA_SIZE);
            return -1;
        }
    } else {
        mem = shmalloc(msg_size);
    }

    memset(mem, 0xff, msg_size);

    shmem_barrier_all();

    run_pingpong(mem, msg_size, num_iters, use_wait, do_quiet, use_flag);

    shmem_barrier_all();

    if (!use_global) {
        shfree(mem);
    }

    shmem_finalize();
    return 0;
}
