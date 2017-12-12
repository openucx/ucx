/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
/* force older C++ version to have SIZE_MAX */
#define __STDC_LIMIT_MACROS
#define __STDC_CONSTANT_MACROS
#include <common/test.h>
extern "C" {
#include <ucs/config/parser.h>
#include <ucs/time/time.h>
}


typedef enum {
    COLOR_RED,
    COLOR_BLUE,
    COLOR_BLACK,
    COLOR_YELLOW,
    COLOR_WHITE,
    COLOR_LAST
} color_t;

const char *color_names[] = {
    /*[COLOR_RED]   =*/ "red",
    /*[COLOR_BLUE]  =*/ "blue",
    /*[COLOR_BLACK] =*/ "black",
    /*[COLOR_YELLOW] =*/ "yellow",
    /*[COLOR_WHITE] =*/ "white",
    /*[COLOR_LAST]  =*/ NULL
};

typedef struct {
    unsigned        color;
} seat_opts_t;

typedef struct {
    seat_opts_t     driver_seat;
    seat_opts_t     passenger_seat;
    seat_opts_t     rear_seat;
} coach_opts_t;

typedef struct {
    unsigned        volume;
    unsigned long   power;
} engine_opts_t;

typedef struct {
    engine_opts_t   engine;
    coach_opts_t    coach;
    unsigned        price;
    const char      *brand;
    const char      *model;
    unsigned        color;
    unsigned long   vin;
} car_opts_t;


ucs_config_field_t seat_opts_table[] = {
  {"COLOR", "black", "Seat color",
   ucs_offsetof(seat_opts_t, color), UCS_CONFIG_TYPE_ENUM(color_names)},

  {NULL}
};

ucs_config_field_t coach_opts_table[] = {
  {"DRIVER_", "COLOR=red", "Driver seat options",
   ucs_offsetof(coach_opts_t, driver_seat), UCS_CONFIG_TYPE_TABLE(seat_opts_table)},

  {"PASSENGER_", "", "Passenger seat options",
   ucs_offsetof(coach_opts_t, passenger_seat), UCS_CONFIG_TYPE_TABLE(seat_opts_table)},

  {"REAR_", "", "Rear seat options",
   ucs_offsetof(coach_opts_t, rear_seat), UCS_CONFIG_TYPE_TABLE(seat_opts_table)},

  {NULL}
};

ucs_config_field_t engine_opts_table[] = {
  {"VOLUME", "6000", "Engine volume",
   ucs_offsetof(engine_opts_t, volume), UCS_CONFIG_TYPE_UINT},

  {"POWER", "200", "Engine power",
   ucs_offsetof(engine_opts_t, power), UCS_CONFIG_TYPE_ULUNITS},

  {NULL}
};

ucs_config_field_t car_opts_table[] = {
  {"ENGINE_", "", "Engine options",
   ucs_offsetof(car_opts_t, engine), UCS_CONFIG_TYPE_TABLE(engine_opts_table)},

  {"COACH_", "PASSENGER_COLOR=blue", "Seats options",
   ucs_offsetof(car_opts_t, coach), UCS_CONFIG_TYPE_TABLE(coach_opts_table)},

  {"PRICE", "999", "Price",
   ucs_offsetof(car_opts_t, price), UCS_CONFIG_TYPE_UINT},

  {"BRAND", "Chevy", "Car brand",
   ucs_offsetof(car_opts_t, brand), UCS_CONFIG_TYPE_STRING},

  {"MODEL", "Corvette", "Car model",
   ucs_offsetof(car_opts_t, model), UCS_CONFIG_TYPE_STRING},

  {"COLOR", "red", "Car color",
   ucs_offsetof(car_opts_t, color), UCS_CONFIG_TYPE_ENUM(color_names)},

  {"VIN", "auto", "Vehicle identification number",
   ucs_offsetof(car_opts_t, vin), UCS_CONFIG_TYPE_ULUNITS},

  {NULL}
};

class test_config : public ucs::test {
protected:

    /*
     * Wrapper class for car options parser.
     */
    class car_opts {
    public:
        car_opts(const char *env_prefix, const char *table_prefix) :
            m_opts(parse(env_prefix, table_prefix)), m_max(1024), m_value(NULL)
        {
            m_value    = new char[m_max];
            m_value[0] = '\0';
        }

        car_opts(const car_opts& orig) : m_max(orig.m_max)
        {
            m_value = new char[m_max];
            strncpy(m_value, orig.m_value, m_max);

            ucs_status_t status = ucs_config_parser_clone_opts(&orig.m_opts,
                                                               &m_opts,
                                                               car_opts_table);
            ASSERT_UCS_OK(status);
        }

        ~car_opts() {
            ucs_config_parser_release_opts(&m_opts, car_opts_table);
            delete [] m_value;
        }

        void set(const char *name, const char *value) {
            ucs_config_parser_set_value(&m_opts, car_opts_table, name, value);
        }

        const char* get(const char *name) {
            ucs_status_t status = ucs_config_parser_get_value(&m_opts,
                                                              car_opts_table,
                                                              name, m_value,
                                                              m_max);
            ASSERT_UCS_OK(status);
            return m_value;
        }

        car_opts_t* operator->() {
            return &m_opts;
        }

        car_opts_t* operator*() {
            return &m_opts;
        }
    private:

        static car_opts_t parse(const char *env_prefix,
                                const char *table_prefix) {
            car_opts_t tmp;
            ucs_status_t status = ucs_config_parser_fill_opts(&tmp,
                                                              car_opts_table,
                                                              env_prefix,
                                                              table_prefix,
                                                              0);
            ASSERT_UCS_OK(status);
            return tmp;
        }

        car_opts_t   m_opts;
        const size_t m_max;
        char         *m_value;
    };
};

UCS_TEST_F(test_config, parse_default) {
    car_opts opts(NULL, "TEST");

    EXPECT_EQ((unsigned)999, opts->price);
    EXPECT_EQ(std::string("Chevy"), opts->brand);
    EXPECT_EQ(std::string("Corvette"), opts->model);
    EXPECT_EQ((unsigned)COLOR_RED, opts->color);
    EXPECT_EQ((unsigned)6000, opts->engine.volume);
    EXPECT_EQ((unsigned)COLOR_RED, opts->coach.driver_seat.color);
    EXPECT_EQ((unsigned)COLOR_BLUE, opts->coach.passenger_seat.color);
    EXPECT_EQ((unsigned)COLOR_BLACK, opts->coach.rear_seat.color);
    EXPECT_EQ(UCS_CONFIG_ULUNITS_AUTO, opts->vin);
    EXPECT_EQ(200UL, opts->engine.power);
}

UCS_TEST_F(test_config, clone) {

    car_opts *opts_clone_ptr;

    {
        /* coverity[tainted_string_argument] */
        ucs::scoped_setenv env1("UCX_COLOR", "white");
        car_opts opts(NULL, NULL);
        EXPECT_EQ((unsigned)COLOR_WHITE, opts->color);

        /* coverity[tainted_string_argument] */
        ucs::scoped_setenv env2("UCX_COLOR", "black");
        opts_clone_ptr = new car_opts(opts);
    }

    EXPECT_EQ((unsigned)COLOR_WHITE, (*opts_clone_ptr)->color);
    delete opts_clone_ptr;
}

UCS_TEST_F(test_config, set_get) {
    car_opts opts(NULL, NULL);
    EXPECT_EQ((unsigned)COLOR_RED, opts->color);
    EXPECT_EQ(std::string(color_names[COLOR_RED]),
              std::string(opts.get("COLOR")));

    opts.set("COLOR", "white");
    EXPECT_EQ((unsigned)COLOR_WHITE, opts->color);
    EXPECT_EQ(std::string(color_names[COLOR_WHITE]),
              std::string(opts.get("COLOR")));

    opts.set("VIN", "123456");
    EXPECT_EQ(123456UL, opts->vin);
}

UCS_TEST_F(test_config, set_get_with_table_prefix) {
    /* coverity[tainted_string_argument] */
    ucs::scoped_setenv env1("UCX_COLOR", "black");
    /* coverity[tainted_string_argument] */
    ucs::scoped_setenv env2("UCX_CARS_COLOR", "white");

    car_opts opts(NULL, "CARS_");
    EXPECT_EQ((unsigned)COLOR_WHITE, opts->color);
    EXPECT_EQ(std::string(color_names[COLOR_WHITE]),
              std::string(opts.get("COLOR")));
}

UCS_TEST_F(test_config, set_get_with_env_prefix) {
    /* coverity[tainted_string_argument] */
    ucs::scoped_setenv env1("UCX_COLOR", "black");
    /* coverity[tainted_string_argument] */
    ucs::scoped_setenv env2("UCX_TEST_COLOR", "white");

    car_opts opts("TEST", NULL);
    EXPECT_EQ((unsigned)COLOR_WHITE, opts->color);
    EXPECT_EQ(std::string(color_names[COLOR_WHITE]),
              std::string(opts.get("COLOR")));
}

UCS_TEST_F(test_config, performance) {

    /* Add stuff to env to presumably make getenv() slower */
    ucs::ptr_vector<ucs::scoped_setenv> env;
    for (unsigned i = 0; i < 300; ++i) {
        env.push_back(new ucs::scoped_setenv(
                        (std::string("MTEST") + ucs::to_string(i)).c_str(),
                        ""));
    }

    /* Now test the time */
    UCS_TEST_TIME_LIMIT(0.005) {
        car_opts opts(NULL, NULL);
    }
}

UCS_TEST_F(test_config, dump) {
    char *dump_data;
    size_t dump_size;
    char line_buf[1024];

    car_opts opts(NULL, NULL);

    /* Dump configuration to a memory buffer */
    dump_data = NULL;
    FILE *file = open_memstream(&dump_data, &dump_size);
    ucs_config_parser_print_opts(file, "", *opts, car_opts_table,
                                 NULL, UCS_CONFIG_PRINT_CONFIG);

    /* Sanity check - all lines begin with UCS_ */
    unsigned num_lines = 0;
    fseek(file, 0, SEEK_SET);
    while (fgets(line_buf, sizeof(line_buf), file)) {
        line_buf[4] = '\0';
        EXPECT_STREQ("UCX_", line_buf);
        ++num_lines;
    }
    EXPECT_EQ(10u, num_lines);

    fclose(file);
    free(dump_data);
}
