/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019. ALL RIGHTS RESERVED.
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

typedef enum {
    MATERIAL_LEATHER,
    MATERIAL_ALCANTARA,
    MATERIAL_TEXTILE,
    MATERIAL_LAST
} material_t;

const char *color_names[] = {
    /* [COLOR_RED]    = */ "red",
    /* [COLOR_BLUE]   = */ "blue",
    /* [COLOR_BLACK]  = */ "black",
    /* [COLOR_YELLOW] = */ "yellow",
    /* [COLOR_WHITE]  = */ "white",
    /* [COLOR_LAST]   = */ NULL
};

const char *material_names[] = {
    /* [MATERIAL_LEATHER]   = */ "leather",
    /* [MATERIAL_ALCANTARA] = */ "alcantara",
    /* [MATERIAL_TEXTILE]   = */ "textile",
    /* [MATERIAL_LAST]      = */ NULL
};

typedef struct {
    color_t         color;
    material_t      material;
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
    color_t         color;
    unsigned long   vin;

    double          bw_bytes;
    double          bw_kbytes;
    double          bw_mbytes;
    double          bw_gbytes;
    double          bw_tbytes;
    double          bw_bits;
    double          bw_kbits;
    double          bw_mbits;
    double          bw_gbits;
    double          bw_tbits;
    double          bw_auto;

    ucs_config_bw_spec_t can_pci_bw; /* CAN-bus */

    int             air_conditioning;
    int             abs;
    int             transmission;

    ucs_time_t      time_value;
    ucs_time_t      time_auto;
    ucs_time_t      time_inf;
} car_opts_t;


ucs_config_field_t seat_opts_table[] = {
  {"COLOR", "black", "Seat color",
   ucs_offsetof(seat_opts_t, color), UCS_CONFIG_TYPE_ENUM(color_names)},

  {"COLOR_ALIAS", NULL, "Seat color",
   ucs_offsetof(seat_opts_t, color), UCS_CONFIG_TYPE_ENUM(color_names)},

  {"MATERIAL", "textile", "Cover seat material",
   ucs_offsetof(seat_opts_t, material), UCS_CONFIG_TYPE_ENUM(material_names)},

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

  {"POWER_ALIAS", NULL, "Engine power",
   ucs_offsetof(engine_opts_t, power), UCS_CONFIG_TYPE_ULUNITS},

  {"FUEL_LEVEL", "", "This is electric car",
   UCS_CONFIG_DEPRECATED_FIELD_OFFSET, UCS_CONFIG_TYPE_DEPRECATED},

  {NULL}
};

ucs_config_field_t car_opts_table[] = {
  {"ENGINE_", "", "Engine options",
   ucs_offsetof(car_opts_t, engine), UCS_CONFIG_TYPE_TABLE(engine_opts_table)},

  {"COACH_", "PASSENGER_COLOR=blue", "Seats options",
   ucs_offsetof(car_opts_t, coach), UCS_CONFIG_TYPE_TABLE(coach_opts_table)},

  {"PRICE", "999", "Price",
   ucs_offsetof(car_opts_t, price), UCS_CONFIG_TYPE_UINT},

  {"PRICE_ALIAS", NULL, "Price",
   ucs_offsetof(car_opts_t, price), UCS_CONFIG_TYPE_UINT},

  {"DRIVER", "", "AI drives a car",
   UCS_CONFIG_DEPRECATED_FIELD_OFFSET, UCS_CONFIG_TYPE_DEPRECATED},

  {"BRAND", "Chevy", "Car brand",
   ucs_offsetof(car_opts_t, brand), UCS_CONFIG_TYPE_STRING},

  {"MODEL", "Corvette", "Car model",
   ucs_offsetof(car_opts_t, model), UCS_CONFIG_TYPE_STRING},

  {"COLOR", "red", "Car color",
   ucs_offsetof(car_opts_t, color), UCS_CONFIG_TYPE_ENUM(color_names)},

  {"VIN", "auto", "Vehicle identification number",
   ucs_offsetof(car_opts_t, vin), UCS_CONFIG_TYPE_ULUNITS},

  {"BW_BYTES", "1024Bs", "Bandwidth in bytes",
   ucs_offsetof(car_opts_t, bw_bytes), UCS_CONFIG_TYPE_BW},

  {"BW_KBYTES", "1024KB/s", "Bandwidth in kbytes",
   ucs_offsetof(car_opts_t, bw_kbytes), UCS_CONFIG_TYPE_BW},

  {"BW_MBYTES", "1024MBs", "Bandwidth in mbytes",
   ucs_offsetof(car_opts_t, bw_mbytes), UCS_CONFIG_TYPE_BW},

  {"BW_GBYTES", "1024GBps", "Bandwidth in gbytes",
   ucs_offsetof(car_opts_t, bw_gbytes), UCS_CONFIG_TYPE_BW},

  {"BW_TBYTES", "1024TB/s", "Bandwidth in tbytes",
   ucs_offsetof(car_opts_t, bw_tbytes), UCS_CONFIG_TYPE_BW},

  {"BW_BITS", "1024bps", "Bandwidth in bits",
   ucs_offsetof(car_opts_t, bw_bits), UCS_CONFIG_TYPE_BW},

  {"BW_KBITS", "1024Kb/s", "Bandwidth in kbits",
   ucs_offsetof(car_opts_t, bw_kbits), UCS_CONFIG_TYPE_BW},

  {"BW_MBITS", "1024Mbs", "Bandwidth in mbits",
   ucs_offsetof(car_opts_t, bw_mbits), UCS_CONFIG_TYPE_BW},

  {"BW_GBITS", "1024Gbps", "Bandwidth in gbits",
   ucs_offsetof(car_opts_t, bw_gbits), UCS_CONFIG_TYPE_BW},

  {"BW_TBITS", "1024Tbs", "Bandwidth in tbits",
   ucs_offsetof(car_opts_t, bw_tbits), UCS_CONFIG_TYPE_BW},

  {"BW_AUTO", "auto", "Auto bandwidth value",
   ucs_offsetof(car_opts_t, bw_auto), UCS_CONFIG_TYPE_BW},

  {"CAN_BUS_BW", "mlx5_0:1024Tbs", "Bandwidth in tbits of CAN-bus",
   ucs_offsetof(car_opts_t, can_pci_bw), UCS_CONFIG_TYPE_BW_SPEC},

  {"AIR_CONDITIONING", "on", "Air conditioning mode",
   ucs_offsetof(car_opts_t, air_conditioning), UCS_CONFIG_TYPE_ON_OFF},

  {"ABS", "off", "ABS mode",
   ucs_offsetof(car_opts_t, abs), UCS_CONFIG_TYPE_ON_OFF},

  {"TRANSMISSION", "auto", "Transmission mode",
   ucs_offsetof(car_opts_t, transmission), UCS_CONFIG_TYPE_ON_OFF_AUTO},

  {"TIME_VAL", "1s", "Time value 1 sec",
   ucs_offsetof(car_opts_t, time_value), UCS_CONFIG_TYPE_TIME_UNITS},

  {"TIME_AUTO", "auto", "Time value \"auto\"",
   ucs_offsetof(car_opts_t, time_auto), UCS_CONFIG_TYPE_TIME_UNITS},

  {"TIME_INF", "inf", "Time value \"inf\"",
   ucs_offsetof(car_opts_t, time_inf), UCS_CONFIG_TYPE_TIME_UNITS},

  {NULL}
};

static std::vector<std::string> config_err_exp_str;

class test_config : public ucs::test {
protected:
    static ucs_log_func_rc_t
    config_error_handler(const char *file, unsigned line, const char *function,
                         ucs_log_level_t level,
                         const ucs_log_component_config_t *comp_conf,
                         const char *message, va_list ap)
    {
        // Ignore errors that invalid input parameters as it is expected
        if (level == UCS_LOG_LEVEL_WARN) {
            std::string err_str = format_message(message, ap);

            for (size_t i = 0; i < config_err_exp_str.size(); i++) {
                if (err_str.find(config_err_exp_str[i]) != std::string::npos) {
                    UCS_TEST_MESSAGE << err_str;
                    return UCS_LOG_FUNC_RC_STOP;
                }
            }
        }

        return UCS_LOG_FUNC_RC_CONTINUE;
    }

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

    static void test_config_print_opts(unsigned flags,
                                       unsigned exp_num_lines,
                                       const char *prefix = NULL)
    {
        char *dump_data;
        size_t dump_size;
        char line_buf[1024];
        char alias[128];
        car_opts opts(UCS_DEFAULT_ENV_PREFIX, NULL);

        memset(alias, 0, sizeof(alias));

        /* Dump configuration to a memory buffer */
        dump_data = NULL;
        FILE *file = open_memstream(&dump_data, &dump_size);
        ucs_config_parser_print_opts(file, "", *opts, car_opts_table,
                                     prefix, UCS_DEFAULT_ENV_PREFIX,
                                     (ucs_config_print_flags_t)flags);

        /* Sanity check - all lines begin with UCS_ */
        unsigned num_lines = 0;
        fseek(file, 0, SEEK_SET);
        while (fgets(line_buf, sizeof(line_buf), file)) {
            if (line_buf[0] == '\n') {
                continue;
            }

            if (line_buf[0] != '#') {
                /* found the name of attribute */

                if (alias[0] != '\0') {
                    /* the code below relies on the fact that all
                     * aliases has the name: "<real_name>_ALIAS" */
                    EXPECT_EQ(0, strncmp(alias, line_buf,
                                         strlen(alias) - strlen("_ALIAS")));
                    memset(alias, 0, sizeof(alias));
                }

                std::string exp_str = "UCX_";
                if (prefix) {
                    exp_str += prefix;
                }
                line_buf[exp_str.size()] = '\0';
                EXPECT_STREQ(exp_str.c_str(), line_buf);
                ++num_lines;
            } else if (strncmp(&line_buf[2], "alias of:",
                               strlen("alias of:")) == 0) {
                /* found the alias name of attribute */

                size_t cnt = 0;
                for (size_t i = 2 + strlen("alias of: ") + 1;
                     line_buf[i] != '\n'; i++) {
                    alias[cnt++] = line_buf[i];
                }
            }
        }

        EXPECT_EQ(exp_num_lines, num_lines);

        fclose(file);
        free(dump_data);
    }
};

UCS_TEST_F(test_config, parse_default) {
    car_opts opts(UCS_DEFAULT_ENV_PREFIX, "TEST");

    EXPECT_EQ(999U, opts->price);
    EXPECT_EQ(std::string("Chevy"), opts->brand);
    EXPECT_EQ(std::string("Corvette"), opts->model);
    EXPECT_EQ(COLOR_RED, opts->color);
    EXPECT_EQ(6000U, opts->engine.volume);
    EXPECT_EQ(COLOR_RED, opts->coach.driver_seat.color);
    EXPECT_EQ(COLOR_BLUE, opts->coach.passenger_seat.color);
    EXPECT_EQ(COLOR_BLACK, opts->coach.rear_seat.color);
    EXPECT_EQ(UCS_ULUNITS_AUTO, opts->vin);
    EXPECT_EQ(200UL, opts->engine.power);

    EXPECT_EQ(1024.0, opts->bw_bytes);
    EXPECT_EQ(UCS_KBYTE * 1024.0, opts->bw_kbytes);
    EXPECT_EQ(UCS_MBYTE * 1024.0, opts->bw_mbytes);
    EXPECT_EQ(UCS_GBYTE * 1024.0, opts->bw_gbytes);
    EXPECT_EQ(UCS_TBYTE * 1024.0, opts->bw_tbytes);

    EXPECT_EQ(128.0, opts->bw_bits);
    EXPECT_EQ(UCS_KBYTE * 128.0, opts->bw_kbits);
    EXPECT_EQ(UCS_MBYTE * 128.0, opts->bw_mbits);
    EXPECT_EQ(UCS_GBYTE * 128.0, opts->bw_gbits);
    EXPECT_EQ(UCS_TBYTE * 128.0, opts->bw_tbits);
    EXPECT_TRUE(UCS_CONFIG_BW_IS_AUTO(opts->bw_auto));

    EXPECT_EQ(UCS_TBYTE * 128.0, opts->can_pci_bw.bw);
    EXPECT_EQ(std::string("mlx5_0"), opts->can_pci_bw.name);

    EXPECT_EQ(UCS_CONFIG_ON, opts->air_conditioning);
    EXPECT_EQ(UCS_CONFIG_OFF, opts->abs);
    EXPECT_EQ(UCS_CONFIG_AUTO, opts->transmission);

    EXPECT_EQ(ucs_time_from_sec(1.0), opts->time_value);
    EXPECT_EQ(UCS_TIME_AUTO, opts->time_auto);
    EXPECT_EQ(UCS_TIME_INFINITY, opts->time_inf);
}

UCS_TEST_F(test_config, clone) {

    car_opts *opts_clone_ptr;

    {
        /* coverity[tainted_string_argument] */
        ucs::scoped_setenv env1("UCX_COLOR", "white");
        /* coverity[tainted_string_argument] */
        ucs::scoped_setenv env2("UCX_PRICE_ALIAS", "0");
        
        car_opts opts(UCS_DEFAULT_ENV_PREFIX, NULL);
        EXPECT_EQ(COLOR_WHITE, opts->color);
        EXPECT_EQ(0U, opts->price);

        /* coverity[tainted_string_argument] */
        ucs::scoped_setenv env3("UCX_COLOR", "black");
        opts_clone_ptr = new car_opts(opts);
    }

    EXPECT_EQ(COLOR_WHITE, (*opts_clone_ptr)->color);
    delete opts_clone_ptr;
}

UCS_TEST_F(test_config, set_get) {
    car_opts opts(UCS_DEFAULT_ENV_PREFIX, NULL);
    EXPECT_EQ(COLOR_RED, opts->color);
    EXPECT_EQ(std::string(color_names[COLOR_RED]),
              std::string(opts.get("COLOR")));

    opts.set("COLOR", "white");
    EXPECT_EQ(COLOR_WHITE, opts->color);
    EXPECT_EQ(std::string(color_names[COLOR_WHITE]),
              std::string(opts.get("COLOR")));

    opts.set("DRIVER_COLOR_ALIAS", "black");
    EXPECT_EQ(COLOR_BLACK, opts->coach.driver_seat.color);
    EXPECT_EQ(std::string(color_names[COLOR_BLACK]),
              std::string(opts.get("COACH_DRIVER_COLOR_ALIAS")));

    opts.set("VIN", "123456");
    EXPECT_EQ(123456UL, opts->vin);
}

UCS_TEST_F(test_config, set_get_with_table_prefix) {
    /* coverity[tainted_string_argument] */
    ucs::scoped_setenv env1("UCX_COLOR", "black");
    /* coverity[tainted_string_argument] */
    ucs::scoped_setenv env2("UCX_CARS_COLOR", "white");

    car_opts opts(UCS_DEFAULT_ENV_PREFIX, "CARS_");
    EXPECT_EQ(COLOR_WHITE, opts->color);
    EXPECT_EQ(std::string(color_names[COLOR_WHITE]),
              std::string(opts.get("COLOR")));
}

UCS_TEST_F(test_config, set_get_with_env_prefix) {
    /* coverity[tainted_string_argument] */
    ucs::scoped_setenv env1("UCX_COLOR", "black");
    /* coverity[tainted_string_argument] */
    ucs::scoped_setenv env2("TEST_UCX_COLOR", "white");

    car_opts opts("TEST_" UCS_DEFAULT_ENV_PREFIX, NULL);
    EXPECT_EQ(COLOR_WHITE, opts->color);
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
    UCS_TEST_TIME_LIMIT(0.05) {
        car_opts opts(UCS_DEFAULT_ENV_PREFIX, NULL);
    }
}

UCS_TEST_F(test_config, unused) {
    ucs::ucx_env_cleanup env_cleanup;

    /* set to warn about unused env vars */
    ucs_global_opts.warn_unused_env_vars = 1;

    const std::string warn_str    = "unused env variable";
    const std::string unused_var1 = "UCX_UNUSED_VAR1";
    /* coverity[tainted_string_argument] */
    ucs::scoped_setenv env1(unused_var1.c_str(), "unused");

    {
        config_err_exp_str.push_back(warn_str + ": " + unused_var1);
        scoped_log_handler log_handler(config_error_handler);
        car_opts opts(UCS_DEFAULT_ENV_PREFIX, NULL);

        ucs_config_parser_warn_unused_env_vars_once(UCS_DEFAULT_ENV_PREFIX);

        config_err_exp_str.pop_back();
    }

    {
        const std::string unused_var2 = "TEST_UNUSED_VAR2";
        /* coverity[tainted_string_argument] */
        ucs::scoped_setenv env2(unused_var2.c_str(), "unused");

        config_err_exp_str.push_back(warn_str + ": " + unused_var2);
        scoped_log_handler log_handler(config_error_handler);
        car_opts opts("TEST_", NULL);

        ucs_config_parser_warn_unused_env_vars_once("TEST_");

        config_err_exp_str.pop_back();
    }

    /* reset to not warn about unused env vars */
    ucs_global_opts.warn_unused_env_vars = 0;
}

UCS_TEST_F(test_config, dump) {
    /* aliases must not be counted here */
    test_config_print_opts(UCS_CONFIG_PRINT_CONFIG, 31u);
}

UCS_TEST_F(test_config, dump_hidden) {
    /* aliases must be counted here */
    test_config_print_opts((UCS_CONFIG_PRINT_CONFIG |
                            UCS_CONFIG_PRINT_HIDDEN),
                           38u);
}

UCS_TEST_F(test_config, dump_hidden_check_alias_name) {
    /* aliases must be counted here */
    test_config_print_opts((UCS_CONFIG_PRINT_CONFIG |
                            UCS_CONFIG_PRINT_HIDDEN |
                            UCS_CONFIG_PRINT_DOC),
                           38u);

    test_config_print_opts((UCS_CONFIG_PRINT_CONFIG |
                            UCS_CONFIG_PRINT_HIDDEN |
                            UCS_CONFIG_PRINT_DOC),
                           38u, "TEST_");
}

UCS_TEST_F(test_config, deprecated) {
    /* set to warn about unused env vars */
    ucs_global_opts.warn_unused_env_vars = 1;

    const std::string warn_str        = " is deprecated";
    const std::string deprecated_var1 = "UCX_DRIVER";
    /* coverity[tainted_string_argument] */
    ucs::scoped_setenv env1(deprecated_var1.c_str(), "Taxi driver");
    config_err_exp_str.push_back(deprecated_var1 + warn_str);

    {
        scoped_log_handler log_handler(config_error_handler);
        car_opts opts(UCS_DEFAULT_ENV_PREFIX, NULL);
    }

    {
        const std::string deprecated_var2 = "UCX_ENGINE_FUEL_LEVEL";
        /* coverity[tainted_string_argument] */
        ucs::scoped_setenv env2(deprecated_var2.c_str(), "58");
        config_err_exp_str.push_back(deprecated_var2 + warn_str);

        scoped_log_handler log_handler_vars(config_error_handler);
        car_opts opts(UCS_DEFAULT_ENV_PREFIX, NULL);
        config_err_exp_str.pop_back();
    }

    config_err_exp_str.pop_back();

    /* reset to not warn about unused env vars */
    ucs_global_opts.warn_unused_env_vars = 0;
}
