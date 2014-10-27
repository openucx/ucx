/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include <ucs/gtest/test.h>
extern "C" {
#include <ucs/config/parser.h>
#include <ucs/time/time.h>
}

#include <boost/lexical_cast.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>


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
} engine_opts_t;

typedef struct {
    engine_opts_t   engine;
    coach_opts_t    coach;
    unsigned        price;
    const char      *brand;
    const char      *model;
    unsigned        color;
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

  {NULL}
};

class test_config : public ucs::test {
protected:

    /*
     * Wrapper class for car options parser.
     */
    class car_opts {
    public:
        car_opts(const char *user_prefix = NULL) : m_opts(parse(user_prefix)) {
        }

        car_opts(const car_opts& orig) :
            m_opts((car_opts_t*)ucs_malloc(sizeof *m_opts, "opts"))
        {
            ucs_status_t status = ucs_config_parser_clone_opts(orig.m_opts,
                                                               m_opts,
                                                               car_opts_table);
            ASSERT_UCS_OK(status);
        }

        ~car_opts() {
            ucs_config_parser_release_opts(m_opts, car_opts_table);
            ucs_free(m_opts);
        }

        void set(const char *name, const char *value) {
            ucs_config_parser_set_value(m_opts, car_opts_table, name, value);
        }

        car_opts_t* operator->() const {
            return m_opts;
        }

        car_opts_t* operator*() const {
            return m_opts;
        }
    private:

        static car_opts_t *parse(const char *user_prefix) {
            car_opts_t *tmp;
            ucs_status_t status = ucs_config_parser_read_opts(car_opts_table,
                                                              user_prefix,
                                                              sizeof(*tmp),
                                                              (void**)&tmp);
            ASSERT_UCS_OK(status);
            return tmp;
        }

        car_opts_t * const m_opts;
    };
};

UCS_TEST_F(test_config, parse_default) {
    car_opts opts("TEST");

    EXPECT_EQ(999, opts->price);
    EXPECT_EQ(std::string("Chevy"), opts->brand);
    EXPECT_EQ(std::string("Corvette"), opts->model);
    EXPECT_EQ(COLOR_RED, opts->color);
    EXPECT_EQ(6000, opts->engine.volume);
    EXPECT_EQ(COLOR_RED, opts->coach.driver_seat.color);
    EXPECT_EQ(COLOR_BLUE, opts->coach.passenger_seat.color);
    EXPECT_EQ(COLOR_BLACK, opts->coach.rear_seat.color);

}

UCS_TEST_F(test_config, parse_with_prefix) {
    ucs::scoped_setenv env1("UCS_COLOR", "white");
    ucs::scoped_setenv env2("UCS_TEST_COLOR", "black");
    ucs::scoped_setenv env3("UCS_DRIVER_COLOR", "yellow");
    ucs::scoped_setenv env4("UCS_TEST_REAR_COLOR", "white");

    car_opts dfl, test("TEST");
    EXPECT_EQ(COLOR_WHITE, dfl->color);
    EXPECT_EQ(COLOR_BLACK, test->color);
    EXPECT_EQ(COLOR_YELLOW, test->coach.driver_seat.color);
    EXPECT_EQ(COLOR_WHITE, test->coach.rear_seat.color);
}

UCS_TEST_F(test_config, clone) {

    boost::shared_ptr<car_opts> opts_clone_ptr;

    {
        ucs::scoped_setenv env1("UCS_TEST_COLOR", "white");
        car_opts opts("TEST");
        EXPECT_EQ(COLOR_WHITE, opts->color);

        ucs::scoped_setenv env2("UCS_TEST_COLOR", "black");
        opts_clone_ptr = boost::make_shared<car_opts>(opts);
    }

    EXPECT_EQ(COLOR_WHITE, (*opts_clone_ptr)->color);
}

UCS_TEST_F(test_config, set) {
    car_opts opts("TEST");
    EXPECT_EQ(COLOR_RED, opts->color);

    opts.set("COLOR", "white");
    EXPECT_EQ(COLOR_WHITE, opts->color);

}

UCS_TEST_F(test_config, performance) {

    /* Add stuff to env to presumably make getenv() slower */
    boost::ptr_vector<ucs::scoped_setenv> env;
    for (unsigned i = 0; i < 300; ++i) {
        env.push_back(new ucs::scoped_setenv(
                        (std::string("MTEST") + boost::lexical_cast<std::string>(i)).c_str(),
                        ""));
    }

    /* Now test the time */
    UCS_TEST_TIME_LIMIT(0.005) {
        car_opts opts("TEST");
    }
}

UCS_TEST_F(test_config, dump) {
    char *dump_data;
    size_t dump_size;
    char line_buf[1024];

    car_opts opts("TEST");

    /* Dump configuration to a memory buffer */
    dump_data = NULL;
    FILE *file = open_memstream(&dump_data, &dump_size);
    ucs_config_parser_print_opts(file, "", *opts, car_opts_table,
                                 0);

    /* Sanity check - all lines begin with UCS_ */
    unsigned num_lines = 0;
    fseek(file, 0, SEEK_SET);
    while (fgets(line_buf, sizeof(line_buf), file)) {
        line_buf[4] = '\0';
        EXPECT_STREQ("UCS_", line_buf);
        ++num_lines;
    }
    EXPECT_EQ(8u, num_lines);

    fclose(file);
    free(dump_data);
}
