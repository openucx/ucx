/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) Los Alamos National Security, LLC. 2018. ALL RIGHTS RESERVED.
*
*/
#include <list>
#include <numeric>
#include <set>
#include <vector>
#include <math.h>

#include "ucp_datatype.h"
#include "ucp_test.h"

#define NUM_MESSAGES 17

class test_ucp_am_base : public ucp_test {
public:
    int sent_ams;
    int recv_ams;
    void *for_release[NUM_MESSAGES];
    int release;

    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.field_mask |= UCP_PARAM_FIELD_FEATURES;
        params.features    = UCP_FEATURE_AM;
        return params;
    }
    
    static void ucp_put_am_cb(void *request, ucs_status_t status);
    static ucs_status_t ucp_process_am_cb(void *arg, void *data, 
                                  size_t length, unsigned flags);

    ucs_status_t am_handler(test_ucp_am_base *me, void *data, 
                            size_t  length, unsigned flags);
};

void test_ucp_am_base::ucp_put_am_cb(void *request, ucs_status_t status){
    return;
}

ucs_status_t test_ucp_am_base::ucp_process_am_cb(void *arg, void *data,
                                                 size_t length, unsigned flags){
    test_ucp_am_base *self = reinterpret_cast<test_ucp_am_base*>(arg);
    return self->am_handler(self, data, length, flags);

}

ucs_status_t test_ucp_am_base::am_handler(test_ucp_am_base *me, void *data, 
                                          size_t length, unsigned flags){
    ucs_status_t status;
    std::vector<char> cmp(length, (char)length);
    std::vector<char> databuf(length, 'r');

    memcpy(&databuf[0], data, length);

    EXPECT_EQ(cmp, databuf);
    if(me->release){
      me->for_release[me->recv_ams] = data;

      status = UCS_INPROGRESS;
    }
    else{
      status = UCS_OK; 
    }

    me->recv_ams++;
    return status;
}

class test_ucp_am : public test_ucp_am_base
{
public:
    ucp_ep_params_t get_ep_params() {
        ucp_ep_params_t params = test_ucp_am_base::get_ep_params();
        params.field_mask |= UCP_EP_PARAM_FIELD_FLAGS;
        params.flags      |= UCP_EP_PARAMS_FLAGS_NO_LOOPBACK;
        return params;
    }
    virtual void init(){
        ucp_test::init();
        sender().connect(&receiver(), get_ep_params());
        receiver().connect(&sender(), get_ep_params());
    }

protected:
    void do_set_am_handler_realloc_test();
    void do_send_process_data_test(int test_release, uint16_t am_id);
    void do_send_process_data_iov_test();
    void set_handlers(uint16_t am_id);
};

void test_ucp_am::set_handlers(uint16_t am_id){
    ucp_worker_set_am_handler(sender().worker(), am_id,
                              ucp_process_am_cb, this, 
                              UCP_AM_FLAG_WHOLE_MSG);
    ucp_worker_set_am_handler(receiver().worker(), am_id,
                              ucp_process_am_cb, this,
                              UCP_AM_FLAG_WHOLE_MSG);
}

void test_ucp_am::do_send_process_data_test(int test_release, uint16_t am_id)
{
    size_t buf_size = pow(2, NUM_MESSAGES - 2); /* minus 2 because 0 and 1*/
    std::vector<char> buf(buf_size);
    ucs_status_ptr_t sstatus = NULL;
    recv_ams = 0;
    sent_ams = 0;
    this->release = test_release;
    
    for (size_t i = 0; i < buf_size + 1; i *= 2) {
        for(size_t j = 0; j < i; j++){
            buf[j] = i;
        }
        sstatus = ucp_am_send_nb(receiver().ep(), am_id, 
                                 buf.data(), 1, ucp_dt_make_contig(i), 
                                 test_ucp_am_base::ucp_put_am_cb, 0);

        EXPECT_FALSE(UCS_PTR_IS_ERR(sstatus));
        wait(sstatus);

        if(i == 0){
          i = 1;
        } 
        sent_ams++;
    }
    while(sent_ams != recv_ams){
        progress();
    }
    if(test_release){
      for(int i = 0; i < recv_ams; i++){
          if(for_release[i] != NULL){
              ucp_am_data_release(receiver().worker(), for_release[i]);
          }
      }
    }
}

void test_ucp_am::do_send_process_data_iov_test()
{
    ucs_status_ptr_t sstatus;
    size_t iovcnt = 2;
    size_t size = 8192;
    size_t index;
    size_t i;
    recv_ams = 0;
    sent_ams = 0;
    release = 0;
    
    std::vector<char> b1(size);
    std::vector<char> b2(size);
    ucp_dt_iov_t iovec[iovcnt];
    
    set_handlers(0);
  
    for(i = 1; i < size; i *= 2){
        for(index = 0; index < i; index++){
            b1[index] = i * 2;
            b2[index] = i * 2;
        }

        iovec[0].buffer = b1.data();
        iovec[1].buffer = b2.data();

        iovec[0].length = i;
        iovec[1].length = i;
        
        sstatus = ucp_am_send_nb(receiver().ep(), 0,
                                 iovec, 2, ucp_dt_make_iov(),
                                 test_ucp_am_base::ucp_put_am_cb, 0);
        wait(sstatus);
        EXPECT_FALSE(UCS_PTR_IS_ERR(sstatus));
        sent_ams++;  
    }
    while(sent_ams != recv_ams){
        progress();
    }
}

void test_ucp_am::do_set_am_handler_realloc_test()
{
    set_handlers(0);
    do_send_process_data_test(0, 0);
    
    set_handlers(1000);
    do_send_process_data_test(0, 0);

    set_handlers(1);
    do_send_process_data_test(0, 1);
}

UCS_TEST_P(test_ucp_am, send_process_am) {
    set_handlers(0);
    do_send_process_data_test(0, 0);
}

UCS_TEST_P(test_ucp_am, send_process_am_release){
    set_handlers(0);
    do_send_process_data_test(1, 0);
}

UCS_TEST_P(test_ucp_am, send_process_iov_am) {
    do_send_process_data_iov_test();
}

UCS_TEST_P(test_ucp_am, set_am_handler_realloc) {
    do_set_am_handler_realloc_test();
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_am)
