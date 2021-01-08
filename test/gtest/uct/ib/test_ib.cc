/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <uct/ib/test_ib.h>
#ifdef HAVE_MLX5_HW
extern "C" {
#include <uct/ib/mlx5/ib_mlx5.h>
}
#endif


test_uct_ib::test_uct_ib() : m_e1(NULL), m_e2(NULL) { }

void test_uct_ib::create_connected_entities() {
    m_e1 = uct_test::create_entity(0);
    m_e2 = uct_test::create_entity(0);
    
    m_entities.push_back(m_e1);
    m_entities.push_back(m_e2);

    m_e1->connect(0, *m_e2, 0);
    m_e2->connect(0, *m_e1, 0);
}

void test_uct_ib::init() {
    uct_test::init();
    create_connected_entities();
    test_uct_ib::m_ib_am_handler_counter = 0;
}

ucs_status_t test_uct_ib::ib_am_handler(void *arg, void *data,
                                        size_t length, unsigned flags) {
    recv_desc_t *my_desc  = (recv_desc_t *) arg;
    uint64_t *test_ib_hdr = (uint64_t *) data;
    uint64_t *actual_data = (uint64_t *) test_ib_hdr + 1;
    unsigned data_length  = length - sizeof(test_ib_hdr);

    my_desc->length = data_length;
    if (*test_ib_hdr == 0xbeef) {
        memcpy(my_desc + 1, actual_data , data_length);
    }
    ++test_uct_ib::m_ib_am_handler_counter;
    return UCS_OK;
}

void test_uct_ib::send_recv_short() {
    size_t start_am_counter = test_uct_ib::m_ib_am_handler_counter;
    uint64_t send_data      = 0xdeadbeef;
    uint64_t test_ib_hdr    = 0xbeef;
    recv_desc_t *recv_buffer;
    ucs_status_t status;

    check_caps_skip(UCT_IFACE_FLAG_AM_SHORT);

    recv_buffer = (recv_desc_t *) malloc(sizeof(*recv_buffer) + sizeof(uint64_t));
    recv_buffer->length = 0; /* Initialize length to 0 */

    /* set a callback for the uct to invoke for receiving the data */
    uct_iface_set_am_handler(m_e2->iface(), 0, ib_am_handler, recv_buffer, 0);

    /* send the data */
    status = uct_ep_am_short(m_e1->ep(0), 0, test_ib_hdr,
                             &send_data, sizeof(send_data));
    EXPECT_TRUE((status == UCS_OK) || (status == UCS_INPROGRESS));

    flush();
    wait_for_value(&test_uct_ib::m_ib_am_handler_counter,
                   start_am_counter + 1, true);

    ASSERT_EQ(sizeof(send_data), recv_buffer->length);
    EXPECT_EQ(send_data, *(uint64_t*)(recv_buffer+1));

    free(recv_buffer);
}

size_t test_uct_ib::m_ib_am_handler_counter = 0;

class test_uct_ib_addr : public test_uct_ib {
public:
    uct_ib_iface_config_t *ib_config() {
        return ucs_derived_of(m_iface_config, uct_ib_iface_config_t);
    }

    void test_address_pack(uint64_t subnet_prefix) {
        uct_ib_iface_t *iface = ucs_derived_of(m_e1->iface(), uct_ib_iface_t);
        static const uint16_t lid_in = 0x1ee7;
        union ibv_gid gid_in;
        uct_ib_address_t *ib_addr;
        size_t address_size;

        gid_in.global.subnet_prefix = subnet_prefix;
        gid_in.global.interface_id  = 0xdeadbeef;

        uct_ib_address_pack_params_t pack_params;
        pack_params.flags     = uct_ib_iface_address_pack_flags(iface);
        pack_params.gid       = gid_in;
        pack_params.lid       = lid_in;
        pack_params.roce_info = iface->gid_info.roce_info;
        /* to suppress gcc 4.3.4 warning */
        pack_params.path_mtu  = (enum ibv_mtu)0;
        pack_params.gid_index = std::numeric_limits<uint8_t>::max();
        pack_params.pkey      = iface->pkey;
        address_size          = uct_ib_address_size(&pack_params);
        ib_addr               = (uct_ib_address_t*)malloc(address_size);
        uct_ib_address_pack(&pack_params, ib_addr);

        uct_ib_address_pack_params_t unpack_params;
        uct_ib_address_unpack(ib_addr, &unpack_params);

        if (uct_ib_iface_is_roce(iface)) {
            EXPECT_TRUE(iface->config.force_global_addr);
            EXPECT_TRUE((unpack_params.flags & UCT_IB_ADDRESS_PACK_FLAG_ETH) != 0);
            EXPECT_EQ(iface->gid_info.roce_info.addr_family,
                      unpack_params.roce_info.addr_family);
            EXPECT_EQ(iface->gid_info.roce_info.ver,
                      unpack_params.roce_info.ver);
        } else {
            EXPECT_EQ(lid_in, unpack_params.lid);
        }

        if (ib_config()->is_global &&
            !(unpack_params.flags & UCT_IB_ADDRESS_PACK_FLAG_ETH)) {
            EXPECT_TRUE(unpack_params.flags & UCT_IB_ADDRESS_PACK_FLAG_SUBNET_PREFIX);
            EXPECT_EQ(gid_in.global.subnet_prefix, unpack_params.gid.global.subnet_prefix);

            EXPECT_TRUE(unpack_params.flags & UCT_IB_ADDRESS_PACK_FLAG_INTERFACE_ID);
            EXPECT_EQ(gid_in.global.interface_id, unpack_params.gid.global.interface_id);
        }

        EXPECT_TRUE(!(unpack_params.flags & UCT_IB_ADDRESS_PACK_FLAG_PATH_MTU));
        EXPECT_EQ(UCT_IB_ADDRESS_INVALID_PATH_MTU, unpack_params.path_mtu);

        EXPECT_TRUE(!(unpack_params.flags & UCT_IB_ADDRESS_PACK_FLAG_GID_INDEX));
        EXPECT_EQ(UCT_IB_ADDRESS_INVALID_GID_INDEX, unpack_params.gid_index);

        EXPECT_TRUE((unpack_params.flags & UCT_IB_ADDRESS_PACK_FLAG_PKEY) != 0);
        EXPECT_EQ(iface->pkey, unpack_params.pkey);

        free(ib_addr);
    }

    void test_fill_ah_attr(uint64_t subnet_prefix) {
        uct_ib_iface_t *iface     = ucs_derived_of(m_e1->iface(), uct_ib_iface_t);
        static const uint16_t lid = 0x1ee7;
        union ibv_gid gid;
        struct ibv_ah_attr ah_attr;

        ASSERT_EQ(iface->config.force_global_addr,
                  ib_config()->is_global || uct_ib_iface_is_roce(iface));

        gid.global.subnet_prefix = subnet_prefix ?: iface->gid_info.gid.global.subnet_prefix;
        gid.global.interface_id  = 0xdeadbeef;

        uct_ib_iface_fill_ah_attr_from_gid_lid(iface, lid, &gid,
                                               iface->gid_info.gid_index, 0,
                                               &ah_attr);

        if (uct_ib_iface_is_roce(iface)) {
            /* in case of roce, should be global */
            EXPECT_TRUE(ah_attr.is_global);
        } else if (ib_config()->is_global) {
            /* in case of global address is forced - ah_attr should use GRH */
            EXPECT_TRUE(ah_attr.is_global);
        } else if (iface->gid_info.gid.global.subnet_prefix == gid.global.subnet_prefix) {
            /* in case of subnets are same - ah_attr depend from forced/nonforced GRH */
            EXPECT_FALSE(ah_attr.is_global);
        } else if (iface->gid_info.gid.global.subnet_prefix != gid.global.subnet_prefix) {
            /* in case of subnets are different - ah_attr should use GRH */
            EXPECT_TRUE(ah_attr.is_global);
        }
    }
};

UCS_TEST_P(test_uct_ib_addr, address_pack) {
    test_address_pack(UCT_IB_LINK_LOCAL_PREFIX);
    test_address_pack(UCT_IB_SITE_LOCAL_PREFIX | htobe64(0x7200));
    test_address_pack(0xdeadfeedbeefa880ul);
}

UCS_TEST_P(test_uct_ib_addr, fill_ah_attr) {
    test_fill_ah_attr(UCT_IB_LINK_LOCAL_PREFIX);
    test_fill_ah_attr(UCT_IB_SITE_LOCAL_PREFIX | htobe64(0x7200));
    test_fill_ah_attr(0xdeadfeedbeefa880ul);
    test_fill_ah_attr(0l);
}

UCS_TEST_P(test_uct_ib_addr, address_pack_global, "IB_IS_GLOBAL=y") {
    test_address_pack(UCT_IB_LINK_LOCAL_PREFIX);
    test_address_pack(UCT_IB_SITE_LOCAL_PREFIX | htobe64(0x7200));
    test_address_pack(0xdeadfeedbeefa880ul);
}

UCS_TEST_P(test_uct_ib_addr, fill_ah_attr_global, "IB_IS_GLOBAL=y") {
    test_fill_ah_attr(UCT_IB_LINK_LOCAL_PREFIX);
    test_fill_ah_attr(UCT_IB_SITE_LOCAL_PREFIX | htobe64(0x7200));
    test_fill_ah_attr(0xdeadfeedbeefa880ul);
    test_fill_ah_attr(0l);
}

UCT_INSTANTIATE_IB_TEST_CASE(test_uct_ib_addr);


test_uct_ib_with_specific_port::test_uct_ib_with_specific_port() {
    m_ibctx    = NULL;
    m_port     = 0;
    m_dev_name = "";

    memset(&m_port_attr, 0, sizeof(m_port_attr));
}

void test_uct_ib_with_specific_port::init() {
    size_t colon_pos = GetParam()->dev_name.find(":");
    std::string port_num_str;

    m_dev_name   = GetParam()->dev_name.substr(0, colon_pos);
    port_num_str = GetParam()->dev_name.substr(colon_pos + 1);

    /* port number */
    if (sscanf(port_num_str.c_str(), "%d", &m_port) != 1) {
        UCS_TEST_ABORT("Failed to get the port number on device: " << m_dev_name);
    }

    std::string abort_reason =
        "The requested device " + m_dev_name +
        " wasn't found in the device list.";
    struct ibv_device **device_list;
    int i, num_devices;

    /* get device list */
    device_list = ibv_get_device_list(&num_devices);
    if (device_list == NULL) {
        abort_reason = "Failed to get the device list.";
        num_devices = 0;
    }

    /* search for the given device in the device list */
    for (i = 0; i < num_devices; ++i) {
        if (strcmp(device_list[i]->name, m_dev_name.c_str())) {
            continue;
        }

        /* found this dev_name on the host - open it */
        m_ibctx = ibv_open_device(device_list[i]);
        if (m_ibctx == NULL) {
            abort_reason = "Failed to open the device.";
        }
        break;
    }

    ibv_free_device_list(device_list);
    if (m_ibctx == NULL) {
        UCS_TEST_ABORT(abort_reason);
    }

    if (ibv_query_port(m_ibctx, m_port, &m_port_attr) != 0) {
        UCS_TEST_ABORT("Failed to query port " << m_port <<
                       "on device: " << m_dev_name);
    }

    try {
        check_port_attr();
    } catch (...) {
        test_uct_ib_with_specific_port::cleanup();
        throw;
    }
}

void test_uct_ib_with_specific_port::cleanup() {
    if (m_ibctx != NULL) {
        ibv_close_device(m_ibctx);
        m_ibctx = NULL;
    }
}

class test_uct_ib_lmc : public test_uct_ib_with_specific_port {
public:
    void init() {
        test_uct_ib_with_specific_port::init();
        test_uct_ib::init();
    }

    void cleanup() {
        test_uct_ib::cleanup();
        test_uct_ib_with_specific_port::cleanup();
    }

    void check_port_attr() {
        /* check if a non zero lmc is set on the port */
        if (!m_port_attr.lmc) {
            UCS_TEST_SKIP_R("lmc is set to zero on an IB port");
        }
    }
};

UCS_TEST_P(test_uct_ib_lmc, non_default_lmc, "IB_LID_PATH_BITS=1") {
    send_recv_short();
}

UCT_INSTANTIATE_IB_TEST_CASE(test_uct_ib_lmc);

class test_uct_ib_gid_idx : public test_uct_ib_with_specific_port {
public:
    void init() {
        test_uct_ib_with_specific_port::init();
        test_uct_ib::init();
    }

    void cleanup() {
        test_uct_ib::cleanup();
        test_uct_ib_with_specific_port::cleanup();
    }

    void check_port_attr() {
        std::stringstream device_str;
        device_str << ibv_get_device_name(m_ibctx->device) << ":" << m_port;

        if (!IBV_PORT_IS_LINK_LAYER_ETHERNET(&m_port_attr)) {
            UCS_TEST_SKIP_R(device_str.str() + " is not Ethernet");
        }

        union ibv_gid gid;
        uct_ib_md_config_t *md_config =
            ucs_derived_of(m_md_config, uct_ib_md_config_t);
        ucs::handle<uct_md_h> uct_md;
        uct_ib_iface_t dummy_ib_iface;
        uct_ib_md_t *ib_md;
        ucs_status_t status;
        uint8_t gid_index;

        UCS_TEST_CREATE_HANDLE(uct_md_h, uct_md, uct_ib_md_close, uct_ib_md_open,
                               &uct_ib_component,
                               ibv_get_device_name(m_ibctx->device), m_md_config);

        ib_md = ucs_derived_of(uct_md, uct_ib_md_t);

        dummy_ib_iface.config.port_num = m_port;
        dummy_ib_iface.super.md        = &ib_md->super;

        ASSERT_EQ(&ib_md->dev, uct_ib_iface_device(&dummy_ib_iface));

        /* uct_ib_iface_init_roce_gid_info() requires only the port from the
         * ib_iface so we can use a dummy one here.
         * this function will set the gid_index in the dummy ib_iface. */
        status = uct_ib_iface_init_roce_gid_info(&dummy_ib_iface,
                                                 md_config->ext.gid_index);
        ASSERT_UCS_OK(status);

        gid_index = dummy_ib_iface.gid_info.gid_index;
        device_str << " gid index " << static_cast<int>(gid_index);

        /* check the gid index */
        if (ibv_query_gid(m_ibctx, m_port, gid_index, &gid) != 0) {
            UCS_TEST_ABORT("failed to query " + device_str.str());
        }

        /* check if the gid is valid to use */
        if (uct_ib_device_is_gid_raw_empty(gid.raw)) {
            UCS_TEST_SKIP_R(device_str.str() + " is empty");
        }

        if (!uct_ib_device_test_roce_gid_index(&ib_md->dev, m_port, &gid,
                                               gid_index)) {
            UCS_TEST_SKIP_R("failed to create address handle on " +
                            device_str.str());
        }
    }
};

UCS_TEST_P(test_uct_ib_gid_idx, non_default_gid_idx, "GID_INDEX=1") {
    send_recv_short();
}

UCT_INSTANTIATE_IB_TEST_CASE(test_uct_ib_gid_idx);


#if HAVE_DEVX
class test_uct_ib_sl : public test_uct_ib_with_specific_port {
public:
    void check_port_attr() {
        ucs_status_t status;
        ucs::handle<uct_md_h> uct_md;

        UCS_TEST_CREATE_HANDLE(uct_md_h, uct_md, uct_ib_md_close, uct_ib_md_open,
                               &uct_ib_component,
                               ibv_get_device_name(m_ibctx->device),
                               m_md_config);

        uct_ib_mlx5_md_t *ib_md = ucs_derived_of(uct_md, uct_ib_mlx5_md_t);

        /* check if OOO SL mask can be detected for the port */
        status = uct_ib_mlx5_devx_query_ooo_sl_mask(ib_md, m_port,
                                                    &m_ooo_sl_mask);
        EXPECT_TRUE((status == UCS_OK) || (status == UCS_ERR_UNSUPPORTED));
        if (status != UCS_OK) {
            UCS_TEST_SKIP_R("OOO SL mask couldn't be detected for " +
                            m_dev_name + ":" + ucs::to_string(m_port));
        }
    }

protected:
    uint16_t m_ooo_sl_mask;
};

UCS_TEST_P(test_uct_ib_sl, check_ib_sl_config) {
    // go over all SLs, check UCTs could be initialized on a specific SL
    // and able to send/recv traffic
    for (uint8_t sl = 0; sl < UCT_IB_SL_NUM; ++sl)  {
        if (!has_transport("rc_verbs") && !has_transport("ud_verbs")) {
            // if AR is configured on the given SL, set AR_ENABLE to "y",
            // otherwise - to "n" in order to test that AR_ENABLE parameter
            // works as expected w/o errors and warnings
            modify_config("IB_AR_ENABLE",
                          (m_ooo_sl_mask & UCS_BIT(sl)) ? "y" : "n");
        }
        modify_config("IB_SL", ucs::to_string(static_cast<uint16_t>(sl)));

        test_uct_ib::init();
        send_recv_short();
        test_uct_ib::cleanup();
    }
}

UCT_INSTANTIATE_IB_TEST_CASE(test_uct_ib_sl);
#endif

class test_uct_ib_utils : public ucs::test {
};

UCS_TEST_F(test_uct_ib_utils, sec_to_qp_time) {
    double avg;
    uint8_t qp_val;

    // 0 sec
    qp_val = uct_ib_to_qp_fabric_time(0);
    EXPECT_EQ(1, qp_val);

    // the average time defined for the [0, 1st element]
    qp_val = uct_ib_to_qp_fabric_time(4.096 * pow(2, 0) / UCS_USEC_PER_SEC);
    EXPECT_EQ(1, qp_val);

    // the time defined for the 1st element
    qp_val = uct_ib_to_qp_fabric_time(4.096 * pow(2, 1) / UCS_USEC_PER_SEC);
    EXPECT_EQ(1, qp_val);

    for (uint8_t index = 2; index <= UCT_IB_FABRIC_TIME_MAX; index++) {
        uint8_t prev_index = index - 1;

        // the time defined for the (i)th element
        qp_val = uct_ib_to_qp_fabric_time(4.096 * pow(2, index) / UCS_USEC_PER_SEC);
        EXPECT_EQ(index % UCT_IB_FABRIC_TIME_MAX, qp_val);

        // avg = (the average time defined for the [(i - 1)th element, (i)th element])
        avg = (4.096 * pow(2, prev_index) + 4.096 * pow(2, index)) * 0.5;
        qp_val = uct_ib_to_qp_fabric_time(avg / UCS_USEC_PER_SEC);
        EXPECT_EQ(index % UCT_IB_FABRIC_TIME_MAX, qp_val);

        // the average time defined for the [(i - 1)th element, avg]
        qp_val = uct_ib_to_qp_fabric_time((4.096 * pow(2, prev_index) + avg) * 0.5 / UCS_USEC_PER_SEC);
        EXPECT_EQ(prev_index, qp_val);

        // the average time defined for the [avg, (i)th element]
        qp_val = uct_ib_to_qp_fabric_time((avg +  4.096 * pow(2, index)) * 0.5 / UCS_USEC_PER_SEC);
        EXPECT_EQ(index % UCT_IB_FABRIC_TIME_MAX, qp_val);
    }
}

UCS_TEST_F(test_uct_ib_utils, sec_to_rnr_time) {
    double avg;
    uint8_t rnr_val;

    // 0 sec
    rnr_val = uct_ib_to_rnr_fabric_time(0);
    EXPECT_EQ(1, rnr_val);

    // the average time defined for the [0, 1st element]
    avg = uct_ib_qp_rnr_time_ms[1] * 0.5;
    rnr_val = uct_ib_to_rnr_fabric_time(avg / UCS_MSEC_PER_SEC);
    EXPECT_EQ(1, rnr_val);

    for (uint8_t index = 1; index < UCT_IB_FABRIC_TIME_MAX; index++) {
        uint8_t next_index = (index + 1) % UCT_IB_FABRIC_TIME_MAX;

        // the time defined for the (i)th element
        rnr_val = uct_ib_to_rnr_fabric_time(uct_ib_qp_rnr_time_ms[index] / UCS_MSEC_PER_SEC);
        EXPECT_EQ(index, rnr_val);

        // avg = (the average time defined for the [(i)th element, (i + 1)th element])
        avg = (uct_ib_qp_rnr_time_ms[index] + uct_ib_qp_rnr_time_ms[next_index]) * 0.5;
        rnr_val = uct_ib_to_rnr_fabric_time(avg / UCS_MSEC_PER_SEC);
        EXPECT_EQ(next_index, rnr_val);

        // the average time defined for the [(i)th element, avg]
        rnr_val = uct_ib_to_rnr_fabric_time((uct_ib_qp_rnr_time_ms[index] + avg) * 0.5 / UCS_MSEC_PER_SEC);
        EXPECT_EQ(index, rnr_val);

        // the average time defined for the [avg, (i + 1)th element]
        rnr_val = uct_ib_to_rnr_fabric_time((avg + uct_ib_qp_rnr_time_ms[next_index]) *
                                             0.5 / UCS_MSEC_PER_SEC);
        EXPECT_EQ(next_index, rnr_val);
    }

    // the time defined for the biggest value
    rnr_val = uct_ib_to_rnr_fabric_time(uct_ib_qp_rnr_time_ms[0] / UCS_MSEC_PER_SEC);
    EXPECT_EQ(0, rnr_val);

    // 1 sec
    rnr_val = uct_ib_to_rnr_fabric_time(1.);
    EXPECT_EQ(0, rnr_val);
}


#if HAVE_DEVX
class test_uct_ib_sl_utils : public test_uct_ib_utils {
protected:
    ucs_status_t ib_select_sl(ucs_ternary_auto_value_t ar_enable,
                              uint64_t test_ooo_sl_mask,
                              const uct_ib_iface_config_t &config,
                              uint8_t &sl) const {
        uint16_t ooo_sl_mask = (test_ooo_sl_mask !=
                                m_ooo_sl_mask_not_detected) ?
                               static_cast<uint16_t>(test_ooo_sl_mask) : 0;
        return uct_ib_mlx5_select_sl(&config, ar_enable, ooo_sl_mask,
                                     (test_ooo_sl_mask !=
                                      m_ooo_sl_mask_not_detected),
                                     "mlx5_0", 1, &sl);
    }

    ucs_status_t select_sl_ok(ucs_ternary_auto_value_t ar_enable,
                              unsigned long config_sl,
                              uint64_t ooo_sl_mask,
                              const uct_ib_iface_config_t &config) const {
        uint16_t sls_with_ar, sls_without_ar;
        ucs_status_t status;
        uint8_t sl;

        if (ooo_sl_mask != m_ooo_sl_mask_not_detected) {
            sls_with_ar    = static_cast<uint16_t>(ooo_sl_mask);
            sls_without_ar = static_cast<uint16_t>(~ooo_sl_mask);
        } else {
            sls_with_ar    =
            sls_without_ar = 0;
        }

        status = ib_select_sl(ar_enable, ooo_sl_mask, config, sl); 
        if ((ooo_sl_mask == 0) || (ar_enable == UCS_AUTO)) {
            if (config_sl == UCS_ULUNITS_AUTO) {
                EXPECT_EQ(m_default_sl, sl);
            } else {
                EXPECT_EQ(static_cast<uint8_t>(config_sl), sl);
            }
        } else if (config_sl == UCS_ULUNITS_AUTO) {
            if ((ar_enable == UCS_YES) ||
                ((ar_enable == UCS_TRY) && (sls_with_ar != 0))) {
                EXPECT_EQ(ucs_ffs64_safe(sls_with_ar), sl);
            } else if ((ar_enable == UCS_NO) ||
                       ((ar_enable == UCS_TRY) && (sls_without_ar != 0))) {
                EXPECT_EQ(ucs_ffs64_safe(sls_without_ar), sl);
            } else {
                EXPECT_EQ(UCS_TRY, ar_enable);
                EXPECT_EQ(m_default_sl, sl);
            }
        } else {
            EXPECT_EQ(static_cast<uint8_t>(config_sl), sl);
        }

        return status;
    }

    static ucs_log_func_rc_t
    wrap_errors_check_sl_masks_logger(const char *file, unsigned line,
                                      const char *function,
                                      ucs_log_level_t level,
                                      const ucs_log_component_config_t *
                                      comp_conf,
                                      const char *message, va_list ap)
    {
        if (level == UCS_LOG_LEVEL_ERROR) {
            std::string err_str = format_message(message, ap);

            for (uint8_t sl = 0; sl < UCT_IB_SL_NUM; ++sl) {
                std::string sl_val = ucs::to_string(static_cast<uint16_t>(sl));

                if ((err_str.find(sl_val + ", ") == std::string::npos) &&
                    (err_str.find(sl_val + " }") == std::string::npos)) {
                    return UCS_LOG_FUNC_RC_CONTINUE;
                }
            }

            return UCS_LOG_FUNC_RC_STOP;
        }

        return UCS_LOG_FUNC_RC_CONTINUE;
    }

    ucs_status_t select_sl_nok(ucs_ternary_auto_value_t ar_enable,
                               unsigned long config_sl, uint64_t ooo_sl_mask,
                               const uct_ib_iface_config_t &config) const {
        scoped_log_handler slh(((ooo_sl_mask != m_ooo_sl_mask_not_detected) &&
                                (config_sl == UCS_ULUNITS_AUTO)) ?
                               wrap_errors_check_sl_masks_logger :
                               wrap_errors_logger);
        uint8_t sl;

        EXPECT_NE(UCS_AUTO, ar_enable);

        return ib_select_sl(ar_enable, ooo_sl_mask, config, sl);
    }

    void select_sl(ucs_ternary_auto_value_t ar_enable, unsigned long config_sl,
                   uint64_t ooo_sl_mask, ucs_status_t exp_status) const {
        uct_ib_iface_config_t config = {};
        ucs_status_t status;

        config.sl = config_sl;

        if (exp_status == UCS_OK) {
            status = select_sl_ok(ar_enable, config_sl, ooo_sl_mask, config);
        } else {
            status = select_sl_nok(ar_enable, config_sl, ooo_sl_mask, config);
        }
        EXPECT_EQ(exp_status, status);
    }

protected:
    const static uint64_t m_ooo_sl_mask_not_detected;
    const static uint8_t  m_default_sl;
};

const uint64_t test_uct_ib_sl_utils::m_ooo_sl_mask_not_detected =
                                     std::numeric_limits<uint64_t>::max();
const uint8_t  test_uct_ib_sl_utils::m_default_sl               = 0;


UCS_TEST_F(test_uct_ib_sl_utils, sl_selection) {
    const ucs_status_t err_status = UCS_ERR_UNSUPPORTED;

    for (unsigned i = 0; i < static_cast<unsigned>(UCS_TERNARY_LAST); i++) {
        ucs_ternary_auto_value_t ar_enable =
            static_cast<ucs_ternary_auto_value_t>(i);

        // select the default SL, with empty OOO SL mask
        select_sl(ar_enable, UCS_ULUNITS_AUTO, 0,
                  (ar_enable == UCS_YES) ? err_status : UCS_OK);

        // select the default SL, without OOO SL mask (not detected)
        select_sl(ar_enable, UCS_ULUNITS_AUTO, m_ooo_sl_mask_not_detected,
                  ((ar_enable != UCS_TRY) && (ar_enable != UCS_AUTO)) ?
                  err_status : UCS_OK);

        for (uint8_t sl = 0; sl < UCT_IB_SL_NUM; ++sl) {
            // select the default SL, with OOO SL mask which contains only <sl>
            select_sl(ar_enable, UCS_ULUNITS_AUTO, UCS_BIT(sl), UCS_OK);

            // select SL=<sl>, with empty OOO SL mask
            select_sl(ar_enable, sl, 0,
                      (ar_enable == UCS_YES) ? err_status : UCS_OK);

            // select SL=<sl>, without OOO SL mask (not detected)
            select_sl(ar_enable, sl, m_ooo_sl_mask_not_detected,
                      ((ar_enable != UCS_TRY) && (ar_enable != UCS_AUTO)) ?
                      err_status : UCS_OK);

            /* select SL=<sl>, with OOO SL mask which contains only <sl> */
            select_sl(ar_enable, sl, UCS_BIT(sl),
                      (ar_enable == UCS_NO) ? err_status : UCS_OK);

            // select SL=<sl>, with OOO SL mask which doesn't contain <sl>, but
            // it contains one another element
            select_sl(ar_enable, sl, UCS_BIT(UCT_IB_SL_NUM - 1 - sl),
                      (ar_enable == UCS_YES) ? err_status : UCS_OK);

            // select SL=<sl>, with OOO SL mask which contains <sl> and two
            // other elements
            select_sl(ar_enable, sl,
                      UCS_BIT(sl)                       |
                      UCS_BIT((sl + 1) % UCT_IB_SL_NUM) |
                      UCS_BIT((sl + 2) % UCT_IB_SL_NUM),
                      (ar_enable == UCS_NO) ? err_status : UCS_OK);

            // select SL=<sl>, with OOO SL mask which doesn't contain <sl>
            select_sl(ar_enable, sl,
                      UCS_BIT((sl + 1) % UCT_IB_SL_NUM) |
                      UCS_BIT((sl + 2) % UCT_IB_SL_NUM) |
                      UCS_BIT((sl + 3) % UCT_IB_SL_NUM),
                      (ar_enable == UCS_YES) ? err_status : UCS_OK);

            // select SL=<sl>, with full OOO SL mask
            select_sl(ar_enable, sl, UCS_MASK(UCT_IB_SL_NUM),
                      (ar_enable == UCS_NO) ? err_status : UCS_OK);

            // select SL=<sl>, with full OOO SL mask, except <sl>
            select_sl(ar_enable, sl, UCS_MASK(UCT_IB_SL_NUM) & ~UCS_BIT(sl),
                      (ar_enable == UCS_YES) ? err_status : UCS_OK);
        }
    }
}

UCS_TEST_F(test_uct_ib_sl_utils, query_ooo_sl_mask) {
    int num_devices;
    struct ibv_device **ib_device_list;
    ucs_status_t status;

    ib_device_list = ibv_get_device_list(&num_devices);
    ASSERT_TRUE(ib_device_list != NULL);

    for (int i = 0; i < num_devices; ++i) {
        const char *dev_name = ibv_get_device_name(ib_device_list[i]);
        uct_md_config_t *md_config;
        uct_ib_mlx5_md_t *ib_mlx5_md;
        uct_ib_device_t *dev;
        uct_md_h md;

        status = uct_md_config_read(&uct_ib_component, NULL, NULL, &md_config);
        EXPECT_UCS_OK(status);
        if (status != UCS_OK) {
            continue;
        }

        status = uct_ib_md_open(&uct_ib_component, dev_name, md_config, &md);
        EXPECT_UCS_OK(status);
        if (status != UCS_OK) {
            goto out_md_config_release;
        }

        ib_mlx5_md = ucs_derived_of(md, uct_ib_mlx5_md_t);
        dev        = &ib_mlx5_md->super.dev;

        for (uint8_t port_num = dev->first_port;
             port_num <= dev->num_ports; ++port_num) {
            uint16_t ooo_sl_mask = 0;
            ucs_string_buffer_t strb;

            status = uct_ib_mlx5_devx_query_ooo_sl_mask(ib_mlx5_md, port_num,
                                                        &ooo_sl_mask);
            EXPECT_TRUE((status == UCS_OK) || (status == UCS_ERR_UNSUPPORTED));
            if ((status != UCS_OK) && (status != UCS_ERR_UNSUPPORTED)) {
                continue;
            }

            ucs_string_buffer_init(&strb);
            UCS_TEST_MESSAGE << "OOO SL mask for " << dev_name << " - { "
                             << ucs_mask_str(ooo_sl_mask, &strb) << " }";
            ucs_string_buffer_cleanup(&strb);
        }

        uct_ib_md_close(md);
out_md_config_release:
        uct_config_release(md_config);
    }

    ibv_free_device_list(ib_device_list);
}
#endif


class test_uct_event_ib : public test_uct_ib {
public:
    test_uct_event_ib() {
        length      = 8;
        test_ib_hdr = 0xbeef;
        m_buf1      = NULL;
        m_buf2      = NULL;
    }

    void init() {
        test_uct_ib::init();

        check_skip_test();

        m_buf1 = new mapped_buffer(length, 0x1, *m_e1);
        m_buf2 = new mapped_buffer(length, 0x2, *m_e2);

        /* set a callback for the uct to invoke for receiving the data */
        uct_iface_set_am_handler(m_e1->iface(), 0, ib_am_handler, m_buf1->ptr(),
                                 0);

        test_uct_event_ib::bcopy_pack_count = 0;
    }

    /* overload `test_uct_ib` variant to pass the async event handler to
     * the receive entity */
    void create_connected_entities() {
        /* `m_e1` entity is used as a receiver in UCT IB Event tests */
        m_e1 = uct_test::create_entity(0, NULL, NULL, NULL, NULL, NULL,
                                       async_event_handler, this);
        m_e2 = uct_test::create_entity(0);

        m_entities.push_back(m_e1);
        m_entities.push_back(m_e2);

        m_e1->connect(0, *m_e2, 0);
        m_e2->connect(0, *m_e1, 0);
    }

    static void async_event_handler(void *arg, unsigned flags) {
        test_uct_event_ib *self = static_cast<test_uct_event_ib*>(arg);
        self->m_async_event_ctx.signal();
    }

    static size_t pack_cb(void *dest, void *arg) {
        const mapped_buffer *buf = (const mapped_buffer *)arg;
        memcpy(dest, buf->ptr(), buf->length());
        ++test_uct_event_ib::bcopy_pack_count;
        return buf->length();
    }

    /* Use put_bcopy here to provide send_cq entry */
    void send_msg_e1_e2(size_t count = 1) {
        for (size_t i = 0; i < count; ++i) {
            ssize_t status = uct_ep_put_bcopy(m_e1->ep(0), pack_cb, (void *)m_buf1,
                                              m_buf2->addr(), m_buf2->rkey());
            if (status < 0) {
                ASSERT_UCS_OK((ucs_status_t)status);
            }
        }
    }

    void send_msg_e2_e1(size_t count = 1) {
        for (size_t i = 0; i < count; ++i) {
            ucs_status_t status = uct_ep_am_short(m_e2->ep(0), 0, test_ib_hdr,
                                                  m_buf2->ptr(), m_buf2->length());
            ASSERT_UCS_OK(status);
        }
    }

    void check_send_cq(uct_iface_t *iface, size_t val) {
        uct_ib_iface_t *ib_iface = ucs_derived_of(iface, uct_ib_iface_t);
        struct ibv_cq  *send_cq = ib_iface->cq[UCT_IB_DIR_TX];

        if (val != send_cq->comp_events_completed) {
            uint32_t completed_evt = send_cq->comp_events_completed;
            /* need this call to acknowledge the completion to prevent iface dtor hung*/
            ibv_ack_cq_events(ib_iface->cq[UCT_IB_DIR_TX], 1);
            UCS_TEST_ABORT("send_cq->comp_events_completed have to be 1 but the value "
                           << completed_evt);
        }
    }

    void check_recv_cq(uct_iface_t *iface, size_t val) {
        uct_ib_iface_t *ib_iface = ucs_derived_of(iface, uct_ib_iface_t);
        struct ibv_cq  *recv_cq = ib_iface->cq[UCT_IB_DIR_RX];

        if (val != recv_cq->comp_events_completed) {
            uint32_t completed_evt = recv_cq->comp_events_completed;
            /* need this call to acknowledge the completion to prevent iface dtor hung*/
            ibv_ack_cq_events(ib_iface->cq[UCT_IB_DIR_RX], 1);
            UCS_TEST_ABORT("recv_cq->comp_events_completed have to be 1 but the value "
                           << completed_evt);
        }
    }

    void cleanup() {
        delete(m_buf1);
        delete(m_buf2);
        test_uct_ib::cleanup();
    }

protected:
    static const unsigned EVENTS = UCT_EVENT_RECV | UCT_EVENT_SEND_COMP;

    size_t length;
    uint64_t test_ib_hdr;
    mapped_buffer *m_buf1, *m_buf2;
    static size_t bcopy_pack_count;
    uct_test::async_event_ctx m_async_event_ctx;
};

size_t test_uct_event_ib::bcopy_pack_count = 0;


UCS_TEST_SKIP_COND_P(test_uct_event_ib, tx_cq,
                     !check_caps(UCT_IFACE_FLAG_PUT_BCOPY |
                                 UCT_IFACE_FLAG_CB_SYNC) ||
                     !check_event_caps(UCT_IFACE_FLAG_EVENT_SEND_COMP |
                                       UCT_IFACE_FLAG_EVENT_RECV))
{
    ucs_status_t status;

    status = uct_iface_event_arm(m_e1->iface(), EVENTS);
    ASSERT_EQ(status, UCS_OK);

    /* check initial state of the fd and [send|recv]_cq */
    EXPECT_FALSE(m_async_event_ctx.wait_for_event(*m_e1, 0));
    check_send_cq(m_e1->iface(), 0);
    check_recv_cq(m_e1->iface(), 0);

    /* send the data */
    send_msg_e1_e2();

    /* make sure the file descriptor is signaled once */
    EXPECT_TRUE(m_async_event_ctx.wait_for_event(*m_e1,
                                                 1000 *
                                                 ucs::test_time_multiplier()));

    status = uct_iface_event_arm(m_e1->iface(), EVENTS);
    ASSERT_EQ(status, UCS_ERR_BUSY);

    /* make sure [send|recv]_cq handled properly */
    check_send_cq(m_e1->iface(), 1);
    check_recv_cq(m_e1->iface(), 0);

    m_e1->flush();
}


UCS_TEST_SKIP_COND_P(test_uct_event_ib, txrx_cq,
                     !check_caps(UCT_IFACE_FLAG_PUT_BCOPY |
                                 UCT_IFACE_FLAG_CB_SYNC   |
                                 UCT_IFACE_FLAG_AM_SHORT) ||
                     !check_event_caps(UCT_IFACE_FLAG_EVENT_SEND_COMP |
                                       UCT_IFACE_FLAG_EVENT_RECV))
{
    const size_t msg_count = 1;
    ucs_status_t status;

    status = uct_iface_event_arm(m_e1->iface(), EVENTS);
    ASSERT_EQ(UCS_OK, status);

    /* check initial state of the fd and [send|recv]_cq */
    EXPECT_FALSE(m_async_event_ctx.wait_for_event(*m_e1, 0));
    check_send_cq(m_e1->iface(), 0);
    check_recv_cq(m_e1->iface(), 0);

    /* send the data */
    send_msg_e1_e2(msg_count);
    send_msg_e2_e1(msg_count);

    twait(150); /* Let completion to be generated */

    /* Make sure all messages delivered */
    while ((test_uct_ib::m_ib_am_handler_counter   < msg_count) ||
           (test_uct_event_ib::bcopy_pack_count < msg_count)) {
        progress();
    }

    /* make sure the file descriptor is signaled */
    EXPECT_TRUE(m_async_event_ctx.wait_for_event(*m_e1,
                                                 1000 *
                                                 ucs::test_time_multiplier()));

    /* Acknowledge all the requests */
    short_progress_loop();
    status = uct_iface_event_arm(m_e1->iface(), EVENTS);
    ASSERT_EQ(UCS_ERR_BUSY, status);

    /* make sure [send|recv]_cq handled properly */
    check_send_cq(m_e1->iface(), 1);
    check_recv_cq(m_e1->iface(), 1);

    m_e1->flush();
    m_e2->flush();
}


UCT_INSTANTIATE_IB_TEST_CASE(test_uct_event_ib);
