--
-- Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2023. ALL RIGHTS RESERVED.
-- See file LICENSE for terms.
--

-- Usage:
-- $ wireshark -Xlua_script:ucx_le64.lua <pcap_file>
--
-- or copy the LUA file in the Wireshark plugins directory.
--   - Help -> About Wireshark -> Folders -> Personal Lua plugins

-- Implements 64-bit little endian UCX protocols, v1 headers only:
--   - ucp stream data
--   - ucp_address
--   - wireup protocol
--   - tcp transport layer
--   - wireup sockaddr
--   - sock cm
--   - AMO SW (over AM)
--   - IB UD NetH / CTL
--   - IB RC proto
--
-- Missing:
--   - v2 headers
--   - rkey decoding
--   - wrapper for calls/offsets to support either le/be, 32-bit
--   - TL specific addresses dissections
--   - unified mode format
--   - ...

local ucx_status = {
    [0] = "OK",
}

local bitset = {
    [0] = "Not Set",
    [1] = "Set",
}

local ucp_tl_name = {
    [0x19cf] = "tcp",
    [0x7563] = "self",
    [0x538d] = "sysv",
    [0xd3a7] = "posix",
    [0x8b47] = "cma",
    [0x0fb3] = "ud_mlx5",
    [0x397e] = "dc_mlx5",
    [0x8a27] = "rc_verbs",
    [0xd47a] = "rc_mlx5",
    [0xd131] = "ud_verbs",
    [0x83dd] = "xpmem",
    [0x1377] = "knem",
}

function get_safe(table, key, default)
    value = table[key]
    if value == nil then
        value = default
    end
    return value
end

function get_ucx_status(status)
    return get_safe(ucx_status, status, "Code " .. tostring(status))
end

function get_tl_name(checksum)
    return get_safe(ucp_tl_name, checksum, "Unknown")
end

function bit_shift(shift)
    return 2 ^ shift
end

function pull(tvbuf, count)
    if tvbuf:len() <= count then
        return nil
    end
    return tvbuf(count, tvbuf:len() - count)
end

function hex_64(value)
    return string.format("0x%s", value)
end


local RNDV_TAG = 0
local RNDV_AM  = 1

local rndv_opcode = {
    [RNDV_TAG] = "TAG",
    [RNDV_AM]  = "AM",
}

local RNDV_RTS = 1
local RNDV_ATS = 2
local RNDV_RTR = 3
local RNDV_ATP = 4

local ucp_rndv_proto = Proto("UCX_RNDV", "UCX Rendezvous")
local ucp_rndv_fields = {
    hdr = ProtoField.uint64("ucx_rndv.hdr", "Header", base.HEX),

    -- AM case
    am_id = ProtoField.uint16("ucx_rndv.am.id",
                              "Active Message ID", base.DEC),

    -- AM flags
    reply = ProtoField.uint16("ucx_rndv.am.flags.reply",
                              "Force Reply EP passed to callback",
                              base.BOOL, bitset, bit_shift(0)),
    eager = ProtoField.uint16("ucx_rndv.am.flags.eager",
                              "Force Eager protocol",
                              base.BOOL, bitset, bit_shift(1)),
    rndv = ProtoField.uint16("ucx_rndv.am.flags.rndv",
                              "Force RNDV protocol",
                              base.BOOL, bitset, bit_shift(2)),

    -- AM user
    hdr_len = ProtoField.uint32("ucx_rndv.am.hdr_len",
                                "User Header Size", base.DEC),
    user_hdr = ProtoField.bytes("ucx_rndv.am.user_hdr",
                                "User Header"),

    -- TAG case, missing

    -- RNDV RTS (Ready to Send) / Common
    ep_id = ProtoField.uint64("ucx_rndv.ep_id", "Endpoint ID", base.HEX),
    req_id = ProtoField.uint64("ucx_rndv.req_id", "Request ID", base.HEX),
    addr = ProtoField.uint64("ucx_rndv.addr", "Memory Address", base.HEX),
    size = ProtoField.uint64("ucx_rndv.size", "Data Size", base.DEC),
    op_code = ProtoField.uint8("ucx_rndv.op_code", "Opcode", base.DEC, rndv_opcode),
    rkey = ProtoField.bytes("ucx_rndv.rkey", "Remote Key"),

    -- RNDV RTR (Ready to Receive)
    sreq_id = ProtoField.uint64("ucx_rndv.sreq_id", "Source Request ID", base.HEX),
    rreq_id = ProtoField.uint64("ucx_rndv.rreq_id", "Receive Request ID", base.HEX),
    offset = ProtoField.uint64("ucx_rndv.offset", "Memory Offset", base.DEC),

    -- RNDV ATP (Ack to Put)
    status = ProtoField.uint64("ucx_rndv.status", "Status", base.DEC, ucx_status),
}
ucp_rndv_proto.fields = ucp_rndv_fields

function ucp_rndv_proto_dissect_ack(tvbuf, pinfo, tree, msg)
    local req_id = tvbuf(0, 8)
    local status = tvbuf(8, 1) -- C enum here takes 1 byte
    local size = tvbuf(9, 8)

    tree = tree:add(ucp_rndv_proto, tvbuf)

    local urf = ucp_rndv_fields
    tree:add_le(urf.req_id, req_id)
    tree:add(urf.status, status)
    tree:add_le(urf.size, size)

    local status_str = get_ucx_status(status:uint())

    msg = msg .. ", Req: " .. hex_64(req_id:le_uint64():tohex())
    msg = msg .. ", Status: " .. status_str
    msg = msg .. ", Size: " .. size:le_uint64()
    tree:append_text(", " .. msg)

    pinfo.cols.protocol = "UCX_RNDV"
    pinfo.cols.info = msg
end

function ucp_rndv_proto_dissect_ats(tvbuf, pinfo, tree)
    ucp_rndv_proto_dissect_ack(tvbuf, pinfo, tree, "ATS")
end

function ucp_rndv_proto_dissect_atp(tvbuf, pinfo, tree)
    ucp_rndv_proto_dissect_ack(tvbuf, pinfo, tree, "ATP")
end

function ucp_rndv_proto_dissect_rtr(tvbuf, pinfo, tree)
    local sreq_id = tvbuf(0, 8) -- local req id
    local rreq_id = tvbuf(8, 8) -- remote req id
    local addr = tvbuf(16, 8)
    local size = tvbuf(24, 8)
    local offset = tvbuf(32, 8)
    local off = 40

    local msg = "RTR"
    pinfo.cols.protocol = "UCX_RNDV"
    tree = tree:add(ucp_rndv_proto, tvbuf)

    local urf = ucp_rndv_fields
    tree:add_le(urf.sreq_id, sreq_id)
    tree:add_le(urf.rreq_id, rreq_id)
    tree:add_le(urf.addr, addr)
    tree:add_le(urf.size, size)
    tree:add_le(urf.offset, offset)

    if addr:le_uint64() ~= 0 then
        local rkey = tvbuf(off, tvbuf:len() - off)
        tree:add_le(urf.rkey, rkey)
    end

    msg = msg .. ", SReq: " .. hex_64(sreq_id:le_uint64():tohex())
    msg = msg .. ", RReq: " .. hex_64(rreq_id:le_uint64():tohex())
    msg = msg .. ", Addr: " .. hex_64(addr:le_uint64():tohex())
    msg = msg .. ", Size: " .. size:le_uint64()
    msg = msg .. ", Off: " .. offset:le_uint64()
    tree:append_text(", " .. msg)
    pinfo.cols.info = msg
end

function ucp_rndv_proto_dissect_rts(tvbuf, pinfo, tree)
    local hdr = tvbuf(0, 8)
    local ep_id = tvbuf(8, 8)
    local req_id = tvbuf(16, 8)
    local addr = tvbuf(24, 8)
    local size = tvbuf(32, 8)
    local op_code = tvbuf(40, 1)
    local off = 41

    local msg = "RTS"
    pinfo.cols.protocol = "UCX_RNDV"
    tree = tree:add(ucp_rndv_proto, tvbuf)

    local urf = ucp_rndv_fields

    local hdr_len = nil
    if op_code:uint() == RNDV_AM then
        local am_id = tvbuf(0, 2)
        local flags = tvbuf(2, 2)
        hdr_len = tvbuf(4, 4)

        tree:add_le(urf.am_id, am_id)
        tree:add_le(urf.reply, flags)
        tree:add_le(urf.eager, flags)
        tree:add_le(urf.rndv, flags)
        tree:add_le(urf.hdr_len, hdr_len)

        msg = msg .. ", AM: " .. am_id:le_uint()
        msg = msg .. ", Hdr: " .. hdr_len:le_uint()
    else
        tree:add_le(urf.hdr, hdr)
    end

    tree:add_le(urf.ep_id, ep_id)
    tree:add_le(urf.req_id, req_id)
    tree:add_le(urf.addr, addr)
    tree:add_le(urf.size, size)
    tree:add_le(urf.op_code, op_code)

    local rkey_len = tvbuf:len() - off - hdr_len:le_uint()
    if rkey_len > 0 then
        local rkey = tvbuf(off, rkey_len)
        tree:add(urf.rkey, rkey)
    end

    if hdr_len ~= nil and hdr_len:uint() > 0 then
        local user = tvbuf(off + rkey_len, hdr_len:le_uint())
        tree:add(urf.user_hdr, user)
    end

    msg = msg .. ", Ep: " .. hex_64(ep_id:le_uint64():tohex())
    msg = msg .. ", Req: " .. hex_64(req_id:le_uint64():tohex())
    msg = msg .. ", Size: " .. size:le_uint64()
    msg = msg .. ", Rkey: " .. rkey_len
    msg = msg .. ", Memory: " .. hex_64(addr:le_uint64():tohex())
    tree:append_text(", " .. msg)
    pinfo.cols.info = msg
end

local amo_sw_opcode = {
    [0] = "ADD",
    [1] = "AND",
    [2] = "OR",
    [3] = "XOR",
    [4] = "SWAP",
    [5] = "CSWAP",
}

local ucp_sw_amo_proto = Proto("UCP_SW_AMO", "UCP SW Atomic")
local ucp_amo_fields = {
    addr = ProtoField.uint64("ucp_sw_amo.addr", "Memory Address", base.HEX),
    ep_id = ProtoField.uint64("ucp_sw_amo.ep_id", "Endpoint ID", base.HEX),
    req_id = ProtoField.uint64("ucp_sw_amo.req_id", "Request ID", base.HEX),
    op_code = ProtoField.uint8("ucp_sw_amo.op_code", "Opcode", base.DEC, amo_sw_opcode),
    size = ProtoField.uint8("ucp_sw_amo.size", "Atomic Size", base.DEC),
    old_value = ProtoField.bytes("ucp_sw_amo.old_value", "Old Value"),
    value = ProtoField.bytes("ucp_sw_amo.value", "Value"),
}
ucp_sw_amo_proto.fields = ucp_amo_fields

function ucp_sw_amo_proto_cmpl_dissector(tvbuf, pinfo, tree)
    local ep_id = tvbuf(0, 8)
    local msg = "EP: " .. hex_64(ep_id:le_uint64():tohex())

    pinfo.cols.protocol = "UCP_SW_AMO"
    tree:add(ucp_sw_amo_proto, tvbuf)
        :append_text(", " .. msg)
        :add_le(ucp_amo_fields.ep_id, ep_id)
end

function ucp_sw_amo_proto.dissector(tvbuf, pinfo, tree)
    local uaf = ucp_amo_fields

    pinfo.cols.protocol = "UCP_SW_AMO"
    tree = tree:add(ucp_sw_amo_proto, tvbuf)

    if tonumber(pinfo.private.atomic_rep) == 1 then
        local req_id = tvbuf(0, 8)

        tree:add_le(uaf.req_id, req_id)
        tvbuf = pull(tvbuf, 8)
        tree:add(uaf.value, tvbuf)

        local size = tvbuf:len()
        local msg = hex_64(req_id:le_uint64():tohex())
        msg = "Req: " .. msg .. ", Size: " .. size .. ", Value: " .. tvbuf:bytes():tohex()
        tree:append_text(", " .. msg)
        return
    end

    local addr = tvbuf(0, 8)
    local ep_id = tvbuf(8, 8)
    local req_id = tvbuf(16, 8)
    local length = tvbuf(24, 1)
    local op_code = tvbuf(25, 1)
    local off = 26

    tree:add_le(uaf.addr, addr)
    tree:add_le(uaf.ep_id, ep_id)
    tree:add_le(uaf.req_id, req_id)
    tree:add_le(uaf.size, length)
    tree:add_le(uaf.op_code, op_code)

    local msg = ", Op: " .. amo_sw_opcode[op_code:uint()]
    msg = msg .. ", Size: " .. length:uint()
    msg = msg .. ", Memory: " .. hex_64(addr:le_uint64():tohex())
    msg = msg .. ", Ep: " .. hex_64(ep_id:le_uint64():tohex())
    msg = msg .. ", Req: " .. hex_64(req_id:le_uint64():tohex())

    local CSWAP = 5
    if op_code:uint() == CSWAP then
        local old_value = tv(off, length:uint()) -- length has to be >0
        tree:add(uaf.old_value, old_value)
        off = off + length:uint()
    end
    if length:uint() > 0 then
        local value = tvbuf(off, length:uint())
        tree:add(uaf.value, value)
    end

    tree:append_text(msg)
end


local ucp_stream_proto = Proto("UCX_STREAM", "UCX Stream Data")
local ucp_stream_fields = {
    ep_id = ProtoField.uint64("ucx_stream.ep_id", "Endpoint ID", base.HEX),
    data = ProtoField.bytes("ucx_stream.data", "Data"),
}
ucp_stream_proto.fields = ucp_stream_fields

function ucp_stream_proto.dissector(tvbuf, pinfo, tree)
    pinfo.cols.protocol = "UCX_STREAM"
    tree = tree:add(ucp_stream_proto, tvbuf)

    local ep_id = tvbuf(0, 8)
    local usf = ucp_stream_fields
    local tvbuf = pull(tvbuf, 8)

    local ep_str = string.format("0x%s", ep_id:le_uint64():tohex())
    local msg = ", Ep: " .. ep_str .. ", Data Len: " .. tvbuf:len()
    tree:append_text(msg)

    tree:add_le(usf.ep_id, ep_id)
    tree:add(usf.data, tvbuf)
end


local ucp_addr_proto = Proto("UCP_ADDRESS", "UCP Address")
local ucp_addr_fields = {
    version = ProtoField.uint8("ucp_addr.version", "Version",
                               base.DEC, {[0] = "1", [1] = "2"}, 0x0f),

    am_only_v1 = ProtoField.uint8("ucp_addr.flags.am_only",
                                  "Active Message",
                                  base.BOOL, bitset, 0x80),
    client_id_v1 = ProtoField.uint8("ucp_addr.flag.client_id",
                                    "Client ID",
                                    base.BOOL, bitset, 0x40),
    worker_uuid_v1 = ProtoField.uint8("ucp_addr.flag.worker_uuid",
                                      "Worker UUID",
                                      base.BOOL, bitset, 0x20),
    debug_info_v1 = ProtoField.uint8("ucp_addr.flag.debug_info",
                                     "Debug Info",
                                     base.BOOL, bitset, 0x10),

    worker_uuid = ProtoField.uint64("ucp_addr.worker_uuid",
                                    "Worker UUID",
                                    base.HEX),
    addr_name_len = ProtoField.uint8("ucp_addr.addr_name_len",
                                     "Address Name Len",
                                     base.DEC),
    addr_name = ProtoField.string("ucp_addr.addr_name",
                                  "Address Name",
                                  base.ASCII),

    -- MD Info relate data
    md_flags_empty_dev = ProtoField.uint8(
            "ucp_addr.device.md_info.flag_empty_dev",
            "Empty Device", base.BOOL, bitset, 0x80),
    md_flags_md_alloc = ProtoField.uint8(
            "ucp_addr.device.md_info.flag_md_alloc",
            "MD can Allocate", base.BOOL, bitset, 0x40),
    md_flags_md_reg = ProtoField.uint8(
            "ucp_addr.device.md_info.flag_md_reg",
            "MD can Register", base.BOOL, bitset, 0x20),
    md_index = ProtoField.uint8(
            "ucp_addr.device.md_info.index",
            "Memory Domain Index", base.DEC, nil, 0x0f),

    -- Device related presence flags
    dev_flags_last = ProtoField.uint8(
            "ucp_addr.device.flags_last", "Last Device",
            base.BOOL, bitset, 0x80),
    dev_flags_num_paths = ProtoField.uint8(
            "ucp_addr.device.flags_num_path",
            "Number of paths",
            base.BOOL, bitset, 0x40),
    dev_flags_sys_device = ProtoField.uint8(
            "ucp_addr.device.flags_sys_device",
            "System Device",
            base.BOOL, bitset, 0x20),

    dev_addr_len = ProtoField.uint8(
            "ucp_addr.device.addr_len",
            "Address Length",
            base.DEC, nil, 0x1f),
    num_paths = ProtoField.uint8(
            "ucp_addr.device.num_paths",
            "Number of Paths",
            base.DEC),
    sys_device = ProtoField.uint8(
            "ucp_addr.device.sys_device",
            "System Device",
            base.DEC),
    device_addr = ProtoField.bytes(
            "ucp_addr.device.device_addr",
            "Device Address"),

    -- For each transport layer under a device
    tl_name_csum = ProtoField.uint16(
            "ucp_addr.device.tl.name_csum",
            "Name Checksum", base.HEX, ucp_tl_name),

    -- Under transport: Interface attributes
    iface_attr_overhead = ProtoField.float(
            "ucp_addr.device.tl.iface.overhead",
            "Overhead"),
    iface_attr_bandwidth = ProtoField.float(
            "ucp_addr.device.tl.iface.bandwidth",
            "Bandwidth"),
    iface_attr_latency_overhead = ProtoField.float(
            "ucp_addr.device.tl.iface.latency_overhead",
            "Latency Overhead"),

    iface_attr_flags_prio = ProtoField.uint32(
            "ucp_addr.device.tl.iface.flags.prio",
            "Priority",
            base.DEC, nil, 0x000000ff),
    iface_attr_flags_amo64 = ProtoField.uint32(
            "ucp_addr.device.tl.iface.flags.amo64",
            "Atomic64 Operation",
            base.BOOL, bitset, 0x80000000),
    iface_attr_flags_amo32 = ProtoField.uint32(
            "ucp_addr.device.tl.iface.flags.amo32",
            "Atomic32 Operation",
            base.BOOL, bitset, 0x40000000),
    iface_attr_rsc_index = ProtoField.uint8(
            "ucp_addr.device.tl.iface.rsc_index",
            "Resource Index",
            base.DEC),

    -- Interface Flags
    iface_attr_flags_am_bcopy = ProtoField.uint32(
            "ucp_addr.device.tl.iface.flags.am_bcopy",
            "Active Message Bcopy",
            base.BOOL, bitset, bit_shift(8)),
    iface_attr_flags_pending = ProtoField.uint32(
            "ucp_addr.device.tl.iface.flags.pending",
            "Pending", base.BOOL, bitset, bit_shift(9)),
    iface_attr_flags_put_short = ProtoField.uint32(
            "ucp_addr.device.tl.iface.flags.put_short",
            "Put Short", base.BOOL, bitset, bit_shift(10)),
    iface_attr_flags_put_bcopy = ProtoField.uint32(
            "ucp_addr.device.tl.iface.flags.put_bcopy",
            "Put Bcopy", base.BOOL, bitset, bit_shift(11)),
    iface_attr_flags_put_zcopy = ProtoField.uint32(
            "ucp_addr.device.tl.iface.flags.put_zcopy",
            "Put Zero Copy", base.BOOL, bitset, bit_shift(12)),
    iface_attr_flags_get_short = ProtoField.uint32(
            "ucp_addr.device.tl.iface.flags.get_short",
            "Get Short", base.BOOL, bitset, bit_shift(13)),
    iface_attr_flags_get_bcopy = ProtoField.uint32(
            "ucp_addr.device.tl.iface.flags.get_bcopy",
            "Get Bcopy", base.BOOL, bitset, bit_shift(14)),
    iface_attr_flags_get_zcopy = ProtoField.uint32(
            "ucp_addr.device.tl.iface.flags.get_zcopy",
            "Get Zero Copy", base.BOOL, bitset, bit_shift(15)),
    iface_attr_flags_connect_to_iface = ProtoField.uint32(
            "ucp_addr.device.tl.iface.flags.connect_to_iface",
            "Connect to Interface",
            base.BOOL, bitset, bit_shift(16)),
    iface_attr_flags_cb_sync = ProtoField.uint32(
            "ucp_addr.device.tl.iface.flags.cb_sync",
            "Synchronous Callback from Worker",
            base.BOOL, bitset, bit_shift(17)),
    iface_attr_flags_cb_async = ProtoField.uint32(
            "ucp_addr.device.tl.iface.flags.cb_async",
            "Asynchronuous Callback",
            base.BOOL, bitset, bit_shift(18)),
    iface_attr_flags_tag_eager_bcopy = ProtoField.uint32(
            "ucp_addr.device.tl.iface.flags.tag_eager_bcopy",
            "Tag Eager Bcopy", base.BOOL, bitset, bit_shift(19)),
    iface_attr_flags_tag_rndv_zcopy = ProtoField.uint32(
            "ucp_addr.device.tl.iface.flags.tag_rndv_zcopy",
            "Tag RNDV Zero Copy", base.BOOL, bitset, bit_shift(20)),
    iface_attr_flags_event_recv = ProtoField.uint32(
            "ucp_addr.device.tl.iface.flags.event_recv",
            "Event Receive", base.BOOL, bitset, bit_shift(21)),


    iface_flags_last = ProtoField.uint8(
            "ucp_addr.device.tl.iface.flags_last",
            "Last Interface",
            base.DEC, bitset, 0x80),
    iface_flags_has_ep_addr = ProtoField.uint8(
            "ucp_addr.device.tl.iface.flags_has_ep_addr",
            "Has Endpoint Address",
            base.DEC, bitset, 0x40),
    iface_flags_iface_addr_len = ProtoField.uint8(
            "ucp_addr.device.tl.iface.addr_len",
            "Address Length",
            base.DEC, nil, 0x1f),

    iface_addr = ProtoField.bytes(
            "ucp_addr.device.tl.iface.addr",
            "Address"),

    -- Endpoint addresses for an interface
    ep_addr_len = ProtoField.uint8(
            "ucp_addr.device.tl.iface.ep.addr_len", "Address Length",
            base.DEC),
    ep_addr = ProtoField.bytes(
            "ucp_addr.device.tl.iface.ep.addr", "Address"),
    ep_lane = ProtoField.uint8(
            "ucp_addr.device.tl.iface.ep.lane", "Lane", -- local or remote
            base.DEC, nil, 0x7f),
    ep_last = ProtoField.uint8(
            "ucp_addr.device.tl.iface.ep.last", "Last Endpoint",
            base.DEC, nil, 0x80),
}
ucp_addr_proto.fields = ucp_addr_fields

function ucp_address_md_info(tvbuf, pinfo, tree)
    local byte = tvbuf(0, 1)
    local count = 1

    local tree = tree:add("Memory Domain Info")
    tree:add(ucp_addr_fields.md_flags_empty_dev, byte)
    tree:add(ucp_addr_fields.md_flags_md_alloc, byte)
    tree:add(ucp_addr_fields.md_flags_md_reg, byte)

    local md_str = "("
    if bf_value(byte:uint(), 0x40) == 1 then
        md_str = md_str .. "Alloc"
    end
    if bf_value(byte:uint(), 0x20) == 1 then
        if md_str:len() ~= 1 then
            md_str = md_str .. ", "
        end
        md_str = md_str .. "Reg"
    end
    md_str = md_str .. ")"

    local md_str = "MD: " .. bf_value(byte:uint(), 0xf) .. " " .. md_str

    -- Missing: handle non v1 case where byte was not enough
    tree:add(ucp_addr_fields.md_index, byte)
    local empty_dev = bit.band(byte:uint(), 0x80)
    return pull(tvbuf, count), empty_dev, md_str
end

function ucp_address_iface(tl_name, iface_id, version, tvbuf, pinfo, tree)
    local uaf = ucp_addr_fields

    subtree = tree:add("Interface "):append_text(" " .. iface_id .. ": ")

    -- Interface attributes
    if version == 1 then
        local overhead = tvbuf(0, 4)
        local bandwidth = tvbuf(4, 4)
        local latency_overhead = tvbuf(8, 4)
        local flags = tvbuf(12, 4)

        subtree:add_le(uaf.iface_attr_overhead, overhead)
        subtree:add_le(uaf.iface_attr_bandwidth, bandwidth)
        subtree:add_le(uaf.iface_attr_latency_overhead, latency_overhead)

        stree = subtree:add("Flags")
        stree:add_le(uaf.iface_attr_flags_amo64, flags)
        stree:add_le(uaf.iface_attr_flags_amo32, flags)

        stree:add_le(uaf.iface_attr_flags_event_recv, flags)
        stree:add_le(uaf.iface_attr_flags_tag_rndv_zcopy, flags)
        stree:add_le(uaf.iface_attr_flags_tag_eager_bcopy, flags)
        stree:add_le(uaf.iface_attr_flags_cb_async, flags)
        stree:add_le(uaf.iface_attr_flags_cb_sync, flags)
        stree:add_le(uaf.iface_attr_flags_connect_to_iface, flags)
        stree:add_le(uaf.iface_attr_flags_get_zcopy, flags)
        stree:add_le(uaf.iface_attr_flags_get_bcopy, flags)

        stree:add_le(uaf.iface_attr_flags_get_short, flags)
        stree:add_le(uaf.iface_attr_flags_put_zcopy, flags)
        stree:add_le(uaf.iface_attr_flags_put_bcopy, flags)
        stree:add_le(uaf.iface_attr_flags_put_short, flags)
        stree:add_le(uaf.iface_attr_flags_pending, flags)
        stree:add_le(uaf.iface_attr_flags_am_bcopy, flags)

        stree:add_le(uaf.iface_attr_flags_prio, flags)
    else
        print("Failure: Unknown interface attributes version!")
    end
    local count = 16

    local rsc_index = tvbuf(count, 1)
    count = count + 1
    subtree:add(uaf.iface_attr_rsc_index, rsc_index)

    -- Interface Address and Flags
    local flags = tvbuf(count, 1)
    count = count + 1
    subtree:add(uaf.iface_flags_last, flags)
    subtree:add(uaf.iface_flags_has_ep_addr, flags)
    subtree:add(uaf.iface_flags_iface_addr_len, flags)

    -- Interface address where to connect
    local addr_len = bit.band(flags:uint(), 0x1f)
    local iface_addr = tvbuf(count, addr_len)
    count = count + addr_len

    subtree:add(uaf.iface_addr, iface_addr)
    local msg = ""
    if tl_name == "tcp" then
        msg = msg .. " TCP/" .. iface_addr:uint()
    else
        msg = msg .. " 0x" .. iface_addr:bytes():tohex()
    end
    subtree:append_text(msg)

    -- For unpack each endpoint address
    local last_ep = 1
    if bit.band(flags:uint(), 0x40) ~= 0 then
        last_ep = 0
    end

    local ep_id = 0
    while last_ep == 0 do
        stree = subtree:add("Endpoint Address")
            :append_text(" " .. ep_id)

        local ep_addr_len = tvbuf(count, 1)
        local len = ep_addr_len:uint()
        local ep_addr = tvbuf(count + 1, len)
        count = count + 1 + len
        local lane = tvbuf(count, 1)
        count = count + 1

        stree:add(uaf.ep_addr_len, ep_addr_len)
        stree:add(uaf.ep_addr, ep_addr)
        stree:add(uaf.ep_lane, lane)
        stree:add(uaf.ep_last, lane)

        last_ep = bit.band(lane:uint(), 0x80)
    end

    local last_interface = 0
    if bit.band(flags:uint(), 0x80) ~= 0 then
        last_interface = 1
    end
    return pull(tvbuf, count), last_interface
end

function ucp_address_tl(tl_id, version, tvbuf, pinfo, tree)
    local tl_name_csum = tvbuf(0, 2)
    tvbuf = pull(tvbuf, 2)
    local csum = tl_name_csum:le_uint()
    local tl_name = get_tl_name(csum)
    local tree = tree:add("Transport Layer"):append_text(
        " " .. tl_id .. " (" .. tl_name  .. ")")

    tree:add_le(ucp_addr_fields.tl_name_csum, tl_name_csum)

    local iface_id = 0
        tvbuf, last
            = ucp_address_iface(tl_name, iface_id, version, tvbuf, pinfo, tree)

    local name = get_safe(ucp_tl_name, csum, string.format("0x%02x", csum))
    return tvbuf, last, name
end

function ucp_address_device(id, version, tvbuf, pinfo, tree)
    orig_tree = tree:add("Device " .. id .. ": ")
    tree = orig_tree

    tvbuf, empty_dev, md_str = ucp_address_md_info(tvbuf, pinfo, tree)
    local msg = ""

    local flags = tvbuf(0, 1)

    local last_dev = bit.band(flags:uint(), 0x80)
    local num_path_flags = bit.band(flags:uint(), 0x40)
    local sys_device_flags = bit.band(flags:uint(), 0x20)
    local addr_len = bit.band(flags:uint(), 0x1f)

    tvbuf = pull(tvbuf, 1)

    local subtree = tree:add("Device Flags")
    subtree:add(ucp_addr_fields.dev_flags_last, flags)
    subtree:add(ucp_addr_fields.dev_flags_num_paths, flags)
    subtree:add(ucp_addr_fields.dev_flags_sys_device, flags)
    -- Handle non v1 overflow at once, same as all above
    subtree:add(ucp_addr_fields.dev_addr_len, flags)

    if num_path_flags == 1 then
        tree:add(ucp_addr_fields.num_paths, tvbuf(0, 1))
        tvbuf = pull(tvbuf, 1)
    end
    if sys_device_flags == 1 then
        tree:add(ucp_addr_fields.sys_device, tvbuf(0, 1))
        tvbuf = pull(tvbuf, 1)
    end
    if addr_len > 0 then
        local addr = tvbuf(0, addr_len)
        tree:add(ucp_addr_fields.device_addr, addr)
        tvbuf = pull(tvbuf, addr_len)
        msg = msg .. ", Addr: " .. addr:bytes():tohex()
    end

    -- Decode all the TL/Interfaces
    local tl_id = 0
    local last = empty_dev
    local iface_names = ""
    while last == 0 do
        if tl_id ~= 0 then
            iface_names = iface_names .. "/"
        end
        tvbuf, last, name = ucp_address_tl(tl_id, version, tvbuf, pinfo, tree)
        iface_names = iface_names .. name
        tl_id = tl_id + 1
    end
    msg = md_str .. ", Ifaces: " .. iface_names .. msg
    orig_tree:append_text(msg)
    return tvbuf, last_dev
end

function ucp_addr_proto.dissector(tvbuf, pinfo, tree)
    local ctx = pinfo.private
    local top = tree:add(ucp_addr_proto, tvbuf)
    local tree = top

    local msg = ""
    local byte = tvbuf(0, 1)
    local version = bit.band(byte:uint(), 0x0f)
    local flags = bit.band(byte:uint(), 0xf0)

    hex_str = string.format("%x", flags)
    subtree = tree:add("Address Flags")
            :append_text(": 0x" .. hex_str)

    if version == 0 then
        -- ucp_address v1
        subtree:add(ucp_addr_fields.am_only_v1, flags)
        subtree:add(ucp_addr_fields.client_id_v1, flags)
        subtree:add(ucp_addr_fields.worker_uuid_v1, flags)
        subtree:add(ucp_addr_fields.debug_info_v1, flags)
        flags = flags / 16
    else
        print("Warning: ucp address v2 not supported, skipping flags...")
        tvbuf = tvbuf(1, tvbuff:len() - 1)
    end
    subtree:add(ucp_addr_fields.version, byte)

    tvbuf = tvbuf(1, tvbuf:len() - 1)

    local worker_uuid_flags = bit.band(flags, 0x2)
    if worker_uuid_flags ~= 0 then
        local uuid = tvbuf(0, 8)
        msg = msg .. ", Worker: 0x" .. uuid:bytes():tohex()

        tree:add_le(ucp_addr_fields.worker_uuid, uuid)
        tvbuf = tvbuf(8, tvbuf:len() - 8)

    end

    local client_id_flags = bit.band(flags, 0x4)
    if client_id_flags ~= 0 then
        tree:add_le(ucp_addr_fields.client_id, tvbuf(0, 8))
        tvbuf = tvbuf(8, tvbuf:len() - 8)
    end

    local debug_info_flags = bit.band(flags, 0x1)
    if tonumber(pinfo.private.under_wireup) == 1 then
        if debug_info_flags ~= 0 then
            local len = tvbuf(0, 1)
            local name = tvbuf(1, len:uint())

            msg = msg .. ", Addr: " .. name:string()

            tree:add(ucp_addr_fields.addr_name_len, len)
            tree:add(ucp_addr_fields.addr_name, name)
            tvbuf = pull(tvbuf, len:uint() + 1)
        end
    end

    local id = 0
    if tvbuf(0, 1):uint() ~= 0xff then
        local last_device = 0
        while last_device == 0 do
            tvbuf, last_device
                = ucp_address_device(id, version + 1, tvbuf, pinfo, tree)
            id = id + 1
        end
    end
    msg = msg .. " (" .. id .. " device(s))"
    top:append_text(msg)
end

local wireup_type_str = {
    [0] = "PRE_REQUEST",
    [1] = "REQUEST",
    [2] = "REPLY",
    [3] = "ACK",
    [4] = "EP_CHECK",
    [5] = "EP_REMOVED",
}

local wireup_proto = Proto("UCX_WIREUP", "UCX Wireup")
local wireup_fields = {
    typ = ProtoField.uint8(
            "ucx_wireup.type", "Message Type", base.DEC, wireup_type_str),
    err_mode = ProtoField.uint8(
            "ucx_wireup.err_mode", "Error Mode", base.HEX, {
                [0] = "None", -- no guarantee
                [1] = "Peer", -- completes all sent requests
            }),
    conn_sn = ProtoField.uint16(
            "ucx_wireup.conn_sn", "Connection Sequence Number", base.DEC),
    src_ep_id = ProtoField.uint64(
            "ucx_wireup.src_ep_id", "Source Endpoint ID", base.HEX),
    dst_ep_id = ProtoField.uint64(
            "ucx_wireup.dst_ep_id", "Destination Endpoint ID", base.HEX),
}
wireup_proto.fields = wireup_fields

function wireup_proto.dissector(tvbuf, pinfo, top)
    pinfo.cols.protocol = "UCX_WIREUP"

    local typ = tvbuf(0, 1)
    local err_mode = tvbuf(1, 1)
    local conn_sn = tvbuf(2, 2)
    local src_ep_id = tvbuf(4, 8)
    local dst_ep_id = tvbuf(12, 8)

    local src_ep = string.format("0x%s", src_ep_id:le_uint64():tohex())
    local dst_ep = string.format("0x%s", dst_ep_id:le_uint64():tohex())
    local typ_str = wireup_type_str[typ:uint()]
    local msg = "Type: " .. typ_str
        .. ", Src Ep: " .. src_ep .. ", Dst Ep: " .. dst_ep
        .. ", Con-Seq: " .. conn_sn:le_uint()

    if conn_sn:le_int() == -1 then
        msg = msg .. " (Invalid)"
    end

    tree = top:add(wireup_proto, tvbuf):append_text(", " .. msg)

    local wf = wireup_fields
    tree:add(wf.typ, typ)
    tree:add(wf.err_mode, err_mode)

    tree:add_le(wf.conn_sn, conn_sn):append_text(", " .. msg)
    tree:add_le(wf.src_ep_id, src_ep_id)
    tree:add_le(wf.dst_ep_id, dst_ep_id)

    tvbuf = pull(tvbuf, 20)
    pinfo.private.under_wireup = 1
    pinfo.cols.info = msg
    ucp_addr_proto.dissector(tvbuf:tvb(), pinfo, top)
end

-- Active Messages IDs
local am_id_str = {
    [1]  = "WIREUP",
    [2]  = "EAGER_ONLY", -- only one fragment
    [3]  = "EAGER FIRST",
    [4]  = "EAGER MIDDLE",

    [6]  = "EAGER_SYNC_ONLY",
    [7]  = "EAGER_SYNC_FIRST",
    [8]  = "EAGER_SYNC_ACK",
    [9]  = "RNDV_RTS",  -- request to send
    [10] = "RNDV_ATS", -- ack to send
    [11] = "RNDV_RTR", -- request to receive
    [12] = "RNDV_DATA",

    [14] = "OFFLOAD_SYNC_ACK", -- response (for tag matching offloaded)
    [15] = "STREAM_DATA", -- Eager STREAM
    [16] = "RNDV_ATP",
    [17] = "PUT",
    [18] = "GET_REQ",
    [19] = "GET_REP",

    [20] = "ATOMIC_REQ", -- ucp/rma/amo_sw.c
    [21] = "ATOMIC_REP",
    [22] = "CMPL", -- completion, send on ATOMIC_REQ with req == 0

    [23] = "AM_SINGLE", -- user defined AM, one fragment
    [24] = "AM_FIRST", -- user defined AM, first fragment of many
    [25] = "AM_MIDDLE", -- user defined AM, middle fragment
    [26] = "AM_SINGLE_REPLY",

    -- Control active messages
    [32] = "TCP_CM", -- Connection Management
    [33] = "TCP_PUT_REQ", -- Internal PUT req
    [34] = "TCP_PUT_ACK", -- Internal PUT ack
    [35] = "TCP_KEEPALIVE",
}

local uct_tcp_cm_event = {
    [1] = "REQ",
    [2] = "ACK", -- not sent in case of CONNECTED_TO_EP
    [3] = "ACK+REQ",
    [4] = "FIN",
}

local uct_tcp_proto = Proto("UCX_TCP", "UCX TCP Transport")
local uct_tcp_fields = {
    -- Active Message Part
    am_id = ProtoField.uint8(
            "ucx_tcp.am_id", "Active Message ID", base.DEC, am_id_str),
    am_len = ProtoField.uint32(
            "ucx_tcp.am_length", "Active Message Length", base.DEC),
    am_data = ProtoField.bytes(
            "ucx_tcp.am_data", "Active Message Data"),

    -- TCP Connection Manager Request
    cm_event = ProtoField.uint8(
            "ucx_tcp.cm.event", "TCP Connection Event", base.DEC,
            uct_tcp_cm_event),
    cm_req_flags_connect_to_ep = ProtoField.uint8(
            "ucx_tcp.cm.req.connect_to_ep", "Connect To EP",
            base.BOOL, bitset, 0x01),
    cm_req_conn_sn = ProtoField.uint64(
            "ucx_tcp.cm.req.conn_sn", "Sequence Number", base.DEC),
    cm_req_map_key = ProtoField.uint64(
            "ucx_tcp.cm.req.map_key", "Point Map Key", base.HEX),
    cm_req_sockaddr = ProtoField.bytes(
            "ucx_tcp.cm.req.sockaddr", "Sockaddr"),

    -- TCP PUT REQ
    put_req_addr = ProtoField.uint64(
            "ucx_tcp.put_req.addr", "Memory Address", base.HEX),
    put_req_len = ProtoField.uint64(
            "ucx_tcp.put_req.length", "Data Length", base.DEC),
    put_req_sn = ProtoField.uint64(
            "ucx_tcp.put_req.sn", "Sequence Number", base.DEC),
    put_data = ProtoField.bytes(
            "ucx_tcp.put_data", "Data"),

    -- TCP PUT ACK
    put_ack_sn = ProtoField.uint64(
            "ucx_tcp.put_ack.sn", "Ack Sequence Number", base.DEC),
}
uct_tcp_proto.fields = uct_tcp_fields

local UCT_TCP_MAGIC = UInt64(0x12345678, 0xCAFEBABE)
local UCT_TCP_HDR = 5

function ucp_am_dissector_wireup(tvbuf, pinfo, tree)
    wireup_proto.dissector(tvbuf:tvb(), pinfo, tree)
end

function uct_tcp_am_dissector_cm(tvbuf, pinfo, tree)
    pinfo.cols.protocol = "UCX_TCP_CM"

    local cm_event = tvbuf(0, 4)
    local msg = "Event: " .. uct_tcp_cm_event[cm_event:le_uint()]

    if tvbuf:len() == 4 then
        return msg
    end
    local flags = tvbuf(4, 1)
    local cm_id = tvbuf(5, 8)
    tvbuf = pull(tvbuf, 13)

    local orig_tree = tree:add("Connection Manager")
    tree = orig_tree

    local utf = uct_tcp_fields
    tree:add_le(utf.cm_event, cm_event)
    if cm_event:le_uint() == 1 then
        -- CM Req event
        tree:add(utf.cm_req_flags_connect_to_ep, flags)

        connect_to_ep = bit.band(flags:uint(), 0x01)
        if connect_to_ep ~= 0 then
            tree:add_le(utf.cm_req_map_key, cm_id)
        else
            tree:add_le(utf.cm_req_conn_sn, cm_id)
        end
        tree:add(utf.cm_req_sockaddr, tvbuf)
    end

    if connect_to_ep ~= nil then
        msg = msg .. ", Con-To-EP: " .. connect_to_ep
        if connect_to_ep == 0 then
            msg = msg .. ", Map-Key: " .. cm_id:le_int64()
        else
            msg = msg .. ", Seq: " .. cm_id:le_int64()
        end
    end
    if tvbuf ~= nil then
        orig_tree:append_text(", Sockaddr: " .. tvbuf:bytes():tohex())
    end
    return msg
end

function ucp_am_dissector_stream(tvbuf, pinfo, tree)
    ucp_stream_proto.dissector(tvbuf:tvb(), pinfo, tree)
end

function uct_tcp_am_dissector_put_req(tvbuf, pinfo, tree)
    local addr = tvbuf(0, 8)
    local len = tvbuf(8, 8)
    local sn = tvbuf(16, 4)

    local tree = tree:add("Put Req")
    tree:add_le(uct_tcp_fields.put_req_addr, addr)
    tree:add_le(uct_tcp_fields.put_req_len, len)
    tree:add_le(uct_tcp_fields.put_req_sn, sn)

    local length = len:le_uint64():tonumber()
    if length > 0 then
        local data = tvbuf(20, length)
        tree:add_le(uct_tcp_fields.put_data, data)
    end

    local addr_str = string.format("0x%s", addr:le_uint64():tohex())
    local msg = "Seq: " .. sn:le_uint()
            .. ", Mem Addr: " .. addr_str .. ", Data Len: " .. len:le_uint64()

    return msg
end

function uct_tcp_am_dissector_put_ack(tvbuf, pinfo, tree)
    local sn = tvbuf(0, 4)

    local tree = tree:add("Put Ack")
    tree:add_le(uct_tcp_fields.put_ack_sn, sn)

    local msg = "Seq: " .. sn:le_uint()
    return msg
end

function ucp_am_dissector_atomic_req(tvbuf, pinfo, tree)
    ucp_sw_amo_proto.dissector(tvbuf:tvb(), pinfo, tree)
end

function ucp_am_dissector_atomic_rep(tvbuf, pinfo, tree)
    pinfo.private.atomic_rep = 1
    ucp_sw_amo_proto.dissector(tvbuf:tvb(), pinfo, tree)
end

function ucp_am_compl(tvbuf, pinfo, tree)
    ucp_sw_amo_proto_cmpl_dissector(tvbuf:tvb(), pinfo, tree)
end

function ucp_am_rndv_rtr(tvbuf, pinfo, tree)
    ucp_rndv_proto_dissect_rtr(tvbuf:tvb(), pinfo, tree)
end

function ucp_am_rndv_rts(tvbuf, pinfo, tree)
    ucp_rndv_proto_dissect_rts(tvbuf:tvb(), pinfo, tree)
end

function ucp_am_rndv_atp(tvbuf, pinfo, tree)
    ucp_rndv_proto_dissect_atp(tvbuf:tvb(), pinfo, tree)
end

function ucp_am_rndv_ats(tvbuf, pinfo, tree)
    ucp_rndv_proto_dissect_ats(tvbuf:tvb(), pinfo, tree)
end

local ucp_am_dissector = {
    [1]  = ucp_am_dissector_wireup,

    [9]  = ucp_am_rndv_rts,
    [10] = ucp_am_rndv_ats,
    [11] = ucp_am_rndv_rtr,

    [15] = ucp_am_dissector_stream,
    [16] = ucp_am_rndv_atp,

    -- Atomic SW, on top of AM
    [20] = ucp_am_dissector_atomic_req,
    [21] = ucp_am_dissector_atomic_rep,
    [22] = ucp_am_compl,

    -- TCP Specific
    [32] = uct_tcp_am_dissector_cm,
    [33] = uct_tcp_am_dissector_put_req,
    [34] = uct_tcp_am_dissector_put_ack,
}

function try_am_dissector(am_id_val, tvbuf, pinfo, tree, orig_tree, msg)
    local func = ucp_am_dissector[am_id_val]
    if func ~= nil then
        local sub_msg = func(tvbuf, pinfo, tree)
        if sub_msg ~= nil then
            orig_tree:append_text(", " .. sub_msg)
            pinfo.cols.info = msg .. ", " .. sub_msg
        end
    end
end

function uct_tcp_dissect(tvbuf, pinfo, tree)
    local am_id = tvbuf(0, 1)
    local am_id_val = am_id:uint()
    local am_len = tvbuf(1, 4)
    local am_data = tvbuf(5, tvbuf:len() - 5)

    local msg = "Id: " .. am_id_str[am_id_val]

    top = tree
    orig_tree = tree:add(uct_tcp_proto, tvbuf)
    tree = orig_tree

    tree:add(uct_tcp_fields.am_id, am_id)
    tree:add_le(uct_tcp_fields.am_len, am_len)
    tree:add(uct_tcp_fields.am_data, am_data)

    tvbuf = pull(tvbuf, 5)

    local tcp_specific = {[32] = 1, [33] = 1, [34] = 1, [35] = 1}
    if tcp_specific[am_id_val] == 1 then
        top = tree
    else
        msg = msg .. ", Len: " .. am_len:le_uint()
    end
    orig_tree:append_text(", " .. msg)

    pinfo.cols.info = msg
    pinfo.cols.protocol = uct_tcp_proto.name

    try_am_dissector(am_id_val, tvbuf, pinfo, top, orig_tree, msg)
end

-- Main UCT TCP Dissector entry point
function uct_tcp_desegment(tvbuf, pinfo, tree)

    -- Older de-segment API:
    --   used due to the put data being inlined after the put req TLV

    if tvbuf:len() < UCT_TCP_HDR then
        pinfo.desegment_offset = 0
        pinfo.desegment_len = DESEGMENT_ONE_MORE_SEGMENT
        return
    end

    local off = 0
    local magic = tvbuf(0, 8):le_uint64()
    if magic == UCT_TCP_MAGIC then
        off = off + 8
    end

    while off + UCT_TCP_HDR <= tvbuf:len() do

        local am_id = tvbuf(off, 1):uint()
        local am_len = tvbuf(off + 1, 4):le_int()
        local size = UCT_TCP_HDR + am_len

        if off + size > tvbuf:len() then
            break
        end

        local PUT_REQ_ID = 33
        if am_id == PUT_REQ_ID then
            -- Overall length with actual data sent
            local put_len = tvbuf(off + UCT_TCP_HDR + 8, 8)
            size = size + put_len:le_uint64():tonumber()

            if off + size > tvbuf:len() then
                break
            end
        end

        local am_tvbuf = tvbuf(off, size)
        uct_tcp_dissect(am_tvbuf, pinfo, tree)
        off = off + size
    end

    if off < tvbuf:len() then
        pinfo.desegment_offset = off
        pinfo.desegment_len = DESEGMENT_ONE_MORE_SEGMENT
    end
end

function get_uct_tcp_len(tvbuf, pinfo, offset)
    local magic = tvbuf(offset, 4):le_uint()
    if magic == 0x12345678 then
        return 8 -- we still need to consume the magic
    end
    local len = tvbuf(offset + 1, 4):le_uint()
    return len + UCT_TCP_HDR
end

function uct_tcp_proto.dissector(tvbuf, pinfo, tree)
    uct_tcp_desegment(tvbuf, pinfo, tree)
end

local uct_tcp_conn_track = {}

function uct_tcp_checker(tvbuf, pinfo, tree)
    -- Ideally we should track the 5-tuple
    if uct_tcp_conn_track[pinfo.src_port] ~= nil then
        return true
    end
    if uct_tcp_conn_track[pinfo.dst_port] ~= nil then
        return true
    end

    -- We expect at least first 8 bytes to arrive in one segment
    if tvbuf:len() < 8 then
        return false
    end
    local magic = tvbuf(0, 8):le_uint64()
    if magic ~= UCT_TCP_MAGIC then
        return false
    end

    table.insert(uct_tcp_conn_track, pinfo_src_port)
    table.insert(uct_tcp_conn_track, pinfo_dst_port)
    DissectorTable.get("tcp.port"):add(pinfo.src_port, uct_tcp_proto)
    DissectorTable.get("tcp.port"):add(pinfo.dst_port, uct_tcp_proto)

    uct_tcp_desegment(tvbuf, pinfo, tree)
    return true
end

uct_tcp_proto:register_heuristic("tcp", uct_tcp_checker)


local wireup_sa_err_handling = {
    [0] = "None",
    [1] = "Peer",
}
local wireup_sa_addr_mode = {
    [0] = "Default",
    [1] = "CM-based",
}

local wireup_sa_proto = Proto("UCX_WIREUP_SA", "UCX Wireup Sockaddr")
local wireup_sa_fields = {
    ep_id = ProtoField.uint64("ucx_wireup_sa.ep_id", "Endpoint ID",
                              base.HEX),
    version = ProtoField.uint8("ucx_wireup_sa.version", "Version",
                               base.DEC, {[0] = "1", [1] = "2"}, 0xe0),
    error_handling = ProtoField.uint8("ucx_wireup_sa.error_handling",
                                      "Error Handling Mode",
                                      base.DEC, wireup_sa_err_handling,
                                      0x1f),
    addr_mode = ProtoField.uint8("ucx_wireup_sa.addr_mode",
                                 "Address Mode",
                                 base.BOOL, wireup_sa_addr_mode,
                                 0x2),
    dev_index = ProtoField.uint8("ucx_wireup_sa.dev_index",
                                 "Device Index",
                                 base.DEC),
}
wireup_sa_proto.fields = wireup_sa_fields

function bf_value(byte, mask)
    local value = bit.band(byte, mask)

    while bit.band(mask, 1) == 0 do
        mask = bit.rshift(mask, 1)
        value = bit.rshift(value, 1)
    end
    return value
end

function wireup_sa_proto.dissector(tvbuf, pinfo, top_tree)
    local ep_id = tvbuf(0, 8)
    local header = tvbuf(8, 1)
    local addr_mode = tvbuf(9, 1)
    local dev_index = tvbuf(10, 1)

    local version = bf_value(header:uint(), 0xe) + 1
    local error_handling = bf_value(header:uint(), 0x1f)
    local addr_mode_num = bf_value(addr_mode:uint(), 0x02)

    local ep_str = string.format("0x%s", ep_id:le_uint64():tohex())
    local error_str = wireup_sa_err_handling[error_handling]
    local addr_str = wireup_sa_addr_mode[addr_mode_num]

    local msg = "Version " .. version .. ", Ep: " .. ep_str
        .. ", Err Mode: " .. error_str .. ", Addr Mode: " .. addr_str

    tree = top_tree:add(wireup_sa_proto, tvbuf)
        :append_text(", " .. msg)
    tree:add_le(wireup_sa_fields.ep_id, ep_id)
    tree:add(wireup_sa_fields.version, header)
    tree:add(wireup_sa_fields.error_handling, header)

    pinfo.cols.protocol = wireup_sa_proto.name
    pinfo.cols.info = msg

    local offset = 9
    if version == 1 then
        -- wireup sockaddr v1
        tree:add(wireup_sa_fields.addr_mode, addr_mode)
        tree:add(wireup_sa_fields.dev_index, dev_index)
        offset = offset + 2
    else
        print("Failure: wireup sockaddr v2 not supported!")
    end

    address = pull(tvbuf, offset)
    pinfo.private.under_wireup = 0
    ucp_addr_proto.dissector(address:tvb(), pinfo, top_tree)
end


local sockcm_proto = Proto("UCX_SOCKCM", "UCX Sock Connection Manager")
local SOCKCM_HDR_LEN = 16
local sockcm_fields = {
    len = ProtoField.uint64("ucx_sockcm.length", "Data Length",
                            base.DEC),
    status = ProtoField.uint8("ucx_sockcm.status", "Status",
                              base.DEC, ucx_status),
}
sockcm_proto.fields = sockcm_fields

function sockcm_dissect(tvbuf, pinfo, tree)
    pinfo.cols.protocol = sockcm_proto.name
    pinfo.cols.info = ""

    local sockcm_len = get_sockcm_len(tvbuf, pinfo, 0)
    local sockcm = tvbuf(0, sockcm_len)
    local len = tvbuf(0, 8)
    local status = tvbuf(8, 1)
    local data_len = len:le_uint64():tonumber()
    local data = tvbuf(16, data_len)

    local status_str = get_ucx_status(status:uint())
    local msg = "Len: " .. data_len .. ", Status: " .. status_str
    pinfo.cols.info = msg

    local subtree = tree:add(sockcm_proto, sockcm)
                        :append_text(", " .. msg)
    subtree:add_le(sockcm_fields.len, len)
    subtree:add(sockcm_fields.status, status)
    if data_len > 0 then
        wireup_sa_proto.dissector(data:tvb(), pinfo, tree)
    end
    return sockcm_len, 0
end

function get_sockcm_len(tvbuf, pinfo, offset)
    local len = tvbuf(offset, 8):le_uint64()
    return len:tonumber() + SOCKCM_HDR_LEN
end

function sockcm_proto.dissector(tvbuf, pinfo, tree)
    local desegment = true -- Work on desegmented data
    dissect_tcp_pdus(tvbuf,
                     tree,
                     SOCKCM_HDR_LEN,
                     get_sockcm_len,
                     sockcm_dissect,
                     desegment)
    return tvbuf:len()
end

local tcp_port = DissectorTable.get("tcp.port")
-- Warning: ucx_perftest also uses :13337 to send non-sockcm payloads
tcp_port:add(13337, sockcm_proto)


-- IB Specific dissections
--------------------------

ud_ctl_proto = Proto("UCX_UD_CTL", "UCX UD Control")
local ud_ctl_fields = {
    typ = ProtoField.uint8("ucx_ud_ctl.type", "Control Type", base.DEC,
                           {[1] = "CREQ", [2] = "CREP"}),

    qp_num = ProtoField.uint32("ucx_ud_ctl.ep_addr.qp_num", "QP Number",
                               base.HEX),
    ep_id = ProtoField.uint32("ucx_ud_ctl.ep_addr.ep_id", "Ep ID",
                              base.DEC),
    src_ep_id = ProtoField.uint32("ucx_ud_ctl.ep_addr.src_ep_id", "Src Ep ID",
                                  base.DEC),

    peer_name = ProtoField.string("ucx_ud_ctl.peer_name",
                                  "Peer Name", base.ASCII),

    peer_id = ProtoField.int32("ucx_ud_ctl.peer_id", "Peer Id", base.DEC),
    conn_sn = ProtoField.int32("ucx_ud_ctl.conn_sn", "Connection Serial Number",
                               base.DEC),
    path_idx = ProtoField.int32("ucx_ud_ctl.path_idx", "Path Index", base.DEC),

    ib_addr = ProtoField.bytes("ucx_ud_ctl.ib_addr", "Infiniband Address"),
}
ud_ctl_proto.fields = ud_ctl_fields

local UD_CTL_CREQ = 1
local UD_CTL_CREP = 2

function ud_ctl_proto.dissector(tvbuf, pinfo, tree)
    local typ = tvbuf(0, 1)

    tree = tree:add(ud_ctl_proto, tvbuf)

    local ucf = ud_ctl_fields
    tree:add(ucf.typ, typ)

    local msg = ""
    if typ:uint() == UD_CTL_CREQ then

        local qp_num = tvbuf(4, 3)
        local ep_id = tvbuf(7, 3)

        local conn_sn = tvbuf(12, 4)
        local path_idx = tvbuf(16, 1)

        local off = 20
        local peer_name = tvbuf(off, 16)
        local peer_id = tvbuf(off + 16, 4)
        local ib_addr = pull(tvbuf, off + 20)

        tree:add_le(ucf.qp_num, qp_num)
        tree:add_le(ucf.ep_id, ep_id)
        tree:add_le(ucf.path_idx, path_idx)

        tree:add(ucf.peer_name, peer_name)
        tree:add_le(ucf.peer_id, peer_id)

        tree:add_le(ucf.conn_sn, conn_sn)
        tree:add_le(ucf.ib_addr, ib_addr)

        msg = msg .. "Type: CREQ"
        msg = msg .. ", QPN: " .. string.format("0x%x", qp_num:le_uint())
        msg = msg .. ", Ep ID: " .. ep_id:le_uint()
        msg = msg .. ", Conn SN: " .. conn_sn:le_uint()
        msg = msg .. ", Peer: " .. string.format(
            "%s:%d", peer_name:string(), peer_id:le_uint())
    end
    if typ:uint() == UD_CTL_CREP then

        local src_ep_id = tvbuf(4, 4)
        local peer_name = tvbuf(20, 16)
        local peer_id = tvbuf(36, 4)

        tree:add_le(ucf.src_ep_id, src_ep_id)
        tree:add(ucf.peer_name, peer_name)
        tree:add_le(ucf.peer_id, peer_id)
        msg = msg .. "Type: CREP"
        msg = msg .. ", Src Ep ID: " .. src_ep_id:le_uint()
        msg = msg .. ", Peer: " .. string.format(
            "%s:%d", peer_name:string(), peer_id:le_uint())
    end

    tree:append_text(", " .. msg)
    pinfo.cols.protocol = ud_ctl_proto.name
    pinfo.cols.info = msg
end

rc_proto = Proto("UCX_RC", "UCX RC Protocol")
local rc_fields = {
    tmh_opcode = ProtoField.uint8("ucx_rc.tmh_opcode", "TMH Opcode", base.DEC),
    am_id = ProtoField.uint8("ucx_rc.am_id", "Active Message ID", base.DEC),

    -- Flags
    fc_soft_req = ProtoField.uint8("ucx_rc.flags.fc_soft_req", "FC Soft Req",
                                   base.BOOL, bitset, bit_shift(5)),
    fc_hard_req = ProtoField.uint8("ucx_rc.flags.fc_hard_req", "FC Hard Req",
                                   base.BOOL, bitset, bit_shift(6)),
    fc_grant = ProtoField.uint8("ucx_rc.flags.fc_grant", "FC Grant",
                                base.BOOL, bitset, bit_shift(7)),
    fc_pure_grant = ProtoField.uint8("ucx_rc.flags.fc_pure_grant",
                                     "FC Pure Grant",
                                     base.BOOL,
                                     {[256-32] = "Set", [0] = "Not Set"},
                                     256-32),
}
rc_proto.fields = rc_fields

function rc_proto.dissector(tvbuf, pinfo, tree)
    local tmh_opcode = tvbuf(0, 1)
    local am_id = tvbuf(1, 1)

    am_id = bit.band(am_id:uint(), (bit_shift(5) - 1))

    local orig_tree = tree
    tree = tree:add(rc_proto, tvbuf)
    tree:add(rc_fields.tmh_opcode, tmh_opcode)
    tree:add(rc_fields.am_id, am_id)

    stree = tree:add("Flags")
    stree:add(rc_fields.fc_soft_req, am_id)
    stree:add(rc_fields.fc_hard_req, am_id)
    stree:add(rc_fields.fc_grant, am_id)
    stree:add(rc_fields.fc_pure_grant, am_id)

    local msg = ""
    msg = msg .. "AM: ".. am_id

    pinfo.cols.protocol = rc_proto.name
    pinfo.cols.info = msg
    tree:append_text(", ".. msg)

    tvbuf = pull(tvbuf, 2)

    try_am_dissector(am_id, tvbuf, pinfo, orig_tree, orig_tree, msg)
end

ud_neth_proto = Proto("UCX_UD_NETH", "UCX UD Network Header")
local ud_neth_fields = {
    dst_id = ProtoField.uint32("ucx_ud_neth.dst_id", "Destination ID",
                               base.DEC),

    -- Flags
    am = ProtoField.uint32("ucx_ud_neth.packet_type.am", "Active Message",
                           base.BOOL, bitset, bit_shift(24)),
    ack_req = ProtoField.uint32("ucx_ud_neth.packet_type.ack_req",
                                "Ack Request",
                                base.BOOL, bitset, bit_shift(25)),
    ecn = ProtoField.uint32("ucx_ud_neth.packet_type.ecn",
                            "Explicit Congestion Notification",
                            base.BOOL, bitset, bit_shift(26)),
    nack = ProtoField.uint32("ucx_ud_neth.packet_type.nack", "Negative ACK",
                             base.BOOL, bitset, bit_shift(27)),
    put = ProtoField.uint32("ucx_ud_neth.packet_type.put", "Put Message",
                            base.BOOL, bitset, bit_shift(28)),
    ctl = ProtoField.uint32("ucx_ud_neth.packet_type.ctl", "Control Message",
                            base.BOOL, bitset, bit_shift(29)),

    am_id = ProtoField.uint32("ucx_ud_neth.am_id", "Active Message ID",
                              base.DEC),
    -- Sequences
    psn = ProtoField.uint16("ucx_ud_neth.psn", "Packet Sequence Number",
                            base.DEC),
    ack_psn = ProtoField.uint16("ucx_ud_neth.ack_psn",
                                "Packet Sequence Number ACK",
                                base.DEC),
}
ud_neth_proto.fields = ud_neth_fields

local ib_bth_opcode_f = Field.new("infiniband.bth.opcode")

local IB_BTH_OPCODE_RC_SEND_FIRST = 0
local IB_BTH_OPCODE_RC_SEND_ONLY  = 4

local IB_BTH_OPCODE_UD_SEND = 100

function add_bit(msg, flags, shift, str)
    if bit.band(flags:le_uint(), bit_shift(shift)) ~= 0 then
        if string.len(msg) > 0 then
            msg = msg .. ","
        end
        return msg .. " " .. str
    end
    return msg
end

function ud_neth_proto.dissector(tvbuf, pinfo, tree)
    local packet_type = tvbuf(0, 4)
    local psn = tvbuf(4, 2)
    local ack_psn = tvbuf(6, 2)

    local orig_tree = tree
    tree = tree:add(ud_neth_proto, tvbuf(0, 8))
    local udf = ud_neth_fields

    local dst_id = bit.band(packet_type:le_uint(), 0xffffff)
    local hex_dst_id = string.format("%d", dst_id)

    local msg = ""
    msg = msg .. "Dst ID: " .. hex_dst_id

    local dst_id = bit.band(packet_type:le_uint(), 0xffffff) / 1
    tree:add_le(udf.dst_id, dst_id)

    -- Flags
    local stree = tree:add("Flags:")

    local flag_msg = ""
    local am_id = nil

    if bit.band(packet_type:le_uint(), bit_shift(24)) ~= 0 then
        stree:add_le(udf.am, packet_type)
        am_id = bit.rshift(packet_type:le_uint(), 27)
        tree:add_le(udf.am_id, am_id)
        flag_msg = add_bit(flag_msg, packet_type, 24, "AM")

        msg = msg .. ", AM: " .. am_id
    else
        stree:add_le(udf.ack_req, packet_type)
        stree:add_le(udf.ecn, packet_type)
        stree:add_le(udf.nack, packet_type)
        stree:add_le(udf.put, packet_type)
        stree:add_le(udf.ctl, packet_type)
        flag_msg = add_bit(flag_msg, packet_type, 25, "ACK_REQ")
        flag_msg = add_bit(flag_msg, packet_type, 26, "ECN")
        flag_msg = add_bit(flag_msg, packet_type, 27, "NACK")
        flag_msg = add_bit(flag_msg, packet_type, 28, "PUT")
        flag_msg = add_bit(flag_msg, packet_type, 29, "CTL")

        if string.len(flag_msg) > 0 then
            msg = msg .. ", Flags:" .. flag_msg
        end
    end
    stree:append_text(flag_msg)

    tree:add_le(udf.psn, psn)
    tree:add_le(udf.ack_psn, ack_psn)

    msg = msg .. ", PSN: " .. psn:le_uint()
    msg = msg .. ", ACK_PSN: " .. ack_psn:le_uint()

    tree:append_text(", " .. msg)
    pinfo.cols.protocol = ud_neth_proto.name
    pinfo.cols.info = msg

    local is_ud_ctl = bit.band(packet_type:le_uint(), bit_shift(29))
    tvbuf = pull(tvbuf, 8)
    if is_ud_ctl ~= 0 then
        ud_ctl_proto.dissector(tvbuf:tvb(), pinfo, orig_tree)
        return
    end

    try_am_dissector(am_id, tvbuf, pinfo, orig_tree, orig_tree, msg)
end

function ib_proto_heuristic_checker(tvbuf, pinfo, tree)
    if tvbuf:len() < 8 then
        return false
    end
    local opcode = ib_bth_opcode_f()()

    if opcode == IB_BTH_OPCODE_UD_SEND then
        ud_neth_proto.dissector(tvbuf, pinfo, tree)
        return true;
    end

    local is_rc = {
        [IB_BTH_OPCODE_RC_SEND_FIRST] = 1,
        [IB_BTH_OPCODE_RC_SEND_ONLY]  = 1,
    }

    if is_rc[opcode] ~= nil then
        rc_proto.dissector(tvbuf, pinfo, tree)
        return true;
    end
    return false
end

ud_neth_proto:register_heuristic("infiniband.payload",
                                 ib_proto_heuristic_checker)
