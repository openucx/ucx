--
-- Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2023. ALL RIGHTS RESERVED.
-- See file LICENSE for terms.
--

-- Usage:
-- $ wireshark -Xlua_script:rmpp.lua <pcap_file>
--
-- or copy the LUA file in the Wireshark plugins directory.
--   - Help -> About Wireshark -> Folders -> Personal Lua plugins

local rmpp_version_str = {
    [0] = "Not Set",
    [1] = "1",
}

local RMPP_STATUS_SUCCESS    =   0
local RMPP_STATUS_RESX       =   1
local RMPP_STATUS_T2L        = 118
local RMPP_STATUS_BAD_LEN    = 119
local RMPP_STATUS_BAD_SEG    = 120
local RMPP_STATUS_BADT       = 121
local RMPP_STATUS_W2S        = 122
local RMPP_STATUS_S2B        = 123
local RMPP_STATUS_BAD_STATUS = 124
local RMPP_STATUS_UNV        = 125
local RMPP_STATUS_TMR        = 126
local RMPP_STATUS_UNSPEC     = 127

local rmpp_status_str = {
    [RMPP_STATUS_SUCCESS]    = "Success",
    [RMPP_STATUS_RESX]       = "Resx",
    [RMPP_STATUS_T2L]        = "T2L",
    [RMPP_STATUS_BAD_LEN]    = "Bad Len",
    [RMPP_STATUS_BAD_SEG]    = "Bad Seq",
    [RMPP_STATUS_BADT]       = "BadT",
    [RMPP_STATUS_W2S]        = "W2S",
    [RMPP_STATUS_S2B]        = "S2B",
    [RMPP_STATUS_BAD_STATUS] = "Bad Status",
    [RMPP_STATUS_UNV]        = "Unv",
    [RMPP_STATUS_TMR]        = "TMR",
    [RMPP_STATUS_UNSPEC]     = "Unspec",
}

local RMPP_TYPE_DATA  = 1
local RMPP_TYPE_ACK   = 2
local RMPP_TYPE_STOP  = 3
local RMPP_TYPE_ABORT = 4

local rmpp_type_str = {
    [RMPP_TYPE_DATA]  = "Data",
    [RMPP_TYPE_ACK]   = "Ack",
    [RMPP_TYPE_STOP]  = "Stop",
    [RMPP_TYPE_ABORT] = "Abort",
}

local RMPP_FLAG_ACTIVE = 1
local RMPP_FLAG_FIRST  = 2
local RMPP_FLAG_LAST   = 4

local rmpp_flag_str = {
    [RMPP_FLAG_ACTIVE] = "Active",
    [RMPP_FLAG_FIRST]  = "First",
    [RMPP_FLAG_LAST]   = "Last",
}

local rmpp_flag_set = {
    [0] = "Not Set",
    [1] = "Set",
}

local rmpp_proto = Proto("RMPP", "Reliable Multi-Packet Transaction Protocol")
local rmpp_fields = {
    version = ProtoField.uint8("rmpp.version", "Version", base.DEC,
                               rmpp_version_str),
    typ = ProtoField.uint8("rmpp.type", "Type", base.DEC, rmpp_type_str),

    -- Flags
    active = ProtoField.uint8("rmpp.flag.active", "Active",
                              base.BOOL, rmpp_flag_set, RMPP_FLAG_ACTIVE),
    first = ProtoField.uint8("rmpp.flag.first", "First",
                             base.BOOL, rmpp_flag_set, RMPP_FLAG_FIRST),
    last = ProtoField.uint8("rmpp.flag.last", "Last",
                            base.BOOL, rmpp_flag_set, RMPP_FLAG_LAST),

    status = ProtoField.uint8("rmpp.status", "Status", base.DEC,
                              rmpp_status_str),

    seg_num = ProtoField.uint32("rmpp.seg_num", "Segment Number", base.DEC),
    paylen_newwin = ProtoField.uint32("rmpp.paylen_newwin", "Length", base.DEC),

    data = ProtoField.bytes("rmpp.data", "RMPP Data"),

    reserved = ProtoField.bytes("rmpp.vendor.reserved", "Reserved"),
    oui = ProtoField.bytes("rmpp.vendor.oui", "OUI"),
    vendor_data = ProtoField.bytes("rmpp.vendor.data", "Vendor Data"),
}
rmpp_proto.fields = rmpp_fields

local IB_MGMT_RMPP_DATA_SIZE = 220

local MAD_HDR_SIZE = 24
local RMPP_HDR_SIZE = 12 -- plus alignment

local ib_mad_mgmtclass_f = Field.new("infiniband.mad.mgmtclass")
local ib_mad_data_f = Field.new("infiniband.mad.data")

function rmpp_post_dissector_check(tvbuf)
    local mgmt_class = ib_mad_mgmtclass_f()
    if mgmt_class == nil or mgmt_class() ~= 0x40 then
        return nil
    end
    local data = ib_mad_data_f()
    if data == nil then
        return nil
    end
    return data.range(), mgmt_class()
end

-- MAD payload offsets
local IB_MGMT_MAD_HDR    = 24
local IB_MGMT_VENDOR_HDR = 40
local IB_MGMT_SA_HDR     = 56
local IB_MGMT_DEVICE_HDR = 64

local IB_MGMT_CLASS_SUBN_ADM            = 0x03
local IB_MGMT_CLASS_DEVICE_MGMT         = 0x06
local IB_MGMT_CLASS_DEVICE_ADM          = 0x10
local IB_MGMT_CLASS_BIS                 = 0x12
local IB_MGMT_CLASS_VENDOR_RANGE2_START = 0x30
local IB_MGMT_CLASS_VENDOR_RANGE2_END   = 0x4F

local mad_offset = {
    [IB_MGMT_CLASS_SUBN_ADM]    = IB_MGMT_SA_HDR,
    [IB_MGMT_CLASS_DEVICE_MGMT] = IB_MGMT_DEVICE_HDR,
    [IB_MGMT_CLASS_DEVICE_ADM]  = IB_MGMT_DEVICE_HDR,
    [IB_MGMT_CLASS_BIS]         = IB_MGMT_DEVICE_HDR,
}

function mad_mgmt_class_vendor(mgmt_class)
    return mgmt_class >= IB_MGMT_CLASS_VENDOR_RANGE2_START and
        mgmt_class <= IB_MGMT_CLASS_VENDOR_RANGE2_END
end

function mad_data_offset(mgmt_class)
    local offset = mad_offset[mgmt_class]
    if offset ~= nil then
        return offset
    end
    if mgmt_class >= IB_MGMT_CLASS_VENDOR_RANGE2_START and
        mgmt_class <= IB_MGMT_CLASS_VENDOR_RANGE2_END then
        return IB_MGMT_VENDOR_HDR
    end
    return IB_MGMT_MAD_HDR
end

function rmpp_proto.dissector(tvbuf, pinfo, tree)
    tvbuf, mgmt_class = rmpp_post_dissector_check(tvbuf)
    if tvbuf == nil then
        return
    end

    tree = tree:add(rmpp_proto, tvbuf)

    local version = tvbuf(0, 1)
    local typ = tvbuf(1, 1)
    local rtime_flags = tvbuf(2, 1)
    local status = tvbuf(3, 1)
    local seg_num = tvbuf(4, 4)
    local paylen_newwin = tvbuf(8, 4)

    local size = paylen_newwin:uint()
    local flags = rtime_flags:uint()

    local is_first = bit.band(flags, RMPP_FLAG_FIRST) ~= 0
    local is_last = bit.band(flags, RMPP_FLAG_LAST) ~= 0
    local is_active = bit.band(flags, RMPP_FLAG_ACTIVE) ~= 0
    local is_middle = not is_first and not is_last

    local rmpp_data_size = tvbuf:len() - RMPP_HDR_SIZE

    if not is_middle or size > 0 then
        if rmpp_data_size > size then
            rmpp_data_size = size
        end
    end

    local data = tvbuf(12, rmpp_data_size)

    local reserved = tvbuf(12, 1)
    local oui = tvbuf(13, 3)

    tree:add(rmpp_fields.version, version)
    tree:add(rmpp_fields.typ, typ)
    tree:add(rmpp_fields.active, rtime_flags)
    tree:add(rmpp_fields.first, rtime_flags)
    tree:add(rmpp_fields.last, rtime_flags)
    tree:add(rmpp_fields.status, status)
    tree:add(rmpp_fields.seg_num, seg_num)

    local entry = tree:add(rmpp_fields.paylen_newwin, paylen_newwin)

    local seg_count = nil
    if is_first == true or typ:uint() == RMPP_TYPE_ACK then
        seg_count = size + IB_MGMT_RMPP_DATA_SIZE - 1
        seg_count = math.floor(seg_count / IB_MGMT_RMPP_DATA_SIZE)

        entry:append_text(" (" .. seg_count .. " segment(s))")
    end

    tree:add(rmpp_fields.data, data)

    local vendor_msg = ""
    if mad_mgmt_class_vendor(mgmt_class) == true then
        if size ~= 1 then
            vendor = tree:add("Vendor")

            if seg_count ~= nil then
                local vendor_size = size - seg_count * 4
                vendor:append_text(" (Total Data Size: " .. vendor_size .. ")")

                vendor_msg = ", Vendor Data Size: " .. vendor_size
            end

            local reserved = data(0, 1)
            local oui = data(1, 3)
            local vendor_data = data(4, data:len() - 4)

            vendor:add(rmpp_fields.reserved, reserved)
            vendor:add(rmpp_fields.oui, oui)
            vendor:add(rmpp_fields.vendor_data, vendor_data)
        end
    end

    local msg = ""
    msg = msg .. "Type: " .. rmpp_type_str[typ:uint()]
    msg = msg .. ", Flags: "

    if is_active == true then
        msg = msg .. "A"
    end
    if is_first == true then
        msg = msg .. "F"
    end
    if is_last == true then
        msg = msg .. "L"
    end

    msg = msg .. ", Status: " .. rmpp_status_str[status:uint()]
    msg = msg .. ", Seg: " .. seg_num:uint()
    msg = msg .. ", Len: " .. paylen_newwin:uint()
    msg = msg .. vendor_msg

    tree:append_text(", " .. msg)
    pinfo.cols.info = msg
    pinfo.cols.protocol = rmpp_proto.name
end

-- Missing: don't use post dissector:
-- - packet-infiniband.c: add mgmt class dissector table
-- - packet-infiniband.c: add mgmt heuristic table
register_postdissector(rmpp_proto)
