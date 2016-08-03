#!/usr/bin/python

"""
This python script parses the binary result files of the instrumentation
capability. Each file is composed of a header and a variable number of
records containing a timestamp and some custom recorded content.
"""

import sys
import struct

HEADER_FORMAT = "1024sIL1024si40sLLLQQQ"
HEADER_FIELDS = [
    "library path",
    "checksum",
    "library loading base",
    "command line",
    "process id",
    "host name",
    "record count",
    "location count",
    "record offset",
    "start time",
    "one second in units",
    "one record in units"
] 

LOCATION_FORMAT = "II256s"
LOCATION_FIELDS = [
    "location",
    "extra",
    "name"
]

RECORD_FORMAT = "LLII"
RECORD_FIELDS = [
    "timestamp",
    "lparam",
    "wparam",
    "location"
]

SECOND_IN_UNITS = 1 # Default value
RECORD_IN_UNITS = 1 # Default value

def read_instrumentation_file(file_path):
    print "\n\n%s :\n" % file_path
    with open(file_path, "rb") as f:
        # Read instrumentation header
        raw_header = f.read(struct.calcsize(HEADER_FORMAT))
        header = dict(zip(HEADER_FIELDS,
            struct.unpack(HEADER_FORMAT, raw_header)))
        for k,v in header.iteritems():
            print "%-20s : %s" % (k, str(v).replace("\x00",""))

        global SECOND_IN_UNITS, RECORD_IN_UNITS
        SECOND_IN_UNITS = header["one second in units"]
        RECORD_IN_UNITS = header["one record in units"]

        locations = {}
        for i in range(header["location count"]):
            raw_location = f.read(struct.calcsize(LOCATION_FORMAT))
            location = dict(zip(LOCATION_FIELDS,
                struct.unpack(LOCATION_FORMAT, raw_location)))
            locations[location["location"]] = location["name"].replace("\x00","")
        for k,v in locations.iteritems():
            print "%-20i : %s" % (k, v)

        for record in range(header["record count"]):
            # Read a single instrumentation record
            raw_record = f.read(struct.calcsize(RECORD_FORMAT))
            record = dict(zip(RECORD_FIELDS,
                struct.unpack(RECORD_FORMAT, raw_record)))

            if record["location"] in locations:
                record["location"] = locations[record["location"]]
            else:
                print "ERROR: Unknown location %s !" % record["location"]
                record["location"] = "Unknown (%s)" % record["location"]

            """
            print "Timestamp: %u\tLocation: %s\tWR ID: %u\tLength: %u" % \
                (record["timestamp"], record["location"],
                 record["lparam"], record["wparam"])
            """

            yield record

def timestamp_analysis(file_path):
    sizes = {}
    req_ids = {}
    for record in read_instrumentation_file(file_path):
        req_id, req_size = record["lparam"], record["wparam"]
        if not req_id:
            # print "ERROR: record id (\"lparam\") is zero!"
            continue
        
        # Check if matches a previous request (by ID)
        if req_id in req_ids:
            prev_req = req_ids[req_id]
            req_size = prev_req["wparam"]
            req_last = prev_req["lparam"]
            interval = (prev_req["location"], record["location"])
            duration = record["timestamp"] - prev_req["timestamp"]
            
            # Produce stats entry for this record
            if req_size not in sizes:
                sizes[req_size] = {}
            locations = sizes[req_size]
            if interval not in locations:
                locations[interval] = \
                    { "count" : 0, "sum" : 0, "max" : 0, "size" : req_size}
            stats = locations[interval]
            
            # Update stats entry
            stats["count"] += 1;
            stats["sum"] += duration;
            if "min" not in stats:
                locations[interval]["min"] = duration
            else:
                stats["min"] = min(stats["min"], duration)
            stats["max"] = max(stats["max"], duration)
        
        # set this 
        req_ids[req_id] = record

    # Re-arrange by amount of messages (importance?)
    by_count = {}
    for size in sizes:
        for interval, stats in sizes[size].iteritems():
            count = stats["count"]
            if count:
                if count not in by_count:
                    by_count[count] = {}
                by_count[count][interval] = stats
    
    # Output stats
    order = by_count.keys()
    order.sort()
    for size in order:
        for interval, stats in by_count[size].iteritems():
            stats["sum"] -= stats["count"] * RECORD_IN_UNITS
            stats["average"] = ((0.0 + stats["sum"]) / stats["count"])
            avg = (10 ** 9) * stats["average"] / SECOND_IN_UNITS
            print "from %s\nto   %s\naverage %f ns\n%s\n" % \
                (interval[0], interval[1], avg, str(stats))
    print "P.S. averages exclude time it takes to record measurements."
    print "SECOND_IN_UNITS is ", SECOND_IN_UNITS
    print "RECORD_IN_UNITS is ", RECORD_IN_UNITS

if __name__ == "__main__":
    for file_path in sys.argv[1:]:
        timestamp_analysis(file_path)
