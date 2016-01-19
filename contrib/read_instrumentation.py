#!/usr/bin/python

import sys
import struct

HEADER_FORMAT = "1024sIL1024si40sLLQQ"
HEADER_FIELDS = [
	"library path", 
	"checksum", 
	"library loading base", 
	"command line", 
	"process id", 
	"host name", 
	"record count", 
	"record offset", 
	"start time", 
	"one second in units"
] 

RECORD_FORMAT = "QQLL"
RECORD_FIELDS = [
	"timestamp",
	"lparam",
	"wparam",
	"location"
]

def read_instrumentation_file(path):
	print "\n\n%s :\n" % path
	with open(path, "rb") as f:
		raw_header = f.read(struct.calcsize(HEADER_FORMAT))
		header = dict(zip(HEADER_FIELDS, struct.unpack(HEADER_FORMAT, raw_header)))
		for k,v in header.iteritems():
			print "%-20s : %s" % (k, v)
	
		for record in range(header["record count"]):
			raw_record = f.read(struct.calcsize(RECORD_FORMAT))
			record = dict(zip(RECORD_FIELDS, struct.unpack(RECORD_FORMAT, raw_record)))
			print record

if __name__ == "__main__":
	for filename in sys.argv[1:]:
		try:
			read_instrumentation_file(filename)
		except:
			pass
