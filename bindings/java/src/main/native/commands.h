/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef COMMANDS_H_
#define COMMANDS_H_

#define PACKED __attribute__((packed))

#include <cstdint>
#include <cstddef>

class commands {
public:
	enum Type {
		JUCX_INVALID = 0, // With respect to Java's enum Worker::CommandType
		JUCX_STREAM_SEND,
		JUCX_STREAM_RECV,
	};

	/*
	 * Compatible with ucs_status_t
	 */
	enum CompStatus {
        JUCX_OK = 0, // With respect to Java's enum Worker::CompletionStatus
        JUCX_ERR,
        JUCX_ERR_CANCELED,
    };

	struct stream_command_t {
		Type        cmd_type;
		uint64_t 	request_id;
		size_t 		length;
		CompStatus  comp_status;

		stream_command_t(uint64_t id = -1,
		                 size_t length = 0,
		                 Type type = JUCX_INVALID,
		                 CompStatus status = JUCX_OK) : request_id(id),
		                                                length(length),
		                                                cmd_type(type),
		                                                comp_status(status) {}

		void init();

		void set(uint64_t id, Type type);
	} PACKED;

	static const size_t EVENT_SIZE = sizeof(stream_command_t);

private:
	commands() {}
};

using CommandType       = commands::Type;       // enum
using CompletionStatus  = commands::CompStatus; // enum
using command           = commands::stream_command_t;

#endif /* COMMANDS_H_ */
