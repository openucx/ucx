import subprocess
import os
import argparse
import re
import datetime,time
import traceback,sys

allow_error_list = [
    'Connection reset by remote peer',
    'UCX-connection.*detected error:',
    'ERROR Remote QP on mlx',
    'UCX  ERROR RC QP',
    'ERROR IB Async event on',
    'setting error flag on connection',
    'Operation rejected by remote peer',
    'got error event RDMA_CM_EVENT_ADDR_ERROR',
    'rdma_accept',
    'UCX  ERROR Remote access on',
    'UCX  ERROR Transport retry count exceeded on',
    'UCX  WARN  failed to disconnect CM lane',
    'ucp_ep_create\(\) failed: Input/output error',
    'terminate connection.*due to Input/output error',
    'UCX  ERROR Local QP operation on',
    'conn_id send request.*failed: Input/output error',
    'deleting connection with status Input/output error',
    'UCX  WARN  failed to disconnect CM lane .* Operation rejected by remote peer',
    'ucp_ep_query\(\) failed: Endpoint timeout',
    'UCX  ERROR rdma_reject.*failed with error: Invalid argument',
    'UCX  ERROR rdma_init_qp_attr.*failed: Invalid argument',
    'UCX  ERROR rdma_establish on ep.*failed: Invalid argument',
    'UCX  ERROR .*client.*failed to process a connect response'
    ]


re_allow_list = re.compile("|".join(allow_error_list), re.I)
re_timestamp = re.compile(r"\[(\d+\.\d+)\].*")
re_traffic = re.compile(r"\[(\d+\.\d+)\].*read (\d+.\d+).*min:(\d+).*write (\d+.\d+).*min:(\d+).*")
re_traffic_read = re.compile(r"\[(\d+\.\d+)\].*read (\d+.\d+) MB\/s min:(\d+).*")
re_traffic_write = re.compile(r"\[(\d+\.\d+)\].*write (\d+.\d+) MB\/s min:(\d+).*")
re_error = re.compile(r".*(error|assert|backtrace|segmentation).*", re.I)
re_warning = re.compile(r".*warn.*", re.I)


def in_allow_list(line, is_allow_list):
    if is_allow_list:
        s = re_allow_list.search(line)
        if s:
            return True
    return False


def process_seek(seek_file):
    data = {}
    if not seek_file or not os.path.exists(seek_file):
        return data

    with open(seek_file) as f:
        for line in f.readlines():
            if line:
                # [log name] \t [position] \t [previous  timestamp]
                d = line.split('\t')
                ts = datetime.datetime.fromtimestamp(float(d[2]))
                rx_ts = float(d[3])
                tx_ts = float(d[4])
                data[d[0]] = {
                    'pos': int(d[1]),
                    'timestamp': ts,
                    'timestamp_rx': rx_ts,
                    'timestamp_tx': tx_ts,
                    }
    # Burn After Reading
    open(seek_file, 'w').close()
    return data


def get_logs(directory):
    client_list = []
    server_list = []
    for f in os.listdir(directory):
        filename = os.path.join(directory, f)
        if os.path.isfile(filename) and "_client_" in f:
            client_list.append(filename)
        if os.path.isfile(filename) and "_server_" in f:
            server_list.append(filename)
    return client_list, server_list


def process_server(files, is_allow_list):
    for log in files:
        with open(log) as f:
            while True:
                line = f.readline()
                if not line:
                    break

                m = re_error.match(line, re.IGNORECASE) or re_warning.match(line, re.IGNORECASE)
                if m and not in_allow_list(line, is_allow_list):
                    raise Exception("Contains error: {}\nLog {}:\nLine {}".format(line, log, line))


def process_client(files, threshold, seek_file, is_allow_list):
    seek_data = process_seek(seek_file)
    for log in files:
        with open(log) as f:
            curr_ts = 0
            curr_traffic_ts = 0
            cur_traffic_date = ""
            prev_traffic_ts = seek_data.get(log, {}).get('timestamp', 0)
            zero_rx_ts = seek_data.get(log, {}).get('timestamp_rx', 0)
            zero_tx_ts = seek_data.get(log, {}).get('timestamp_tx', 0)
            pos_prev = seek_data.get(log, {}).get('pos', 0)
            f.seek(pos_prev)
            i = 0
            while True:
                line = f.readline()
                if not line:
                    if seek_file and cur_traffic_date:
                        pos = f.tell()
                        with open(seek_file, 'a+') as s:
                            s.write("{}\t{}\t{}\t{}\t{}\n".format(
                                log, pos, cur_traffic_date, zero_rx_ts, zero_tx_ts))
                    break

                timestamp_match = re_timestamp.match(line)
                if timestamp_match:
                    date = float(timestamp_match.group(1))
                    curr_ts = datetime.datetime.fromtimestamp(date)
                    if not prev_traffic_ts:
                        prev_traffic_ts = curr_ts

                i += 1
                read_match = re_traffic_read.match(line)
                write_match = re_traffic_write.match(line)

                current_match = None

                if read_match:
                    current_match = read_match
                    cur_traffic_date = current_match.group(1)
                    date_traffic = float(cur_traffic_date)
                    curr_traffic_ts = datetime.datetime.fromtimestamp(date_traffic)
                    rx = float(current_match.group(2))
                    min_server_rx = int(current_match.group(3))

                    if min_server_rx == 0 and zero_rx_ts:
                        delta = curr_traffic_ts - datetime.datetime.fromtimestamp(zero_rx_ts)
                        if delta.total_seconds() > threshold * 60:
                            raise Exception("Have read min:0 servers {} minutes \
                                (more threshold:{})\nlog {}:\nLine {}".format(
                                    delta.total_seconds()/60.0, threshold, log, line))
                    else:
                        zero_rx_ts = date_traffic

                    if not rx:
                        raise Exception("Have read zero speed:\nLog {}:\nLine {}".format(log, line))
                    prev_traffic_ts = curr_traffic_ts

                if write_match:
                    current_match = write_match
                    cur_traffic_date = current_match.group(1)
                    date_traffic = float(cur_traffic_date)
                    curr_traffic_ts = datetime.datetime.fromtimestamp(date_traffic)
                    tx = float(current_match.group(2))
                    min_server_tx=int(current_match.group(3))

                    if min_server_tx == 0 and zero_tx_ts:
                        delta = curr_traffic_ts - datetime.datetime.fromtimestamp(zero_tx_ts)
                        if delta.total_seconds() > threshold * 60:
                            raise Exception("Have write min:0 servers {} minutes \
                                (more threshold:{})\nLog {}:\nLine {}".format(
                                    delta.total_seconds()/60.0, threshold, log, line))
                    else:
                        zero_tx_ts = date_traffic

                    if not tx:
                        raise Exception("Have write zero speed:\nLog {}:\nLine {}".format(log, line))

                    prev_traffic_ts = curr_traffic_ts


                if current_match and prev_traffic_ts:
                    delta = curr_traffic_ts - prev_traffic_ts
                    if delta.total_seconds() > threshold * 60:
                        raise Exception("Have delta {} more {} minutes\nLog {}:\nLine {}".format(
                            delta.total_seconds()/60.0, threshold, log, line))

                if not current_match:
                    current_match = re_error.match(line, re.IGNORECASE)
                    if current_match:
                        if not in_allow_list(line, is_allow_list):
                            raise Exception("contains error: {}\nLog {}:\nLine {}".format(line, log, line))
                    else:
                        current_match = re_warning.match(line, re.IGNORECASE)
                        if current_match:
                            print("log {} [{}] contains warning: {}".format(log, i, line))

                if curr_ts and (curr_ts - prev_traffic_ts).total_seconds() > threshold * 60:
                    raise Exception("No traffic\n{}\nLog {}".format(line, log))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str,
                        help='Log filename')
    parser.add_argument('-d', '--directory', type=str,
                        help='Directory name with Logs')
    parser.add_argument('-t', '--no_traffic_in_sec', type=int, default=1,
                        help='No traffic Threshold in min ')
    parser.add_argument('-s', '--seek', type=str, default="",
                        help='path to seek file')
    parser.add_argument('-r', '--role', type=str, default="client", choices=['client', 'server'],
                        help='choice role if you set filename')
    parser.add_argument('--no-allow-list', dest='allow_list', action='store_false')

    args = parser.parse_args()

    clients = []
    servers = []
    if args.filename:
        if args.role == "client":
            clients.append(args.filename)
        elif args.role == "server":
            servers.append(args.filename)

    if args.directory:
        clients, servers = get_logs(args.directory)

    try:
        process_client(clients, args.no_traffic_in_sec, args.seek, args.allow_list)
        process_server(servers, args.allow_list)
    except Exception as e:
        print("Error iodemo analyzer: {}\n".format(e))
        traceback.print_exc(file=sys.stdout)
        exit(1)
