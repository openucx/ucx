
# Running iodemo


## Generate launch script per host

The run_io_demo.sh will generate launch script with name ```iodemo_commands_$(hostname).sh```

``` bash
%./run_io_demo.sh --dry-run -i eth0 -H $(hostname),$(hostname) --num-clients 5 \
	--num-servers 5 --tasks-per-node 10 --duration $((10*60)) $PWD/tester
Launch configuration:
               host_list : 'mtr-vdi-326,mtr-vdi-326'
          tasks_per_node : '10'
                  map_by : 'node'
             num_clients : '5'
             num_servers : '5'
              iodemo_exe : '/labhome/miked/workspace/git/forks/ucx/test/apps/iodemo/tester'
      iodemo_client_args : ''
                  net_if : 'eth0'
           base_port_num : '20000'
                duration : '600'
        client_wait_time : '2'
                launcher : 'pdsh -b -w'
                 dry_run : '1'
iodemo_mtr-vdi-326_server_00.log
iodemo_mtr-vdi-326_server_01.log
iodemo_mtr-vdi-326_client_00.log
iodemo_mtr-vdi-326_client_01.log
iodemo_mtr-vdi-326_server_00.log
iodemo_mtr-vdi-326_server_01.log
iodemo_mtr-vdi-326_client_00.log
iodemo_mtr-vdi-326_client_01.log

% ls -1 iodemo_commands_mtr-vdi-326.sh
iodemo_commands_mtr-vdi-326.sh

```

## Check what tags are provided by script for start/stop/status operations

``` bash
%./iodemo_commands_mtr-vdi-326.sh --show-tags
Showing tags
==== Servers:
server_0
server_1
server_2
server_3
server_4
==== Clients:
client_0
client_1
client_2
client_3
client_4
```

## Running individual processes by tag (start/stop/status)



``` bash
%./iodemo_commands_mtr-vdi-326.sh -tag server_0 -status
%./iodemo_commands_mtr-vdi-326.sh -tag server_0 -stop
%./iodemo_commands_mtr-vdi-326.sh -tag server_0 -start

```

## Running all processes

``` bash
%./iodemo_commands_mtr-vdi-326.sh
```

