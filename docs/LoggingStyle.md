# UCX Logging Style

## Log Levels

| Level   | Description                                                     |
|---------|-----------------------------------------------------------------|
| `fatal` | Unrecoverable error and the program is aborted immediately      |
| `error` | Unexpected error and the program could not continue as usual    |
| `warn`  | Unexpected situation but the program can continue running       |
| `diag`  | Silent adjustment or handled error a user would want to know    |
| `info`  | One-time information about the configuration which was selected |
| `debug` | Small volume of logging, proportional to the number of objects  |
| `trace` | Larger volume of logging, in special flows during runtime       |
| `req`   | UCP requests                                                    |
| `data`  | Dumps every packet sent/received                                |
| `async` | Async context events, such as timers and signal handlers        |
| `func`  | Function calls, printed as the function name and arguments      |
| `poll`  | Every polling iteration, including the ones which found nothing |

## Choosing a Level

* Use `fatal`, `error`, `warn`, `diag`, and `info` for messages addressed to
  the user, and `debug` and `trace` for messages addressed to a UCX developer
* Use `ucs_trace_req()` for per-request events and `ucs_trace_data()` for
  per-packet events, and not `trace` for either
* Use `debug` and not `trace` for one-time flows such as initialization and
  device discovery, because `--disable-logging`, used by
  `contrib/configure-release`, compiles out levels above `debug`
* Keep the same level for the same event in different code paths

## General

* Use UCX logging macros for runtime diagnostics, not `printf`
* Use lowercase letters
* Avoid using `=`: prefer `"device %s"` instead of `"device=%s"`  
  This allows selecting the value using double-click from the terminal and searching for it in text editors
* Print flags using characters, for example:

  ```C
  "%c%c", (flag1 ? '1' : '-'), (flag2 ? '2' : '-')
  ```

## Message Content

* Prefer one line per event, holding the information relevant to it, and avoid
  `ucs_log_indent()` unless it tracks a long and complex flow such as protocol
  selection or wireup
* Collect variable-length values in a `ucs_string_buffer_t`, and dump a
  container with one helper printing one line per element
* Keep multi-line tables for output which the user asked for, such as `ucx_info`
  or `UCX_PROTO_INFO`, and print them with `ucs_string_buffer_dump()` or
  `ucs_log_print_compact()`
* Identify the object with the values which are relevant to it, such as device
  name, `sys_dev`, bus id, lane index, md map, etc., using the existing
  `*_FMT`/`*_ARG` pairs
* Print the decision which follows an unexpected value: prefer
  `"unsupported memory type %s, falling back to host"` instead of
  `"unsupported memory type %s"`

## Errors

* Print `%m` (system error code) for every system call error message
* Print error message in the first place the error is detected
* Print the exact cause of the error and not the assumed reason, because the
  assumption may not be true on all systems / in the future

## InfiniBand

* Print LID as integer: `"lid %d"`
* Print QP numbers as hex: `"qp 0x%x"`
