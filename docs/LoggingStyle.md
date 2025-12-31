# UCX Logging Style

## Log Levels

| Level   | Description                                                     |
|---------|-----------------------------------------------------------------|
| `error` | Unexpected error and the program could not continue as usual    |
| `warn`  | Unexpected situation but the program can continue running       |
| `debug` | Small volume of logging, proportional to the number of objects  |
| `trace` | Larger volume of logging, in special flows during runtime       |
| `req`   | UCP requests                                                    |
| `data`  | Dumps every packet sent/received                                |

## General

* Use lowercase letters
* Avoid using `=`: prefer `"device %s"` instead of `"device=%s"`  
  This allows selecting the value using double-click from the terminal and searching for it in text editors
* Print flags using characters, for example:

  ```C
  "%c%c", (flag1 ? '1' : '-'), (flag2 ? '2' : '-')
  ```

## Errors

* Print `%m` (system error code) for every system call error message
* Print error message in the first place the error is detected
* Print the exact cause of the error and not the assumed reason, because the
  assumption may not be true on all systems / in the future

## InfiniBand

* Print LID as integer: `"lid %d"`
* Print QP numbers as hex: `"qp 0x%x"`
