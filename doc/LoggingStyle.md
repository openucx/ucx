

* Meaning of log levels:
 - error - unexpected error and the program could not continue as usual
 - warn  - unexpected situation but the program can continue running
 - debug - small volume of logging, proportional to the number of objects created.
 - trace - larger volume of logging, in special flows during runtime
 - req   - UCP requests
 - data  - dumps every packet sent / received

* General:
 - use small letters
 - avoid using '=': "device %s" instead of "device=%s" - to allow selecting the
   value using double-click from the terminal, and searching for it in text editors.
 - print flags using characters, for exaple:
    "%c%c", (flag1 ? '1' : '-'), (flag2 ? '2' : '-') 
    
* Errors:
 - print %m (system error code) for every system call error message
 - print error message in the first place the error is detected. 
 - print the exact cause of the error and not the assumed reason, because the
   assumption may not be true on all systems / in the future.

* IB:
 - print LID as integer ("lid %d")
 - print QP numbers as hex number ("qp 0x%x")


