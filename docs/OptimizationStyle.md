
The following guide provides rules-of-thumb for what are the 
optimization levels expected in UCX in different cases. The goal is 
to achieve reasonable balance between optimization and amount of 
effort invested, and make it consistent across all UCX modules.


Memory footprint optimizations
------------------------------

The most important aspect of memory usage optimization is scalability.
It means the amount of memory being used should depend as little as 
possible on the number of connections created. 

* Avoid at all costs of enlarging the endpoint structure and remote
  memory key structure.
* Number of requests and other descriptors should be proportional to 
  the number of in-flight operations, and not the number of 
  connections.
* Number of buffers used from memory pools should be limited. 
* It's not a problem to add small fields to structures which exist 
  per thread/process/device (such as iface, worker, context, md).


Performance optimizations
-------------------------

The required level of optimization depends on the context - small 
messages should be highly optimized, while other cases should be only
reasonably optimized. The requirements below are for the binary code, 
not for the source code. For example, inline function or a conditional
which is resolved in compile time are not significant. This requires 
some level of understanding of compiler optimizations.

* Data path for small/medium messages (about 2k and lower):
  - No system calls
  - No malloc()/free() - use memory pool instead
  - Avoid locks if possible. If needed, use spinlock, no mutex.
  - Reduce function calls and conditionals ("if").
  - Move error and slow-path handling code to non-inline functions, so
    their local variables will not add overhead to the prologue and 
    epilogue of the fast-path function.

* Data path for small messages ("short"):
  - Take care of the small-message case first.
  - Avoid function calls.
  - Avoid extra pointer dereference, especially store operations.
  - Avoid adding conditionals, if absolutely required use ucs_likely/
    ucs_unlikely macros.
  - Avoid bus-locked instructions (atomics).
  - No malloc()/free() nor system calls.
  - Limit the scope of local variables (the time from first to last 
    time it is used) - larger scopes causes spilling more variables to
    the stack.
  - Use benchmarks (such as ucx_perftest) and performance analysis 
    tools (such as perf) to make sure changes to the fast patch do not 
    impact latency and message rate.

* Pending operation flows and large messages are not considered fast 
  path, but they should still have a reasonable level of optimization:
  - No system calls / malloc / free
  - It's ok to reasonable add pointer dereferences, conditionals, 
    function calls, etc. Having a readable code here is more important
    than saving one conditional or function call.
  - Protocol-level performance considerations are more important here, 
    such as fairness between connections, fast convergence, etc.
  - Need to make sure we don't have O(n) complexity. As a thumb rule, 
    all scheduling mechanisms have to be O(1).

* Object creation and destruction flows:
  - It's ok to use system calls / malloc / free.
  - Connection creation time must be O(n) (n = number of connections).
    make sure that creating/destroying an endpoint does not require 
    going over all existing endpoints.
   


