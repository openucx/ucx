# The UCX code style

## Style
  * 4 spaces, no tabs
  * up to 80 columns
  * single space around operators
  * no spaces in the end-of-line
  * indent function arguments on column
  * indent structure fields on column
  * scope: open on same line, except function body, which is on a new line.
  * indent multiple consecutive assignments on the column
  * 2 space lines between types and prototypes (header files)
  * 1 space line between functions (source files) 


## Naming convention:
  * lower case, underscores
  * names must begin with ucp_/uct_/ucs_/ucm_
  * macro names must begin with UCP_/UCT_/UCS_/UCM_
  * an output argument which is a pointer to a user variable has _p suffix
  * value types (e.g struct types, integer types) have _t suffix
  * pointer to structs, which are used as API handles, have _h suffix
  * macro arguments begin with _ (e.g _value) to avoid confusion with variables
  * no leading underscores in function names
  * ### Header file name suffixes:
     * _fwd.h   for a files with a types/function forward declarations
     * _types.h if contains a type declarations
     * .inl     for inline functions
     * _def.h   with a preprocessor macros


## C++
  * used only for unit testing
  * lower-case class names (same as stl/boost)
 

## Include order:
   1. config.h
   2. specific internal header
   3. UCX headers
   4. system headers


## Doxygen
  * all interface H/C files should have doxygen documentation.
 

## Error handling
  * all internal error codes must be ucs_status_t
  * a function which returns error should print a log message
  * the function which prints the log message is the first one which decides which
    error it is. If a functions returns an error because it's callee returned 
    erroneous ucs_status_t, it does not have to print a log message.
  * destructors are not able to propagate error code to the caller because they
    return void. also, users are not ready to handle errors during cleanup flow.
    therefore a destructor should handle an error by printing a warning or an
    error message.


## Testing
  * every major feature or bugfix must be accompanied with a unit test. In case
    of a fix, the test should fail without the fix.


## Examples

### if style

Good
```C
    if (val != XXX) {
        /* snip */
    } else if (val == YYY) {
        /* code here */
    } else {
        /* code here */
    }
```

Bad
```C
    if(val != XXX) {   /* Require space after if */
    if (val != XXX){   /* Require space after )  */
    if ( val != XXX) { /* Remove space after (   */
```

### goto style

Good
```C
err_free:
    ucs_free(thread);
err:
    --ucs_async_thread_global_context.use_count;
out_unlock:
    ucs_assert_always(ucs_async_thread_global_context.thread != NULL);
    *thread_p = ucs_async_thread_global_context.thread;
```

Bad
```C
err_free:
    ucs_free(thread);
/*    !!!Remove this line!!!    */
err:
    --ucs_async_thread_global_context.use_count;
```

### structure assignment

Good

```C
    event.events   = events;
    event.data.fd  = event_fd;
    event.data.ptr = udata;

```

Bad
```C
    /* Align = position */
    event.events = events;
    event.data.fd = event_fd;
    event.data.ptr = udata;
```

### comment in C file

Good
```C
/* run-time CPU detection */
```

Bad: require C style `/* .. */` comment.

```C
// run-time CPU detection
```

### no spaces in the end-of-line

Good
```C
    int fd;
```

Bad
```
    int fd;  
        /* ^^ Remove trailing space */
```

### macro definition

Good
```C
    #define UCS_MACRO_SHORT(_obj, _field, _val) \
        (_obj)->_field = (_val)

    #define UCS_MACRO_LONG(_obj, _field1, _field2, _val1, _val2) \
        { \
            typeof((_obj)->_field1) sum = (_val1) + (_val2); \
            \
            (_obj)->_field1 = sum; \
            (_obj)->_field2 = sum; \
        }

    #define UCS_MACRO_LONG_RET_VAL(_obj, _field, _val, _func) \
        ({ \
            ucs_status_t status; \
            \
            (_obj)->_field = (_val); \
            \
            status = _func(_obj); \
            status; \
        })
```

Bad
```C
    #define UCS_MACRO_SHORT(_obj, _field, _val) \
        _obj->_field = _val /* need to wrap macro arguments by () */

    #define UCS_MACRO_LONG(_obj, _field1, _field2, _val1, _val2) \
            /* possible mixing declarations and code */ \
            typeof((_obj)->_field1) sum = (_val1) + (_val2); \
            \
            (_obj)->_field1 = sum; \
            (_obj)->_field2 = sum;

    #define UCS_MACRO_LONG_RET_VAL(_obj, _field, _val, _func) \
        ({                                                    \
            ucs_status_t status;                              \
                                                              \
            (_obj)->_field = (_val);                          \
                                                              \
            status = _func(_obj);                             \
            status;                                           \
        }) /* wrong alignment of "\" */
```
