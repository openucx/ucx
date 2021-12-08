# The UCX code style

## Style
  * 4 spaces, no tabs
  * Up to 80 columns
  * Single space around operators
  * No spaces in the end-of-line
  * Indent function arguments on column
  * Indent structure fields on column
  * Scope: open on same line, except function body, which is on a new line.
  * Indent multiple consecutive assignments on the column
  * 2 space lines between types and prototypes (header files)
  * 1 space line between functions (source files) 


## Naming convention:
  * Lower case, underscores
  * Names must begin with ucp_/uct_/ucs_/ucm_
  * Macro names must begin with UCP_/UCT_/UCS_/UCM_
  * An output argument which is a pointer to a user variable has _p suffix
  * Value types (e.g struct types, integer types) have _t suffix
  * Pointer to structs, which are used as API handles, have _h suffix
  * Macro arguments begin with _ (e.g _value) to avoid confusion with variables
  * No leading underscores in function names
  * ### Header file name suffixes:
     * _fwd.h   for a files with a types/function forward declarations
     * _types.h if contains a type declarations
     * .inl     for inline functions
     * _def.h   with a preprocessor macros


## C++
  * Used only for unit testing
  * Lower-case class names (same as stl/boost)
  * The unit tests in test/gtest are written using [C++11](https://en.cppreference.com/w/cpp/11). Whenever applicable, the usage of advanced language features is allowed and preferred over legacy code. For example:
    * Prefer references over pointers
    * `auto` for type deduction 
    * Use [move semantics](https://www.cprogramming.com/c++11/rvalue-references-and-move-semantics-in-c++11.html) where applicable
    * `constexpr` for compile-time values, instead of `const` or `#define`
    * `using` [instead of](https://en.cppreference.com/w/cpp/language/type_alias) `typedef`
    * [List initialization](https://en.cppreference.com/w/cpp/language/list_initialization)
    * [Range-based](https://en.cppreference.com/w/cpp/language/range-for) `for` loop
 

## Include order:
   1. config.h
   2. specific internal header
   3. UCX headers
   4. system headers


## Doxygen
  * All interface H/C files should have doxygen documentation.
 

## Error handling
  * All internal error codes must be ucs_status_t
  * A function which returns error should print a log message
  * The function which prints the log message is the first one which decides which
    error it is. If a functions returns an error because it's callee returned 
    erroneous ucs_status_t, it does not have to print a log message.
  * Destructors are not able to propagate error code to the caller because they
    return void. also, users are not ready to handle errors during cleanup flow.
    therefore a destructor should handle an error by printing a warning or an
    error message.


## Testing
  * Every major feature or bugfix must be accompanied with a unit test. In case
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
            ucs_typeof((_obj)->_field1) sum = (_val1) + (_val2); \
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
