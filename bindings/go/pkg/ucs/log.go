package ucs

/*
#cgo CFLAGS: -I/home/bryan/src/ucx/release/include -DUCS_LOG_LEVEL_TRACE_LAST=UCS_LOG_LEVEL_LAST
#cgo LDFLAGS: -L/home/bryan/src/ucx/release/lib -lucs

#include <stdlib.h>
#include <ucs/debug/log_def.h>

void go_ucs_warn(const char *message) {
	ucs_warn(message);
}

void go_ucs_error(const char *message) {
	ucs_error(message);
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func LogWarnf(format string, args ...interface{}) {
	message := fmt.Sprintf(format, args...)
	m_cstr := C.CString(message)
	defer C.free(unsafe.Pointer(m_cstr))
	C.go_ucs_warn(m_cstr)
}

func LogErrorf(format string, args ...interface{}) {
	message := fmt.Sprintf(format, args...)
	m_cstr := C.CString(message)
	defer C.free(unsafe.Pointer(m_cstr))
	C.go_ucs_error(m_cstr)
}
