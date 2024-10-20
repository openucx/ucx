#![allow(unused_imports)]

mod ffi;
use crate::ffi::*;

pub mod am;
pub mod context;
pub mod ep;
pub mod worker;

use std::ffi::CString;
use std::ptr::NonNull;

// UCX request backed by a ucs_status_ptr_t that is non-null and not an error, thus is a request pointer
pub struct Request {
    pub(crate) handle: NonNull<::std::os::raw::c_void>,
}

impl Drop for Request {
    fn drop(&mut self) {
        unsafe { ucp_request_free(self.handle.as_ptr()) };
    }
}

impl Request {
    // new assumes that the type has already been error checked.
    #[inline]
    pub fn new(request_handle: *mut std::os::raw::c_void) -> Option<Request> {
        let request = NonNull::<::std::os::raw::c_void>::new(request_handle);
        match request {
            None => None,
            Some(x) => Some(Request { handle: x }),
        }
    }

    // check an outstanding request. Returns an error if the request had an error, returns false if the request is not completed, returns true if the request is completed
    #[inline]
    pub fn check_status(&self) -> Result<bool, ucs_status_t> {
        let status = unsafe { ucp_request_check_status(self.handle.as_ptr()) };
        if status as usize >= ucs_status_t::UCS_ERR_LAST as usize {
            return Err(unsafe { std::mem::transmute(status as i8) });
        }
        Ok(status == ucs_status_t::UCS_OK)
    }
}

// In UCX we usually use a ucs_status_ptr_t to represent the status of a nonblocking operation
// in this the possible outcomes can be UCS_OK, where the application can reuse all the input
// parameters immediately, a pointer that can be queried for the status of the underlying
// nonblocking operation, or an error. Rust APIs operate similarly, except it uses the Rust
// type system to express this. It will have a Result type that either contains an Ok() type
// or an Err() type. It also has an Option() type that basically is the equivalent of a nullable
// pointer, except Rust will force the user to be sure to check the Option().

// This helper function will automatically translate the ucs_status_ptr_t into a Result that
// either is an empty Ok() as the equivilent to UCS_OK, a Ok(Request) that represents getting
// back a pointer or an Err(ucs_status_t) that indicates an error. Compile test shows that this
// produces extremely efficient assembly

#[inline]
pub fn status_ptr_to_result(ptr: ucs_status_ptr_t) -> Result<Option<Request>, ucs_status_t> {
    // This is equivlent to the UCS_PTR_IS_ERR() macro.
    if ptr as usize >= ucs_status_t::UCS_ERR_LAST as usize {
        // The transmute() function is how you access C style memory magic. This function will
        // take the intput pointer and then translate it into i8 and then rust will turn the i8
        // into the proper ucs_status_t.
        return Err(unsafe { std::mem::transmute(ptr as i8) });
    }
    Ok(Request::new(ptr))
}

#[inline]
pub fn status_to_result(status: ucs_status_t) -> Result<(), ucs_status_t> {
    if (status as i8) < 0 {
        return Err(status);
    }
    Ok(())
}

pub struct RequestParam {
    pub(crate) handle: ucp_request_param_t,
}

#[derive(Debug, Copy, Clone)]
pub struct RequestParamBuilder {
    uninit_handle: std::mem::MaybeUninit<ucp_request_param_t>,
    field_mask: u32,
}

impl RequestParamBuilder {
    pub fn new() -> RequestParamBuilder {
        let uninit_params = std::mem::MaybeUninit::<ucp_request_param_t>::uninit();
        RequestParamBuilder {
            uninit_handle: uninit_params,
            field_mask: 0,
        }
    }

    pub fn force_imm_cmpl(&mut self) -> &mut RequestParamBuilder {
        if self.field_mask & ucp_op_attr_t::UCP_OP_ATTR_FLAG_NO_IMM_CMPL as u32 != 0 {
            panic!("Requesting UCP_OP_ATTR_FLAG_FORCE_IMM_CMPL while UCP_OP_ATTR_FLAG_NO_IMM_CMPL is also set");
        }
        self.field_mask |= ucp_op_attr_t::UCP_OP_ATTR_FLAG_FORCE_IMM_CMPL as u32;
        self
    }

    pub fn no_imm_cmpl(&mut self) -> &mut RequestParamBuilder {
        if self.field_mask & ucp_op_attr_t::UCP_OP_ATTR_FLAG_FORCE_IMM_CMPL as u32 != 0 {
            panic!("Requesting UCP_OP_ATTR_FLAG_NO_IMM_CMPL while UCP_OP_ATTR_FLAG_FORCE_IMM_CMPL is also set");
        }
        self.field_mask |= ucp_op_attr_t::UCP_OP_ATTR_FLAG_NO_IMM_CMPL as u32;
        self
    }

    pub fn build(&mut self) -> RequestParam {
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.op_attr_mask = self.field_mask;

        let ucp_param = RequestParam {
            handle: unsafe { self.uninit_handle.assume_init() },
        };

        ucp_param
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context;
    use crate::context::Context;
    use crate::ep;
    use crate::worker;

    const TEST_AM_ID: u32 = 5;

    extern "C" fn init(_request: *mut ::std::os::raw::c_void) {}

    extern "C" fn cleanup(_request: *mut ::std::os::raw::c_void) {}

    unsafe extern "C" fn am_cb(
        arg: *mut ::std::os::raw::c_void,
        header: *const ::std::os::raw::c_void,
        header_length: usize,
        _data: *mut ::std::os::raw::c_void,
        _length: usize,
        _param: *const ucp_am_recv_param_t,
    ) -> ucs_status_t {
        let message = std::slice::from_raw_parts_mut(arg as *mut i8, 1);
        let in_data = std::slice::from_raw_parts(header as *const i8, header_length);
        message[0] = in_data[0];
        ucs_status_t::UCS_OK
    }

    #[test]
    fn it_works() {
        let mut message = vec![0];
        let features = context::Flags::Am
            | context::Flags::Rma
            | context::Flags::Amo32
            | context::Flags::Amo64;
        let params = context::ParamsBuilder::new()
            .features(features)
            .mt_workers_shared(1)
            .request_init(init)
            .request_cleanup(cleanup)
            .request_size(8)
            .name("My Awesome Test")
            .tag_sender_mask(std::u64::MAX)
            .estimated_num_eps(4)
            .estimated_num_ppn(2)
            .build();
        let context = Context::new(&context::Config::default(), &params).unwrap();

        let worker_features = worker::ParamsBuilder::new()
            .thread_mode(ucs_thread_mode_t::UCS_THREAD_MODE_MULTI)
            .build();
        let worker = context.worker_create(&worker_features).unwrap();

        let am_params = am::HandlerParamsBuilder::new()
            .id(TEST_AM_ID)
            .cb(am_cb)
            .arg(message.as_mut_ptr() as *mut std::os::raw::c_void)
            .build();
        worker.am_register(&am_params).unwrap();

        let addr = worker.pack_address().unwrap();
        let ep_param = ep::ParamsBuilder::new().local_address(&addr).build();
        let ep = worker.create_ep(&ep_param).unwrap();

        let tag = vec![32];
        let am_flags = RequestParamBuilder::new()
            //.force_imm_cmpl() // uncomment this line to see the the compile time error checker in action
            .no_imm_cmpl()
            .build();

        let req = ep
            .am_send(TEST_AM_ID, tag.as_slice(), b"", &am_flags)
            .unwrap();
        if req.is_some() {
            let req = req.unwrap();
            while !req.check_status().unwrap() {
                worker.progress();
            }
        }
        assert_eq!(message[0], tag[0]);
    }
}
