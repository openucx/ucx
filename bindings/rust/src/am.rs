use crate::ep::Ep;
use crate::ffi::*;
use crate::status_ptr_to_result;
use crate::status_to_result;
use crate::worker::Worker;
use crate::Request;
use crate::RequestParam;
use bitflags::bitflags;

type AmRecvCb = unsafe extern "C" fn(
    arg: *mut ::std::os::raw::c_void,
    header: *const ::std::os::raw::c_void,
    header_length: usize,
    data: *mut ::std::os::raw::c_void,
    length: usize,
    param: *const ucp_am_recv_param_t,
) -> ucs_status_t;

impl Worker<'_> {
    #[inline]
    pub fn am_register(&self, am_param: &HandlerParams) -> Result<(), ucs_status_t> {
        status_to_result(unsafe { ucp_worker_set_am_recv_handler(self.handle, &am_param.handle) })
    }
}

impl Ep<'_> {
    #[inline]
    pub fn am_send(
        &self,
        id: u32,
        header: &[u8],
        data: &[u8],
        params: &RequestParam,
    ) -> Result<Option<Request>, ucs_status_t> {
        status_ptr_to_result(unsafe {
            ucp_am_send_nbx(
                self.handle,
                id,
                header.as_ptr() as _,
                header.len(),
                data.as_ptr() as _,
                data.len(),
                &params.handle,
            )
        })
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct CbFlags: u32 {
        const WholeMsg = ucp_am_cb_flags::UCP_AM_FLAG_WHOLE_MSG as u32;
    const PersistentData = ucp_am_cb_flags::UCP_AM_FLAG_PERSISTENT_DATA as u32;
    }
}

#[derive(Debug, Clone)]
pub struct HandlerParamsBuilder {
    uninit_handle: std::mem::MaybeUninit<ucp_am_handler_param_t>,
    flags: u64,
}

impl HandlerParamsBuilder {
    #[inline]
    pub fn new() -> HandlerParamsBuilder {
        let uninit_params = std::mem::MaybeUninit::<ucp_am_handler_param_t>::uninit();
        HandlerParamsBuilder {
            uninit_handle: uninit_params,
            flags: 0,
        }
    }

    #[inline]
    pub fn id(&mut self, id: u32) -> &mut HandlerParamsBuilder {
        self.flags |= ucp_am_handler_param_field::UCP_AM_HANDLER_PARAM_FIELD_ID as u64;
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.id = id;
        self
    }

    #[inline]
    pub fn flags(&mut self, flags: CbFlags) -> &mut HandlerParamsBuilder {
        self.flags |= ucp_am_handler_param_field::UCP_AM_HANDLER_PARAM_FIELD_FLAGS as u64;
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.flags = flags.bits();
        self
    }

    #[inline]
    pub fn cb(&mut self, cb: AmRecvCb) -> &mut HandlerParamsBuilder {
        self.flags |= ucp_am_handler_param_field::UCP_AM_HANDLER_PARAM_FIELD_CB as u64;
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.cb = Some(cb);
        self
    }

    #[inline]
    pub fn arg(&mut self, arg: *mut std::os::raw::c_void) -> &mut HandlerParamsBuilder {
        self.flags |= ucp_am_handler_param_field::UCP_AM_HANDLER_PARAM_FIELD_ARG as u64;
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.arg = arg;
        self
    }

    #[inline]
    pub fn build(&mut self) -> HandlerParams {
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.field_mask = self.flags;

        let handler_param = HandlerParams {
            handle: unsafe { self.uninit_handle.assume_init() },
        };

        handler_param
    }
}

pub struct HandlerParams {
    pub(crate) handle: ucp_am_handler_param_t,
}
