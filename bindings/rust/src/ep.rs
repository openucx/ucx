use crate::ffi::*;
use crate::status_ptr_to_result;
use crate::status_to_result;
use crate::worker::RemoteWorkerAddress;
use crate::worker::Worker;
use crate::worker::WorkerAddress;
use bitflags::bitflags;
use std::ffi::CString;
use std::ptr::NonNull;

#[derive(Debug, Clone)]
pub struct Ep {
    pub(crate) handle: ucp_ep_h,
}

impl Ep {
    pub fn new(ep_params: &Params, worker: &Worker) -> Result<Ep, ucs_status_t> {
        let mut ep: ucp_ep_h = std::ptr::null_mut();
        let result =
            status_to_result(unsafe { ucp_ep_create(worker.handle, &ep_params.handle, &mut ep) });
        match result {
            Ok(()) => Ok(Ep { handle: ep }),
            Err(ucs_status_t) => Err(ucs_status_t),
        }
    }
}

impl Drop for Ep {
    fn drop(&mut self) {
        let param: ucp_request_param_t = unsafe { std::mem::zeroed() };
        let result =
            status_ptr_to_result(unsafe { ucp_ep_close_nbx(self.handle, &param) }).unwrap();
        if result.is_some() {
            unsafe { ucp_request_free(result.unwrap().handle.as_mut()) };
        }
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct UcpEpFields: u64 {
        const None = ucp_err_handling_mode_t::UCP_ERR_HANDLING_MODE_NONE as u64;
        const Peer = ucp_err_handling_mode_t::UCP_ERR_HANDLING_MODE_PEER as u64;
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct ParamsFlags: u64 {
        const ClientServer = ucp_ep_params_flags_field::UCP_EP_PARAMS_FLAGS_CLIENT_SERVER as u64;
        const NoLoopback = ucp_ep_params_flags_field::UCP_EP_PARAMS_FLAGS_NO_LOOPBACK as u64;
        const SendClientId = ucp_ep_params_flags_field::UCP_EP_PARAMS_FLAGS_SEND_CLIENT_ID as u64;
    }
}

#[derive(Debug, Clone)]
pub struct Params {
    pub(crate) handle: ucp_ep_params_t,
    name: Option<CString>,
}

#[derive(Debug, Clone)]
pub struct ParamsBuilder {
    uninit_handle: std::mem::MaybeUninit<ucp_ep_params_t>,
    field_mask: u64,
    name: Option<CString>,
}

impl ParamsBuilder {
    pub fn new() -> ParamsBuilder {
        let uninit_params = std::mem::MaybeUninit::<ucp_ep_params_t>::uninit();
        ParamsBuilder {
            uninit_handle: uninit_params,
            field_mask: 0,
            name: None,
        }
    }

    pub fn local_address(&mut self, worker_address: &WorkerAddress) -> &mut ParamsBuilder {
        self.field_mask |= ucp_ep_params_field::UCP_EP_PARAM_FIELD_REMOTE_ADDRESS as u64;
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.address = worker_address.handle;
        self
    }

    pub fn address(&mut self, worker_address: &RemoteWorkerAddress) -> &mut ParamsBuilder {
        self.field_mask |= ucp_ep_params_field::UCP_EP_PARAM_FIELD_REMOTE_ADDRESS as u64;
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        let (address, _) = worker_address.get_handle();
        params.address = address;
        self
    }

    pub fn name(&mut self, name: &str) -> &mut ParamsBuilder {
        self.field_mask |= ucp_ep_params_field::UCP_EP_PARAM_FIELD_NAME as u64;
        let name_cs = CString::new(name).unwrap();
        self.name = Some(name_cs);
        self
    }

    pub fn build(&mut self) -> Params {
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.field_mask = self.field_mask;
        let mut ep_param = Params {
            handle: unsafe { self.uninit_handle.assume_init() },
            name: None,
        };
        if self.name.is_some() {
            let new_name = self.name.clone().unwrap();
            ep_param.handle.name = new_name.as_ptr();
            ep_param.name = Some(new_name);
        }
        ep_param
    }
}
