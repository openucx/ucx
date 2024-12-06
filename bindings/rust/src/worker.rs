use crate::context::Context;
use crate::ep;
use crate::ep::Ep;
use crate::ffi::*;
use crate::status_to_result;
use bitflags::bitflags;
use std::ffi::CString;
use std::ptr::NonNull;

pub struct Worker<'a> {
    pub(crate) handle: ucp_worker_h,
    #[allow(dead_code)]
    parent: &'a Context,
}

impl Drop for Worker<'_> {
    fn drop(&mut self) {
        unsafe { ucp_worker_destroy(self.handle) };
    }
}

impl Worker<'_> {
    pub(crate) fn new<'a>(
        context: &'a Context,
        params: &'a Params,
    ) -> Result<Worker<'a>, ucs_status_t> {
        let mut worker: ucp_worker_h = std::ptr::null_mut();

        let result = status_to_result(unsafe {
            ucp_worker_create(context.handle, &params.handle, &mut worker)
        });
        match result {
            Ok(()) => Ok(Worker {
                handle: worker,
                parent: context,
            }),
            Err(ucs_status_t) => Err(ucs_status_t),
        }
    }

    pub fn pack_address(&self) -> Result<WorkerAddress, ucs_status_t> {
        let mut address: *mut ucp_address_t = std::ptr::null_mut();
        let mut size: usize = 0;

        let result = status_to_result(unsafe {
            ucp_worker_get_address(self.handle, &mut address, &mut size)
        });
        match result {
            Ok(()) => Ok(WorkerAddress {
                handle: address,
                parent: self,
                size: size,
            }),
            Err(ucs_status_t) => Err(ucs_status_t),
        }
    }

    #[inline]
    pub fn progress(&self) -> bool {
        let progress = unsafe { ucp_worker_progress(self.handle) };
        progress > 0
    }

    pub fn create_ep(&self, ep_params: &ep::Params) -> Result<Ep, ucs_status_t> {
        return Ep::new(&ep_params, &self);
    }
}

pub struct RemoteWorkerAddress {
    address: Vec<u8>,
}

impl RemoteWorkerAddress {
    pub fn new(address: Vec<u8>) -> RemoteWorkerAddress {
        RemoteWorkerAddress { address: address }
    }

    pub fn get_handle(&self) -> (*const ucp_address_t, usize) {
        (
            self.address.as_ptr() as *const ucp_address_t,
            self.address.len(),
        )
    }
}

pub struct WorkerAddress<'a> {
    pub(crate) handle: *const ucp_address_t,
    size: usize,
    parent: &'a Worker<'a>,
}

impl WorkerAddress<'_> {
    pub fn to_bytes(&self) -> Vec<u8> {
        unsafe { std::slice::from_raw_parts(self.handle as *const u8, self.size) }.to_vec()
    }
}

impl Drop for WorkerAddress<'_> {
    fn drop(&mut self) {
        unsafe {
            ucp_worker_release_address(self.parent.handle, self.handle as *mut ucp_address_t)
        };
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct UcpWorkerFlags: u64 {
        const Flags = ucp_worker_flags_t::UCP_WORKER_FLAG_IGNORE_REQUEST_LEAK as u64;
    }
}

impl ParamsBuilder {
    pub fn new() -> ParamsBuilder {
        let uninit_params = std::mem::MaybeUninit::<ucp_worker_params_t>::uninit();
        ParamsBuilder {
            uninit_handle: uninit_params,
            field_mask: 0,
            name: None,
        }
    }

    pub fn thread_mode(&mut self, thread_mode: ucs_thread_mode_t) -> &mut ParamsBuilder {
        self.field_mask |= ucp_worker_params_field::UCP_WORKER_PARAM_FIELD_THREAD_MODE as u64;
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.thread_mode = thread_mode;
        self
    }

    pub fn cpu_set(&mut self, cpu_set: ucs_cpu_set_t) -> &mut ParamsBuilder {
        self.field_mask |= ucp_worker_params_field::UCP_WORKER_PARAM_FIELD_CPU_MASK as u64;
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.cpu_mask = cpu_set;
        self
    }

    pub fn events(&mut self, events: u32) -> &mut ParamsBuilder {
        self.field_mask |= ucp_worker_params_field::UCP_WORKER_PARAM_FIELD_EVENTS as u64;
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.events = events;
        self
    }

    pub fn user_data(&mut self, data: *mut std::ffi::c_void) -> &mut ParamsBuilder {
        self.field_mask |= ucp_worker_params_field::UCP_WORKER_PARAM_FIELD_USER_DATA as u64;
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.user_data = data;
        self
    }

    pub fn event_fd(&mut self, event_fd: i32) -> &mut ParamsBuilder {
        self.field_mask |= ucp_worker_params_field::UCP_WORKER_PARAM_FIELD_EVENT_FD as u64;
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.event_fd = event_fd;
        self
    }

    pub fn flags(&mut self, flags: UcpWorkerFlags) -> &mut ParamsBuilder {
        self.field_mask |= ucp_worker_params_field::UCP_WORKER_PARAM_FIELD_FLAGS as u64;
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.flags = flags.bits();
        self
    }

    pub fn name(&mut self, name: &str) -> &mut ParamsBuilder {
        self.field_mask |= ucp_worker_params_field::UCP_WORKER_PARAM_FIELD_NAME as u64;
        let name_cs = CString::new(name).unwrap();
        self.name = Some(name_cs);
        self
    }

    pub fn am_alignment(&mut self, am_alignment: usize) -> &mut ParamsBuilder {
        self.field_mask |= ucp_worker_params_field::UCP_WORKER_PARAM_FIELD_AM_ALIGNMENT as u64;
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.am_alignment = am_alignment;
        self
    }

    pub fn client_id(&mut self, client_id: u64) -> &mut ParamsBuilder {
        self.field_mask |= ucp_worker_params_field::UCP_WORKER_PARAM_FIELD_CLIENT_ID as u64;
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.client_id = client_id;
        self
    }

    pub fn build(&mut self) -> Params {
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.field_mask = self.field_mask;

        let mut ucp_param = Params {
            name: None,
            handle: unsafe { self.uninit_handle.assume_init() },
        };

        if self.name.is_some() {
            let new_name = self.name.clone().unwrap();
            ucp_param.handle.name = new_name.as_ptr();
            ucp_param.name = Some(new_name);
        }

        ucp_param
    }
}

#[derive(Debug, Clone)]
pub struct ParamsBuilder {
    uninit_handle: std::mem::MaybeUninit<ucp_worker_params_t>,
    field_mask: u64,
    name: Option<CString>,
}

#[derive(Debug, Clone)]
pub struct Params {
    pub(crate) handle: ucp_worker_params_t,
    name: Option<CString>,
}
