use crate::ffi::*;
use crate::status_to_result;
use crate::worker;
use crate::worker::Worker;
use bitflags::bitflags;
use std::ffi::CString;

type RequestInitCb = unsafe extern "C" fn(request: *mut ::std::os::raw::c_void);
type RequestCleanUpCb = unsafe extern "C" fn(request: *mut ::std::os::raw::c_void);

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Flags: u64 {
        const Tag = ucp_feature::UCP_FEATURE_TAG as u64;
        const Rma = ucp_feature::UCP_FEATURE_RMA as u64;
        const Amo32 = ucp_feature::UCP_FEATURE_AMO32 as u64;
        const Amo64 = ucp_feature::UCP_FEATURE_AMO64 as u64;
        const Wakeup = ucp_feature::UCP_FEATURE_WAKEUP as u64;
        const Stream = ucp_feature::UCP_FEATURE_STREAM as u64;
        const Am = ucp_feature::UCP_FEATURE_AM as u64;
        const ExportedMemH = ucp_feature::UCP_FEATURE_EXPORTED_MEMH as u64;
    }
}

pub struct Config {
    handle: *mut ucp_config_t,
}

impl Config {
    pub fn read(name: &str, file: &str) -> Result<*mut ucp_config_t, ucs_status_t> {
        let mut config: *mut ucp_config_t = std::ptr::null_mut();
        let c_name = CString::new(name).unwrap();
        let c_file = CString::new(file).unwrap();
        status_to_result(unsafe { ucp_config_read(c_name.as_ptr(), c_file.as_ptr(), &mut config) })
            .unwrap();
        return Ok(config);
    }
}

impl Default for Config {
    fn default() -> Self {
        let config = Config::read("", "").unwrap();
        Config { handle: config }
    }
}

impl Drop for Config {
    fn drop(&mut self) {
        unsafe { ucp_config_release(self.handle) };
    }
}

#[derive(Debug, Clone)]
pub struct ParamsBuilder {
    uninit_handle: std::mem::MaybeUninit<ucp_params_t>,
    field_mask: u64,
    name: Option<CString>,
}

#[derive(Debug, Clone)]
pub struct Params {
    handle: ucp_params_t,
    name: Option<CString>,
}

// This builder wraps up the unsafe parts of building the ucp_param_t struct. On construction
// it makes a zero filled ucp_params_t which Rust considers uninitialized. Each call on the builder
// will fill in the fields of the struct and add the mask for that field. On the final build()
// it will fill in the final value of the features field_mask and proclame the rest of the struct
// as initialized. This is Rust safe because all of the other fields are guaranteed to not be used
// by the library since the proper feature flag is not set.

impl ParamsBuilder {
    pub fn new() -> ParamsBuilder {
        let uninit_params = std::mem::MaybeUninit::<ucp_params_t>::uninit();
        ParamsBuilder {
            uninit_handle: uninit_params,
            field_mask: 0,
            name: None,
        }
    }

    pub fn features(&mut self, features: Flags) -> &mut ParamsBuilder {
        self.field_mask |= ucp_params_field::UCP_PARAM_FIELD_FEATURES as u64;
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.features = features.bits();
        self
    }

    pub fn request_size(&mut self, size: usize) -> &mut ParamsBuilder {
        self.field_mask |= ucp_params_field::UCP_PARAM_FIELD_REQUEST_SIZE as u64;
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.request_size = size;
        self
    }

    pub fn request_init(&mut self, cb: RequestInitCb) -> &mut ParamsBuilder {
        self.field_mask |= ucp_params_field::UCP_PARAM_FIELD_REQUEST_INIT as u64;
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };

        params.request_init = Some(cb);
        self
    }

    pub fn request_cleanup(&mut self, cb: RequestCleanUpCb) -> &mut ParamsBuilder {
        self.field_mask |= ucp_params_field::UCP_PARAM_FIELD_REQUEST_CLEANUP as u64;
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.request_cleanup = Some(cb);
        self
    }

    pub fn tag_sender_mask(&mut self, mask: u64) -> &mut ParamsBuilder {
        self.field_mask |= ucp_params_field::UCP_PARAM_FIELD_TAG_SENDER_MASK as u64;
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.tag_sender_mask = mask;
        self
    }

    pub fn mt_workers_shared(&mut self, shared: i32) -> &mut ParamsBuilder {
        self.field_mask |= ucp_params_field::UCP_PARAM_FIELD_MT_WORKERS_SHARED as u64;
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.mt_workers_shared = shared;
        self
    }

    pub fn estimated_num_eps(&mut self, num_eps: usize) -> &mut ParamsBuilder {
        self.field_mask |= ucp_params_field::UCP_PARAM_FIELD_ESTIMATED_NUM_EPS as u64;
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.estimated_num_eps = num_eps;
        self
    }

    pub fn estimated_num_ppn(&mut self, num_ppn: usize) -> &mut ParamsBuilder {
        self.field_mask |= ucp_params_field::UCP_PARAM_FIELD_ESTIMATED_NUM_PPN as u64;
        let params = unsafe { &mut *self.uninit_handle.as_mut_ptr() };
        params.estimated_num_ppn = num_ppn;
        self
    }

    pub fn name(&mut self, name: &str) -> &mut ParamsBuilder {
        self.field_mask |= ucp_params_field::UCP_PARAM_FIELD_NAME as u64;
        let name_cs = CString::new(name).unwrap();
        self.name = Some(name_cs);
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

impl Context {
    pub fn new(config: &Config, params: &Params) -> Result<Context, ucs_status_t> {
        let mut context: ucp_context_h = std::ptr::null_mut();

        let result = status_to_result(unsafe {
            ucp_init_version(
                UCP_API_MAJOR,
                UCP_API_MINOR,
                &params.handle,
                config.handle,
                &mut context,
            )
        });
        match result {
            Ok(()) => Ok(Context { handle: context }),
            Err(ucs_status_t) => Err(ucs_status_t),
        }
    }

    pub fn worker_create<'a>(
        &'a self,
        params: &'a worker::Params,
    ) -> Result<Worker<'a>, ucs_status_t> {
        Worker::new(self, params)
    }
}

pub struct Context {
    pub(crate) handle: ucp_context_h,
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { ucp_cleanup(self.handle) };
    }
}
