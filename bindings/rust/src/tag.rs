use crate::ep::Ep;
use crate::ffi::*;
use crate::status_ptr_to_result;
use crate::status_to_result;
use crate::worker::Worker;
use crate::Request;
use crate::RequestParam;
use bitflags::bitflags;

impl Ep {
    pub fn tag_send(
        &self,
        data: &[u8],
        tag: u64,
        param: &RequestParam,
    ) -> Result<Option<Request>, ucs_status_t> {
        status_ptr_to_result(unsafe {
            ucp_tag_send_nbx(
                self.handle,
                data.as_ptr() as _,
                data.len(),
                tag,
                &param.handle,
            )
        })
    }
}

pub struct MessageHandle {
    pub(crate) handle: ucp_tag_message_h,
    pub(crate) info: ucp_tag_recv_info_t,
    removed: bool,
}

impl MessageHandle {
    pub fn len(&self) -> usize {
        self.info.length
    }

    pub fn sender_tag(&self) -> u64 {
        self.info.sender_tag
    }
}

pub struct TagInfo {
    pub(crate) handle: ucp_tag_recv_info_t,
}

impl TagInfo {
    pub fn len(&self) -> usize {
        self.handle.length
    }

    pub fn sender_tag(&self) -> u64 {
        self.handle.sender_tag
    }
}

impl Worker {
    pub fn tag_recv(
        &self,
        data: &mut [u8],
        tag: u64,
        mask: u64,
        param: &RequestParam,
    ) -> Result<Option<Request>, ucs_status_t> {
        status_ptr_to_result(unsafe {
            ucp_tag_recv_nbx(
                self.handle,
                data.as_ptr() as _,
                data.len(),
                tag,
                mask,
                &param.handle,
            )
        })
    }

    pub fn tag_probe(&self, tag: u64, tag_mask: u64, remove: bool) -> Option<MessageHandle> {
        let mut info = std::mem::MaybeUninit::<ucp_tag_recv_info_t>::uninit();
        let handle = unsafe {
            ucp_tag_probe_nb(self.handle, tag, tag_mask, remove as i32, info.as_mut_ptr())
        };

        if !handle.is_null() {
            Some(MessageHandle {
                handle: handle,
                info: unsafe { info.assume_init() },
                removed: remove,
            })
        } else {
            None
        }
    }

    pub fn tag_msg_recv(
        &self,
        data: &mut [u8],
        message: &MessageHandle,
        param: &RequestParam,
    ) -> Result<Option<Request>, ucs_status_t> {
        if !message.removed {
            panic!("Tried to call tag_msg_recv() on a MessageHandle that didn't remove the entry!");
        }
        status_ptr_to_result(unsafe {
            ucp_tag_msg_recv_nbx(
                self.handle,
                data.as_ptr() as _,
                data.len(),
                message.handle,
                &param.handle,
            )
        })
    }
}

impl Request {
    pub fn tag_recv_test(&mut self) -> Result<Option<TagInfo>, ucs_status_t> {
        let mut info = std::mem::MaybeUninit::<ucp_tag_recv_info_t>::uninit();
        let status = unsafe { ucp_tag_recv_request_test(self.handle.as_mut(), info.as_mut_ptr()) };
        match status {
            ucs_status_t::UCS_OK => Ok(None),
            ucs_status_t::UCS_INPROGRESS => Ok(Some(unsafe {
                TagInfo {
                    handle: info.assume_init(),
                }
            })),
            _ => Err(status),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context;
    use crate::context::Context;
    use crate::ep;
    use crate::tests::setup_default;
    use crate::tests::CommsContext;
    use crate::worker;
    use crate::worker::RemoteWorkerAddress;
    use crate::RequestParamBuilder;
    use std::rc::Rc;

    const TAG_FULL: u64 = u64::MAX;

    #[test]
    fn tag_send() {
        let comms = setup_default();
        let mut recv_buff = vec![0];
        let send_buff = vec![32];
        let tag_flags = RequestParamBuilder::new().no_imm_cmpl().build();
        let _send_req = comms
            .ep
            .tag_send(send_buff.as_slice(), TAG_FULL, &tag_flags)
            .unwrap();
        let recv_req = comms
            .worker
            .tag_recv(recv_buff.as_mut_slice(), TAG_FULL, TAG_FULL, &tag_flags)
            .unwrap()
            .unwrap();
        while !recv_req.check_finished().unwrap() {
            comms.worker.progress();
        }
        assert_eq!(send_buff[0], recv_buff[0]);
    }

    #[test]
    fn tag_probe() {
        let comms = setup_default();
        let mut recv_buff = vec![0];
        let send_buff = vec![32];
        let tag_flags = RequestParamBuilder::new().no_imm_cmpl().build();
        let _send_req = comms
            .ep
            .tag_send(send_buff.as_slice(), TAG_FULL, &tag_flags)
            .unwrap();
        let mut msg = comms.worker.tag_probe(TAG_FULL, TAG_FULL, true);
        while msg.is_none() {
            comms.worker.progress();
            msg = comms.worker.tag_probe(TAG_FULL, TAG_FULL, true);
        }
        let msg = msg.unwrap();
        let recv_req = comms
            .worker
            .tag_msg_recv(recv_buff.as_mut_slice(), &msg, &tag_flags)
            .unwrap()
            .unwrap();
        while !recv_req.check_finished().unwrap() {
            comms.worker.progress();
        }
        assert_eq!(send_buff[0], recv_buff[0]);
    }
}
