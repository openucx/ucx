/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package goucxtests

import (
	"math/big"
	"syscall"
	"testing"
	"time"
	. "ucx"
)

func TestUcpWorkerEfd(t *testing.T) {
	ucpContext, _ := NewUcpContext((&UcpParams{}).EnableTag().EnableWakeup())
	defer ucpContext.Close()
	ucpWorkerParams := &UcpWorkerParams{}
	ucpWorkerParams.SetThreadMode(UCS_THREAD_MODE_SINGLE)
	ucpWorkerParams.WakeupEdge()

	ucpWorker, err := ucpContext.NewWorker(ucpWorkerParams)
	defer ucpWorker.Close()

	if err != nil {
		t.Fatalf("Failed to create a worker %v", err)
	}

	eventsChan := make(chan int)
	started := make(chan bool)

	go func() {
		var event syscall.EpollEvent
		var events [4]syscall.EpollEvent

		efd, err := ucpWorker.GetEfd()
		if err != nil {
			t.Fatalf("Failed to get worker efd: %v", err)
		}

		epfdLocal, err := syscall.EpollCreate1(0)
		if err != nil {
			t.Fatalf("Failed to call EpollCreate1 %v", err)
		}
		defer syscall.Close(epfdLocal)

		event.Events = syscall.EPOLLIN
		event.Fd = int32(efd)
		if err = syscall.EpollCtl(epfdLocal, syscall.EPOLL_CTL_ADD, efd, &event); err != nil {
			t.Fatalf("Failed to call EpollCtl %v", err)
		}

		for ; ucpWorker.Arm() != UCS_OK; ucpWorker.Progress() {

		}
		started <- true

		nevents, err := syscall.EpollWait(epfdLocal, events[:], -1)
		if err != nil {
			t.Fatalf("Failed to call epoll_wait %v", err)
		}

		eventsChan <- nevents
	}()

	select {
	case <-started:
	case <-time.After(500 * time.Millisecond):
		t.Fatalf("Didn't start after 500 ms")
	}
	ucpWorker.Signal()

	nevents := <-eventsChan
	if nevents <= 0 {
		t.Fatalf("No events in epoll %v", err)
	}
}

func TestUcpWorkerWakeup(t *testing.T) {
	ucpContext, _ := NewUcpContext((&UcpParams{}).EnableTag().EnableWakeup())
	defer ucpContext.Close()
	ucpWorkerParams := &UcpWorkerParams{}
	var cpuMask big.Int
	cpuMask.SetBit(&cpuMask, 0, 1)
	ucpWorkerParams.SetCpuMask(&cpuMask).WakeupTX().WakeupRX()
	ucpWorkerParams.SetThreadMode(UCS_THREAD_MODE_MULTI)
	ucpWorker, err := ucpContext.NewWorker(ucpWorkerParams)
	defer ucpWorker.Close()

	if err != nil {
		t.Fatalf("Failed to create a worker %v", err)
	}

	quit := make(chan int)
	started := make(chan bool)

	go func() {
		started <- true
		ucpWorker.Wait()
		quit <- 0
	}()

	<-started
	ucpWorker.Signal()

	if <-quit != 0 {
		t.Fatalf("Didn't exit from wait")
	}
}
