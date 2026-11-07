/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package goucxtests

import (
	"fmt"
	"math/big"
	"syscall"
	"testing"
	"time"
	. "github.com/openucx/ucx/bindings/go/src/ucx"
)

const wakeupTimeout = 10 * time.Second

func TestUcpWorkerEfd(t *testing.T) {
	ucpContext, _ := NewUcpContext((&UcpParams{}).EnableTag().EnableWakeup())
	defer ucpContext.Close()
	ucpWorkerParams := &UcpWorkerParams{}
	ucpWorkerParams.SetThreadMode(UCS_THREAD_MODE_SERIALIZED)
	ucpWorkerParams.WakeupEdge()

	ucpWorker, err := ucpContext.NewWorker(ucpWorkerParams)
	defer ucpWorker.Close()

	if err != nil {
		t.Fatalf("Failed to create a worker %v", err)
	}

	eventsChan := make(chan int)
	started := make(chan bool)
	// t.Fatalf must be called only from the test goroutine
	threadErr := make(chan error, 1)

	go func() {
		var event syscall.EpollEvent
		var events [4]syscall.EpollEvent

		efd, err := ucpWorker.GetEfd()
		if err != nil {
			threadErr <- fmt.Errorf("Failed to get worker efd: %v", err)
			return
		}

		epfdLocal, err := syscall.EpollCreate1(0)
		if err != nil {
			threadErr <- fmt.Errorf("Failed to call EpollCreate1 %v", err)
			return
		}
		defer syscall.Close(epfdLocal)

		event.Events = syscall.EPOLLIN
		event.Fd = int32(efd)
		if err = syscall.EpollCtl(epfdLocal, syscall.EPOLL_CTL_ADD, efd, &event); err != nil {
			threadErr <- fmt.Errorf("Failed to call EpollCtl %v", err)
			return
		}

		for ; ucpWorker.Arm() != UCS_OK; ucpWorker.Progress() {

		}
		started <- true

		var nevents int
		// epoll_wait is never restarted after a signal handler, e.g. the Go
		// runtime preemption signal, so retry it on EINTR
		for {
			nevents, err = syscall.EpollWait(epfdLocal, events[:], -1)
			if err != syscall.EINTR {
				break
			}
		}
		if err != nil {
			threadErr <- fmt.Errorf("Failed to call epoll_wait %v", err)
			return
		}

		eventsChan <- nevents
	}()

	select {
	case <-started:
	case err := <-threadErr:
		t.Fatal(err)
	case <-time.After(500 * time.Millisecond):
		t.Fatalf("Didn't start after 500 ms")
	}
	ucpWorker.Signal()

	select {
	case nevents := <-eventsChan:
		if nevents <= 0 {
			t.Fatalf("No events in epoll %v", nevents)
		}
	case err := <-threadErr:
		t.Fatal(err)
	case <-time.After(wakeupTimeout):
		t.Fatalf("No events in epoll after %v", wakeupTimeout)
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

	quit := make(chan error, 1)
	started := make(chan bool)

	go func() {
		started <- true
		quit <- ucpWorker.Wait()
	}()

	<-started
	ucpWorker.Signal()

	select {
	case err := <-quit:
		if err != nil {
			t.Fatalf("Failed to wait %v", err)
		}
	case <-time.After(wakeupTimeout):
		t.Fatalf("Didn't exit from wait after %v", wakeupTimeout)
	}
}
