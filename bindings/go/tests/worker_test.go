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

// wakeupEntity is a context with the wakeup feature and a worker on it
type wakeupEntity struct {
	*TestEntity
	// Set when a goroutine of the test may still use the worker: destroying the
	// worker under that goroutine could crash the test binary
	leak bool
}

// prepareWakeupWorker creates a wakeupEntity, which is destroyed when the test
// ends unless its leak is set
func prepareWakeupWorker(t *testing.T, ucpWorkerParams *UcpWorkerParams) *wakeupEntity {
	entity := &wakeupEntity{
		TestEntity: prepareContext(t, (&UcpParams{}).EnableTag().EnableWakeup()),
	}

	worker, err := entity.context.NewWorker(ucpWorkerParams)
	if err != nil {
		entity.Close()
		t.Fatalf("Failed to create a worker %v", err)
	}

	entity.worker = worker
	t.Cleanup(func() {
		if !entity.leak {
			entity.Close()
		}
	})
	return entity
}

func TestUcpWorkerEfd(t *testing.T) {
	ucpWorkerParams := &UcpWorkerParams{}
	ucpWorkerParams.SetThreadMode(UCS_THREAD_MODE_SERIALIZED)
	ucpWorkerParams.WakeupEdge()
	entity := prepareWakeupWorker(t, ucpWorkerParams)
	ucpWorker := entity.worker

	eventsChan := make(chan int, 1)
	started := make(chan bool, 1)
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
	case <-time.After(wakeupTimeout):
		// The goroutine may still use the worker
		entity.leak = true
		t.Fatalf("Didn't start after %v", wakeupTimeout)
	}
	if err := ucpWorker.Signal(); err != nil {
		// The goroutine still waits on the worker efd
		entity.leak = true
		t.Fatalf("Failed to signal the worker %v", err)
	}

	select {
	case nevents := <-eventsChan:
		if nevents <= 0 {
			t.Fatalf("No events in epoll %v", nevents)
		}
	case err := <-threadErr:
		t.Fatal(err)
	case <-time.After(wakeupTimeout):
		// The goroutine still waits on the worker efd
		entity.leak = true
		t.Fatalf("No events in epoll after %v", wakeupTimeout)
	}
}

func TestUcpWorkerWakeup(t *testing.T) {
	ucpWorkerParams := &UcpWorkerParams{}
	var cpuMask big.Int
	cpuMask.SetBit(&cpuMask, 0, 1)
	ucpWorkerParams.SetCpuMask(&cpuMask).WakeupTX().WakeupRX()
	ucpWorkerParams.SetThreadMode(UCS_THREAD_MODE_MULTI)
	entity := prepareWakeupWorker(t, ucpWorkerParams)
	ucpWorker := entity.worker

	quit := make(chan error, 1)
	started := make(chan bool)

	go func() {
		started <- true
		quit <- ucpWorker.Wait()
	}()

	<-started
	if err := ucpWorker.Signal(); err != nil {
		// The goroutine may still use the worker
		entity.leak = true
		t.Fatalf("Failed to signal the worker %v", err)
	}

	select {
	case err := <-quit:
		if err != nil {
			t.Fatalf("Failed to wait %v", err)
		}
	case <-time.After(wakeupTimeout):
		// The goroutine is still in ucpWorker.Wait()
		entity.leak = true
		t.Fatalf("Didn't exit from wait after %v", wakeupTimeout)
	}
}
