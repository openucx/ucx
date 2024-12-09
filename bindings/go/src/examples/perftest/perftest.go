/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package main

import (
	"errors"
	"flag"
	"fmt"
	"net"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	. "github.com/openucx/ucx/bindings/go/src/ucx"
	"unsafe"
	. "github.com/openucx/ucx/bindings/go/src/cuda"
	"runtime"
)

type PerfTestParams struct {
	messageSize   uint64
	memType       UcsMemoryType
	numThreads    uint
	numIterations uint
	port          uint
	wakeup        bool
	ip            string
	printInterval uint
	warmUpIter    uint
}

type PerfTest struct {
	context              *UcpContext
	memory               *UcpMemory
	memParams            *UcpMemAttributes
	worker               *UcpWorker
	listener             *UcpListener
	ep                   *UcpEp
	numCompletedRequests int32
	wg                   sync.WaitGroup
	quit                 chan struct{}
	wake                 []chan struct{}
}

var perfTestParams = PerfTestParams{}
var perfTest = PerfTest{}

// Returns address of current thread memory slice.
func getAddressOffsetForThread(t uint) unsafe.Pointer {
	var baseAddress uint = uint(uintptr(perfTest.memParams.Address))
	var offset uint = baseAddress + t*uint(perfTestParams.messageSize)
	return unsafe.Pointer(uintptr(offset))
}

// Printing functions

func printRule() {
	dashes := strings.Repeat("-", 20)
	fmt.Printf("%20s|%20s|%20s|\n", dashes, dashes, dashes)
}

func printHeader() {
	printRule()
	fmt.Printf("%20s|%20s|%20s|\n", "# iterations", "Bandwidth (Mb/s)", "Messages")
	printRule()
}

func printStatistics() {
	last := atomic.LoadInt32(&perfTest.numCompletedRequests)
	ticker := time.NewTicker(time.Duration(perfTestParams.printInterval) * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			curr := atomic.LoadInt32(&perfTest.numCompletedRequests)
			rate := (curr - last) / int32(perfTestParams.printInterval)
			bw := float64(perfTestParams.messageSize) * float64(rate) * float64(1e-6)
			fmt.Printf("%20d|%20f|%20d|\n", curr, bw, rate)
			last = curr
		case <-perfTest.quit:
			return
		}
	}
}

func printTotalStatistics(duration time.Duration) {
	totalBytesTransfered := perfTestParams.messageSize * uint64(perfTest.numCompletedRequests)
	avgBw := float64(totalBytesTransfered) * float64(1e-6) / duration.Seconds()

	printRule()
	fmt.Printf("Number of iterations: %v, number of threads: %v, message size: %v, "+
		"memory type: %v, average bandwidth (Mb/s): %.3f \n", perfTestParams.numIterations,
		perfTestParams.numThreads, perfTestParams.messageSize, perfTestParams.memType, avgBw)
}

func initContext() {
	params := (&UcpParams{}).EnableAM()

	if perfTestParams.wakeup {
		params.EnableWakeup()
	}

	perfTest.context, _ = NewUcpContext(params)
}

func tryCudaSetDevice() {
	if perfTestParams.memType == UCS_MEMORY_TYPE_CUDA {
		runtime.LockOSThread()
		if ret := CudaSetDevice(); ret != nil {
			panic(ret)
		}
	}
}

func initMemory() error {
	var err error
	memTypeMask, _ := perfTest.context.MemoryTypesMask()

	if !IsMemTypeSupported(perfTestParams.memType, memTypeMask) {
		return errors.New("requested memory type is unsupported")
	}

	mmapParams := &UcpMmapParams{}
	mmapParams.SetMemoryType(perfTestParams.memType).Allocate()
	mmapParams.SetLength(perfTestParams.messageSize * uint64(perfTestParams.numThreads))

	tryCudaSetDevice()

	perfTest.memory, err = perfTest.context.MemMap(mmapParams)
	if err != nil {
		return err
	}

	perfTest.memParams, err = perfTest.memory.Query(UCP_MEM_ATTR_FIELD_ADDRESS)
	if err != nil {
		return err
	}
	return nil
}

func initWorker() {
	workerParams := (&UcpWorkerParams{}).SetThreadMode(UCS_THREAD_MODE_MULTI)

	if perfTestParams.wakeup {
		workerParams.WakeupTX()
		workerParams.WakeupRX()
	}

	perfTest.worker, _ = perfTest.context.NewWorker(workerParams)
}

func epErrorHandling(ep *UcpEp, status UcsStatus) {
	if status != UCS_ERR_CONNECTION_RESET {
		errorString := fmt.Sprintf("Endpoint error: %v", status.String())
		panic(errorString)
	}
}

func flush() error {
	request, err := perfTest.ep.FlushNonBlocking(nil)
	if err != nil {
		return err
	}

	for request.GetStatus() == UCS_INPROGRESS {
		progressWorker()
	}

	if status := request.GetStatus(); status != UCS_OK {
		return NewUcxError(status)
	}

	request.Close()
	return nil
}

func clientConnectWorker() error {
	var err error
	epParams := &UcpEpParams{}
	serverAddress, _ := net.ResolveTCPAddr("tcp", fmt.Sprintf("%v:%v", perfTestParams.ip, perfTestParams.port))
	epParams.SetPeerErrorHandling().SetErrorHandler(epErrorHandling).SetSocketAddress(serverAddress)

	perfTest.ep, err = perfTest.worker.NewEndpoint(epParams)
	if err != nil {
		return err
	}

	return flush()
}

func initListener() error {
	var err error
	listenerParams := &UcpListenerParams{}
	addr, _ := net.ResolveTCPAddr("tcp", fmt.Sprintf("0.0.0.0:%v", perfTestParams.port))

	listenerParams.SetSocketAddress(addr)
	listenerParams.SetConnectionHandler(func(connRequest *UcpConnectionRequest) {
		perfTest.ep, _ = perfTest.worker.NewEndpoint(
			(&UcpEpParams{}).SetConnRequest(connRequest).SetErrorHandler(epErrorHandling).SetPeerErrorHandling())

		fmt.Printf("Got connection. Starting benchmark...\n")
	})

	perfTest.listener, err = perfTest.worker.NewListener(listenerParams)
	if err != nil {
		return err
	}
	fmt.Printf("Started receiver listener on address: %v \n", addr)
	return nil
}

func progressWorker() {
	for perfTest.worker.Progress() != 0 {
	}
	if perfTestParams.wakeup {
		perfTest.worker.Wait()
	}
}

func closeAll() {
	if perfTest.ep != nil {
		perfTest.ep.CloseNonBlockingForce(nil)
	}

	if perfTest.listener != nil {
		perfTest.listener.Close()
	}

	if perfTest.worker != nil {
		perfTest.worker.Close()
	}

	if perfTest.memory != nil {
		perfTest.memory.Close()
	}

	if perfTest.context != nil {
		perfTest.context.Close()
	}
}

func serverAmRecvHandler(header unsafe.Pointer, headerSize uint64, data *UcpAmData, replyEp *UcpEp) UcsStatus {
	tid := *(*uint)(header)
	if !data.IsDataValid() {
		request, _ := data.Receive(getAddressOffsetForThread(tid), perfTestParams.messageSize,
			     (&UcpRequestParams{}).SetMemType(perfTestParams.memType))
		request.Close()
	}
	atomic.AddInt32(&perfTest.numCompletedRequests, 1)
	return UCS_OK
}

func serverStart() error {
	initContext()
	if err := initMemory(); err != nil {
		return err
	}

	initWorker()
	perfTest.worker.SetAmRecvHandler(0, UCP_AM_FLAG_WHOLE_MSG, serverAmRecvHandler)
	if err := initListener(); err != nil {
		return err
	}

	totalNumRequests := int32(perfTestParams.warmUpIter + perfTestParams.numIterations)
	tryCudaSetDevice()

	for atomic.LoadInt32(&perfTest.numCompletedRequests) < totalNumRequests {
		progressWorker()
	}

	flush()
	closeAll()
	return nil
}

func clientThreadDoIter(t uint) {
	tryCudaSetDevice()

	requestParams := (&UcpRequestParams{}).SetMemType(perfTestParams.memType)

	header := unsafe.Pointer(&t)
	request, err := perfTest.ep.SendAmNonBlocking(0, header, uint64(unsafe.Sizeof(t)), getAddressOffsetForThread(t), perfTestParams.messageSize, 0, requestParams)
	if err != nil {
		panic(err)
	}
	for request.GetStatus() == UCS_INPROGRESS {
		<-perfTest.wake[t]
	}
	if request.GetStatus() != UCS_OK {
		errorString := fmt.Sprintf("Request completion error: %v", request.GetStatus().String())
		panic(errorString)
	}
	request.Close()

	atomic.AddInt32(&perfTest.numCompletedRequests, 1)
	perfTest.wg.Done()
}

func clientProgress() {
	for atomic.LoadInt32(&perfTest.numCompletedRequests) < int32(perfTestParams.numIterations) {
		progressWorker()
		for _, ch := range perfTest.wake {
			select {
			case ch <- struct{}{}:
			default:
			}
		}
	}
}

func clientStart() error {
	initContext()
	if err := initMemory(); err != nil {
		return err
	}

	initWorker()
	if err := clientConnectWorker(); err != nil {
		return err
	}

	perfTest.wake = make([]chan struct{}, perfTestParams.numThreads)
	for i := range perfTest.wake {
		perfTest.wake[i] = make(chan struct{})
	}

	var start time.Time
	perfTest.quit = make(chan struct{})
	perfTest.numCompletedRequests = int32(-perfTestParams.warmUpIter)
	printHeader()
	go printStatistics()
	go clientProgress()
	for {
		threads := perfTestParams.numThreads
		if perfTest.numCompletedRequests <= 0 {
			start = time.Now()
		} else if perfTest.numCompletedRequests == int32(perfTestParams.numIterations) {
			break
		} else if perfTest.numCompletedRequests > int32(perfTestParams.numIterations - perfTestParams.numThreads) {
			threads = perfTestParams.numIterations - uint(perfTest.numCompletedRequests)
		}
		perfTest.wg.Add(int(threads))
		for t := uint(0); t < threads; t += 1 {
			go clientThreadDoIter(t)
		}
		perfTest.wg.Wait()
	}
	printTotalStatistics(time.Since(start))
	close(perfTest.quit)

	closeAll()
	return nil
}

func main() {
	flag.UintVar(&perfTestParams.numThreads, "t", 1, "number of goroutines for send")
	flag.Uint64Var(&perfTestParams.messageSize, "s", 4096, "size of the message in bytes")
	flag.UintVar(&perfTestParams.port, "p", 36458, "port to bind")
	flag.UintVar(&perfTestParams.numIterations, "n", 1000, "number of iterations to run")
	flag.UintVar(&perfTestParams.printInterval, "I", 1, "print summary every n seconds")
	flag.BoolVar(&perfTestParams.wakeup, "wakeup", false, "use polling: false(default)")
	flag.UintVar(&perfTestParams.warmUpIter, "warmup", 100, "warmup iterations")
	flag.StringVar(&perfTestParams.ip, "i", "", "server address to connect")

	perfTestParams.memType = UCS_MEMORY_TYPE_HOST
	flag.Var(&perfTestParams.memType, "m", "memory type: host(default), cuda")

	flag.Parse()

	var err error
	if perfTestParams.ip == "" {
		err = serverStart()
	} else {
		err = clientStart()
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "error running benchmark: %v\n", err)
		os.Exit(1)
	}

}
