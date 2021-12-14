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
	. "ucx"
	"unsafe"
)

type PerfTestParams struct {
	messageSize   uint64
	memType       UcsMemoryType
	numThreads    uint
	numIterations uint
	port          uint
	wakeup        bool
	ip            string
	printIter     uint
	warmUpIter    uint
}

type PerfTest struct {
	context              *UcpContext
	memory               *UcpMemory
	memParams            *UcpMemAttributes
	perThreadWorkers     []*UcpWorker
	listener             *UcpListener
	eps                  []*UcpEp
	reverseEps           []*UcpEp
	numCompletedRequests uint32
	wg                   sync.WaitGroup
	completionTime       []time.Duration
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
func printHeader() {
	dashes := strings.Repeat("-", 20)
	fmt.Printf("|%20s|%20s|%20s|%20s|\n", dashes, dashes, dashes, dashes)
	fmt.Printf("|%20s|%20s|%20s|%20s|\n", "Thread", "Iteration", "Latency ", "Bandwidth (Gb/s)")
	fmt.Printf("|%20s|%20s|%20s|%20s|\n", dashes, dashes, dashes, dashes)
}

func printPerThreadStatistics(i uint, t uint) {
	bw := float64(perfTestParams.messageSize) * float64(8e-9) / perfTest.completionTime[t].Seconds()
	fmt.Printf("|%20s|%20s|%20s|%20f|\n", fmt.Sprintf("%v/%v", t+1, perfTestParams.numThreads),
		fmt.Sprintf("%v/%v", i, perfTestParams.numIterations), perfTest.completionTime[t], bw)
}

func printTotalStatistics(duration time.Duration) {
	totalBytesTransfered := perfTestParams.messageSize * uint64(perfTestParams.numThreads) * uint64(perfTestParams.numIterations)
	avgLat := float64(duration.Milliseconds()) / float64(perfTestParams.numIterations)
	avgBw := float64(totalBytesTransfered) * float64(8e-9) / duration.Seconds()

	dashes := strings.Repeat("-", 20)
	fmt.Printf("|%20s|%20s|%20s|%20s|\n", dashes, dashes, dashes, dashes)
	fmt.Printf("Number of iterations: %v, number of threads: %v, message size: %v, "+
		"memory type: %v, average latency (ms): %v, average bandwidth (Gb/s): %.3f \n", perfTestParams.numIterations,
		perfTestParams.numThreads, perfTestParams.messageSize, perfTestParams.memType, avgLat, avgBw)
}

func initContext() {
	params := (&UcpParams{}).EnableAM()

	if perfTestParams.wakeup {
		params.EnableWakeup()
	}

	perfTest.context, _ = NewUcpContext(params)
}

func initMemory() error {
	var err error
	var dummyMemh *UcpMemory
	memTypeMask, _ := perfTest.context.MemoryTypesMask()

	if !IsMemTypeSupported(perfTestParams.memType, memTypeMask) {
		return errors.New("requested memory type is unsupported")
	}

	mmapParams := &UcpMmapParams{}
	// Allocate dummy host memory, to initialize cuda context if the memType is Cuda.
	mmapParams.SetMemoryType(UCS_MEMORY_TYPE_HOST).Allocate()
	mmapParams.SetLength(1)
	dummyMemh, err = perfTest.context.MemMap(mmapParams)
	if err != nil {
		return err
	}
	dummyMemh.Close()

	mmapParams.SetMemoryType(perfTestParams.memType).Allocate()
	mmapParams.SetLength(perfTestParams.messageSize * uint64(perfTestParams.numThreads))

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

func initWorker(i int) {
	workerParams := (&UcpWorkerParams{}).SetThreadMode(UCS_THREAD_MODE_MULTI)

	if perfTestParams.wakeup {
		workerParams.WakeupTX()
		workerParams.WakeupRX()
	}

	perfTest.perThreadWorkers[i], _ = perfTest.context.NewWorker(workerParams)
}

func epErrorHandling(ep *UcpEp, status UcsStatus) {
	if status != UCS_ERR_CONNECTION_RESET {
		fmt.Printf("Endpoint error: %v \n", status.String())
	}
}

func clientConnectWorker(i int) error {
	var err error
	epParams := &UcpEpParams{}
	serverAddress, _ := net.ResolveTCPAddr("tcp", fmt.Sprintf("%v:%v", perfTestParams.ip, perfTestParams.port))
	epParams.SetPeerErrorHandling().SetErrorHandler(epErrorHandling).SetSocketAddress(serverAddress)

	perfTest.eps[i], err = perfTest.perThreadWorkers[i].NewEndpoint(epParams)
	if err != nil {
		return err
	}

	request, err := perfTest.eps[i].FlushNonBlocking(nil)
	if err != nil {
		return err
	}

	for request.GetStatus() == UCS_INPROGRESS {
		progressWorker(i)
	}

	if status := request.GetStatus(); status != UCS_OK {
		return NewUcxError(status)
	}

	request.Close()
	return nil
}

func initListener() error {
	var err error
	listenerParams := &UcpListenerParams{}
	addr, _ := net.ResolveTCPAddr("tcp", fmt.Sprintf("0.0.0.0:%v", perfTestParams.port))

	listenerParams.SetSocketAddress(addr)
	listenerParams.SetConnectionHandler(func(connRequest *UcpConnectionRequest) {
		// No need to synchronize, since reverse eps creating from a single thread.
		numConnections := len(perfTest.reverseEps)
		reverseEp, _ := perfTest.perThreadWorkers[1+numConnections].NewEndpoint(
			(&UcpEpParams{}).SetConnRequest(connRequest).SetErrorHandler(epErrorHandling).SetPeerErrorHandling())

		perfTest.reverseEps = append(perfTest.reverseEps, reverseEp)
		fmt.Printf("Got connection for thread %v. Starting benchmark...\n", numConnections)
	})

	perfTest.listener, err = perfTest.perThreadWorkers[0].NewListener(listenerParams)
	if err != nil {
		return err
	}
	fmt.Printf("Started receiver listener on address: %v \n", addr)
	return nil
}

func progressWorker(i int) {
	for perfTest.perThreadWorkers[i].Progress() != 0 {
	}
	if perfTestParams.wakeup {
		perfTest.perThreadWorkers[i].Wait()
	}
}

func close() {
	for _, reverseEp := range perfTest.reverseEps {
		reverseEp.CloseNonBlockingForce(nil)
	}

	for _, ep := range perfTest.eps {
		ep.CloseNonBlockingForce(nil)
	}

	if perfTest.listener != nil {
		perfTest.listener.Close()
	}

	for _, worker := range perfTest.perThreadWorkers {
		worker.Close()
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
	if data.IsDataValid() {
		atomic.AddUint32(&perfTest.numCompletedRequests, 1)
	} else {
		data.Receive(getAddressOffsetForThread(tid), perfTestParams.messageSize,
			(&UcpRequestParams{}).SetMemType(perfTestParams.memType).SetCallback(func(request *UcpRequest, status UcsStatus, length uint64) {
				atomic.AddUint32(&perfTest.numCompletedRequests, 1)
				request.Close()
			}))
	}
	return UCS_OK
}

func serverStart() error {
	initContext()
	if err := initMemory(); err != nil {
		return err
	}
	// 1 global worker for listener progress and N threads for data receive.
	perfTest.perThreadWorkers = make([]*UcpWorker, perfTestParams.numThreads+1)
	initWorker(0)
	if err := initListener(); err != nil {
		return err
	}

	// Submit AM recv handler for each thread
	for t := uint(0); t < perfTestParams.numThreads; t += 1 {
		initWorker(int(t) + 1)
		perfTest.perThreadWorkers[t+1].SetAmRecvHandler(t, UCP_AM_FLAG_WHOLE_MSG, serverAmRecvHandler)
	}

	for i := uint(0); i < perfTestParams.numIterations; i += 1 {
		perfTest.numCompletedRequests = 0

		perfTest.wg.Add(int(perfTestParams.numThreads + 1))
		for t := uint(0); t < perfTestParams.numThreads+1; t += 1 {
			go func(tid uint) {
				for perfTest.numCompletedRequests < uint32(perfTestParams.numThreads) {
					progressWorker(int(tid))
				}
				perfTest.wg.Done()
			}(t)
		}

		perfTest.wg.Wait()
	}
	close()
	return nil
}

func clientThreadDoIter(i int, t uint) {
	start := time.Now()
	var request *UcpRequest
	requestParams := (&UcpRequestParams{}).SetMemType(perfTestParams.memType)

	header := unsafe.Pointer(&t)
	request, _ = perfTest.eps[t].SendAmNonBlocking(t, header, uint64(unsafe.Sizeof(t)), getAddressOffsetForThread(t), perfTestParams.messageSize, 0, requestParams)

	for request.GetStatus() == UCS_INPROGRESS {
		progressWorker(int(t))
	}
	perfTest.completionTime[t] = time.Since(start)
	request.Close()

	if (i > 0) && (uint(i)%perfTestParams.printIter) == 0 {
		printPerThreadStatistics(uint(i), t)
	}
	perfTest.wg.Done()
}

func clientStart() error {
	initContext()
	if err := initMemory(); err != nil {
		return err
	}

	perfTest.perThreadWorkers = make([]*UcpWorker, perfTestParams.numThreads)
	perfTest.eps = make([]*UcpEp, perfTestParams.numThreads)
	for i := 0; i < int(perfTestParams.numThreads); i += 1 {
		initWorker(i)
		if err := clientConnectWorker(i); err != nil {
			return err
		}
	}

	var totalDuration time.Duration = 0
	printHeader()
	for i := -int(perfTestParams.warmUpIter); i < int(perfTestParams.numIterations); i += 1 {
		perfTest.wg.Add(int(perfTestParams.numThreads))
		for t := uint(0); t < perfTestParams.numThreads; t += 1 {
			go clientThreadDoIter(i, t)
		}
		perfTest.wg.Wait()
		var maxDuration time.Duration = 0
		for _, threadDuration := range perfTest.completionTime {
			if threadDuration > maxDuration {
				maxDuration = threadDuration
			}
		}
		totalDuration += maxDuration
	}
	printTotalStatistics(totalDuration)

	close()
	return nil
}

func main() {
	flag.UintVar(&perfTestParams.numThreads, "t", 1, "number of threads for send: 1(default)")
	flag.Uint64Var(&perfTestParams.messageSize, "s", 4096, "size of the message in bytes: 4096(default)")
	flag.UintVar(&perfTestParams.port, "p", 36458, "port to bind: 36458(default)")
	flag.UintVar(&perfTestParams.numIterations, "n", 1000, "Number of iterations to run: 1000(default)")
	flag.UintVar(&perfTestParams.printIter, "printIter", 100, "Print summary every n iterations: 1000(default)")
	flag.BoolVar(&perfTestParams.wakeup, "wakeup", false, "use polling: false(default)")
	flag.UintVar(&perfTestParams.warmUpIter, "warmup", 5, "warmup iterations: 5(default)")
	flag.StringVar(&perfTestParams.ip, "i", "", "server address to connect")

	perfTestParams.memType = UCS_MEMORY_TYPE_HOST
	flag.CommandLine.Func("m", "memory type: host(default), cuda", func(p string) error {
		mtypeStr := strings.ToLower(p)
		if mtypeStr == "host" {
			perfTestParams.memType = UCS_MEMORY_TYPE_HOST
		} else if mtypeStr == "cuda" {
			perfTestParams.memType = UCS_MEMORY_TYPE_CUDA
		} else {
			return errors.New("memory type can be host or cuda")
		}
		return nil
	})

	flag.Parse()

	perfTest.completionTime = make([]time.Duration, perfTestParams.numThreads)
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
