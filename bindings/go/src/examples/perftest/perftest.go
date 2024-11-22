/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package main

import (
	"errors"
	"flag"
	"fmt"
	"math"
	"net"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	. "github.com/openucx/ucx/bindings/go/src/ucx"
	"unsafe"
	. "github.com/openucx/ucx/bindings/go/src/cuda"
	"runtime"
	"github.com/docker/go-units"
)

type PerfTestParams struct {
	messageSizes  string
	memType       UcsMemoryType
	numThreads    uint
	numIterations uint
	port          uint
	wakeup        bool
	ip            string
	printInterval float64
	warmUpIter    uint
}

type PerfTest struct {
	context              *UcpContext
	memory               *UcpMemory
	memParams            *UcpMemAttributes
	worker               *UcpWorker
	listener             *UcpListener
	ep                   *UcpEp
	messageSize          uint64
	numCompletedRequests int32
	wg                   sync.WaitGroup
	wake                 []chan struct{}
	runProgress          int32
	statReport           int
}

const (
	STAT_CMD_RESET = iota
	STAT_CMD_PAUSE
	STAT_CMD_STOP
)

const (
	STAT_REPORT_SIZE = 1 << iota
	STAT_REPORT_ITER_NUM
)

var perfTestParams = PerfTestParams{}
var perfTest = PerfTest{}

// Returns address of current thread memory slice.
func getAddressOffsetForThread(t uint) unsafe.Pointer {
	var baseAddress uint = uint(uintptr(perfTest.memParams.Address))
	var offset uint = baseAddress + t*uint(perfTest.messageSize)
	return unsafe.Pointer(uintptr(offset))
}

// Printing functions

func printHeader() {
	dashes := strings.Repeat("-", 20)
	switch perfTest.statReport {
	case STAT_REPORT_ITER_NUM:
		fmt.Printf("|%20s|%20s|%20s|\n", "# iterations", "Bandwidth (Mb/s)", "Messages/s")
		fmt.Printf("|%20s|%20s|%20s|\n", dashes, dashes, dashes)
	case STAT_REPORT_SIZE:
		fmt.Printf("|%20s|%20s|%20s|\n", "Size", "Bandwidth (Mb/s)", "Messages/s")
		fmt.Printf("|%20s|%20s|%20s|\n", dashes, dashes, dashes)
	case STAT_REPORT_SIZE|STAT_REPORT_ITER_NUM:
		fmt.Printf("|%20s|%20s|%20s|%20s|\n", "# iterations", "Size", "Bandwidth (Mb/s)", "Messages/s")
		fmt.Printf("|%20s|%20s|%20s|%20s|\n", dashes, dashes, dashes, dashes)
	}
}

func printStatistics(statCmd chan int) {
	var last int32
	d := time.Duration(perfTestParams.printInterval * float64(time.Second))
	ticker := time.NewTicker(d)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			curr := atomic.LoadInt32(&perfTest.numCompletedRequests)
			rate := float64(curr - last) / perfTestParams.printInterval
			bw := float64(perfTest.messageSize) * rate * float64(1e-6)
			if perfTest.statReport & STAT_REPORT_SIZE != 0 {
				fmt.Printf("|%20d|%20s|%20f|%20f|\n", curr, "", bw, rate)
			} else {
				fmt.Printf("|%20d|%20f|%20f|\n", curr, bw, rate)
			}
			last = curr
		case cmd := <-statCmd:
			switch cmd {
			case STAT_CMD_PAUSE:
				ticker.Stop()
			case STAT_CMD_RESET:
				last = 0
				ticker.Reset(d)
			case STAT_CMD_STOP:
				return
			}
		}
	}
}

func printTotalStatistics(duration time.Duration) {
	totalBytesTransfered := perfTest.messageSize * uint64(perfTest.numCompletedRequests)
	bw := float64(totalBytesTransfered) * float64(1e-6) / duration.Seconds()
	rate := float64(perfTest.numCompletedRequests) / duration.Seconds()
	switch perfTest.statReport {
	case STAT_REPORT_SIZE|STAT_REPORT_ITER_NUM:
		fmt.Printf("|%20s|%20d|%20f|%20f|\n", "", perfTest.messageSize, bw, rate)
	case STAT_REPORT_SIZE:
		fmt.Printf("|%20d|%20f|%20f|\n", perfTest.messageSize, bw, rate)
	default:
		fmt.Printf("Number of iterations: %v, number of threads: %v, message size: %v, "+
	                   "memory type: %v, average bandwidth (Mb/s): %.3f \n", perfTest.numCompletedRequests,
			   perfTestParams.numThreads, perfTest.messageSize, perfTestParams.memType, bw)
        }
}

func nextSize(v uint64, step uint64) uint64 {
	pow2 := uint64(1);
	for pow2 <= v {
		pow2 *= 2
	}

	v = uint64(float64(v) * math.Pow(2, 1.0 / float64(step))) + 1;
	if v > pow2 {
		v = pow2
	}
	return v
}

func parseSizes(input string) (uint64, uint64, uint64) {
	values := strings.Split(input, ":")
	min, _ := units.RAMInBytes(values[0])
	if len(values) == 1 {
		return uint64(min), uint64(min), 1
	}
	max, _ := units.RAMInBytes(values[1])
	if len(values) == 2 {
		return uint64(min), uint64(max), 1
	}
	step, _ := strconv.ParseUint(values[2], 10, 64)
	return uint64(min), uint64(max), step
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
	mmapParams.SetLength(perfTest.messageSize * uint64(perfTestParams.numThreads))

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
	} else {
		os.Exit(0)
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

func progressThread() {
	for atomic.LoadInt32(&perfTest.runProgress) == 1 {
		progressWorker()
		for _, ch := range perfTest.wake {
			select {
			case ch <- struct{}{}:
			default:
			}
		}
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
		request, _ := data.Receive(getAddressOffsetForThread(tid), perfTest.messageSize,
			     (&UcpRequestParams{}).SetMemType(perfTestParams.memType))
		request.Close()
	}
	atomic.AddInt32(&perfTest.numCompletedRequests, 1)
	return UCS_OK
}

func serverStart() error {
	_, perfTest.messageSize, _ = parseSizes(perfTestParams.messageSizes)

	initContext()
	if err := initMemory(); err != nil {
		return err
	}

	initWorker()
	perfTest.worker.SetAmRecvHandler(0, UCP_AM_FLAG_WHOLE_MSG, serverAmRecvHandler)
	if err := initListener(); err != nil {
		return err
	}

	tryCudaSetDevice()
	for {
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
	request, err := perfTest.ep.SendAmNonBlocking(0, header, uint64(unsafe.Sizeof(t)), getAddressOffsetForThread(t), perfTest.messageSize, 0, requestParams)
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

func align(v uint, a uint) int32 {
	return int32((v + a - 1) / a * a)
}

func clientStart() error {
	min, max, step := parseSizes(perfTestParams.messageSizes)

	initContext()
	perfTest.messageSize = max
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

	warmUpIter := align(perfTestParams.warmUpIter, perfTestParams.numThreads)
	numIterations := align(perfTestParams.numIterations, perfTestParams.numThreads)

	if min != max {
		perfTest.statReport |= STAT_REPORT_SIZE
	}

	if perfTestParams.printInterval > 0 {
		perfTest.statReport |= STAT_REPORT_ITER_NUM
	}

	if perfTest.statReport != 0 {
		printHeader()
	}

	var start time.Time
	statCmd := func(int) {}
	if perfTestParams.printInterval > 0 {
		statCmdCh := make(chan int)
		statCmd = func(cmd int) {
			statCmdCh <- cmd
		}
		go printStatistics(statCmdCh)
	}

	atomic.StoreInt32(&perfTest.runProgress, 1)
	go progressThread()
	for perfTest.messageSize = min; perfTest.messageSize <= max; perfTest.messageSize = nextSize(perfTest.messageSize, step) {
		perfTest.numCompletedRequests = -warmUpIter
		statCmd(STAT_CMD_PAUSE)
		for perfTest.numCompletedRequests != numIterations {
			if perfTest.numCompletedRequests == 0 {
				start = time.Now()
				statCmd(STAT_CMD_RESET)
			}
			perfTest.wg.Add(int(perfTestParams.numThreads))
			for t := uint(0); t < perfTestParams.numThreads; t += 1 {
				go clientThreadDoIter(t)
			}
			perfTest.wg.Wait()
		}
		printTotalStatistics(time.Since(start))
	}
	atomic.StoreInt32(&perfTest.runProgress, 0)
	statCmd(STAT_CMD_STOP)

	closeAll()
	return nil
}

func main() {
	flag.UintVar(&perfTestParams.numThreads, "t", 1, "number of goroutines for send")
	flag.StringVar(&perfTestParams.messageSizes, "s", "4096", "sizes of the messages in bytes: min:max:step")
	flag.UintVar(&perfTestParams.port, "p", 36458, "port to bind")
	flag.UintVar(&perfTestParams.numIterations, "n", 1000, "number of iterations to run")
	flag.Float64Var(&perfTestParams.printInterval, "I", 1, "print summary every n seconds")
	flag.BoolVar(&perfTestParams.wakeup, "wakeup", false, "use polling: false(default)")
	flag.UintVar(&perfTestParams.warmUpIter, "warmup", 100, "warmup iterations")
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
