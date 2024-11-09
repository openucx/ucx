package main

import (
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"log"
	"net/http"
	"time"
	"math/rand"
	"bytes"
	"strconv"
	"strings"
	"unsafe"

	uhttp "github.com/openucx/ucx/bindings/go/src/ucx/http"
)

func rootHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "Welcome to the root page!")
}

func helloHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "Hello, World!")
}

var globalData []byte

func dataHandler(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		w.Header().Set("Content-Length", strconv.Itoa(len(globalData)))
		w.Write(globalData)
	case http.MethodPut:
		data, err := ioutil.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "Error reading request body", http.StatusInternalServerError)
			return
		}
		defer r.Body.Close()
		globalData = data
		w.WriteHeader(http.StatusOK)
	default:
		http.Error(w, "Invalid request method", http.StatusMethodNotAllowed)
	}
}

func dataHandler2(w http.ResponseWriter, r *http.Request) {
	size, _ := strconv.ParseInt(strings.TrimPrefix(r.URL.Path, "/data/"), 10, 64)
	w.Header().Set("Content-Length", strconv.FormatInt(size, 10))
	w.Write(globalData[:size])
}

func PrintHexComparison(slice1, slice2 []byte) {
	maxLen := len(slice1)
	if len(slice2) > maxLen {
		maxLen = len(slice2)
	}

	for i := 0; i < maxLen; i += 8 {
		fmt.Printf("0x%04X   ", i)
		hexStr1 := ""
		for j := i; j < i+8; j++ {
			if j < len(slice1) {
				hexStr1 += fmt.Sprintf("%02X ", slice1[j])
			} else {
				hexStr1 += "   "
			}
		}
		fmt.Printf("%-20s", hexStr1)

		hexStr2 := ""
		for j := i; j < i+8; j++ {
			if j < len(slice2) {
				hexStr2 += fmt.Sprintf("%02X ", slice2[j])
			} else {
				hexStr2 += "   "
			}
		}
		fmt.Printf("| %-20s\n", hexStr2)
	}
}

func main() {
	var (
		serverMode bool
		doPut bool
		doGet bool
		doLoop bool
		addr string
		object_size uint64 = 1<<30
	)

	flag := flag.NewFlagSet("myflag", flag.ExitOnError)
	flag.BoolVar(&serverMode, "s", false, "Server");
	flag.BoolVar(&doPut, "p", false, "PUT");
	flag.BoolVar(&doGet, "g", false, "GET");
	flag.BoolVar(&doLoop, "l", false, "PUT-GET");
	flag.StringVar(&addr, "a", "2.1.3.34:13337", "Address");
	flag.Uint64Var(&object_size, "m", 1<<30, "message size");

	if err := flag.Parse(os.Args[1:]); err != nil {
		os.Exit(1)
	}

	if serverMode {
		obj := make([]byte, object_size)
		rand.Read(obj)
		globalData = obj
		http.HandleFunc("/", rootHandler)
		http.HandleFunc("/hello", helloHandler)
		http.HandleFunc("/data", dataHandler)
		http.HandleFunc("/data/", dataHandler2)

		serve, err := uhttp.StartServer(addr, http.DefaultServeMux)
		if err != nil {
			log.Fatalf("FATAL: StartServer: %v", err)
		}
		fmt.Printf("Serve on %s\n", addr)
		serve()
	} else if doPut {
		obj := make([]byte, object_size)
		rand.Read(obj)
		fileobj := bytes.NewReader(obj)

		req, _ := http.NewRequest("PUT", "/data", fileobj)
		req.Header.Set("Content-Length", strconv.FormatUint(object_size, 10))
		dateHdr := time.Now().UTC().Format("20060102T150405Z")
		req.Header.Set("X-Amz-Date", dateHdr)
		t, _ := uhttp.NewTransport(addr)
		defer t.Close()
		client := &http.Client{Transport: t}
		resp, err := client.Do(req)
		defer resp.Body.Close()
		if err != nil {
			log.Fatalf("FATAL: Error uploading: %v", err)
		}
		fmt.Printf("Upload status %s: resp: %+v\n", resp.Status, resp)
	} else if doLoop {
		obj := make([]byte, object_size)
		rand.Read(obj)
		obj32 := *(*[]uint32)(unsafe.Pointer(&obj))
		obj32 = obj32[:len(obj)/4]
		for i := 0; i < len(obj32); i++ {
			obj32[i] = uint32(i)
		}
		fileobj := bytes.NewReader(obj)

		t, _ := uhttp.NewTransport(addr)
		defer t.Close()
		client := &http.Client{Transport: t}

		putReq, _ := http.NewRequest("PUT", "/data", fileobj)
		putReq.Header.Set("Content-Length", strconv.FormatUint(object_size, 10))
		dateHdr := time.Now().UTC().Format("20060102T150405Z")
		putReq.Header.Set("X-Amz-Date", dateHdr)
		putResp, err := client.Do(putReq)
		defer putResp.Body.Close()
		if err != nil {
			log.Fatalf("FATAL: Error uploading: %v", err)
		}
		fmt.Printf("Upload status %s: resp: %+v\n", putResp.Status, putResp)

		getReq, _ := http.NewRequest("GET", "/data", nil)
		getResp, err := client.Do(getReq)
		defer getResp.Body.Close()
		if err != nil {
			log.Fatalf("FATAL: Error uploading: %v", err)
		}
		read := new(bytes.Buffer)
		_, err = io.Copy(read, getResp.Body)
		if !bytes.Equal(obj, read.Bytes()) {
			PrintHexComparison(obj, read.Bytes())
		}
	} else if doGet {
		t, _ := uhttp.NewTransport(addr)
		defer t.Close()
		client := &http.Client{Transport: t}
		resp, _ := client.Get("/")
		defer resp.Body.Close()
		body, _ := ioutil.ReadAll(resp.Body)
		fmt.Printf("%s\n", body)

		for i := 0; i < 30 ; i++ {
			getReq, _ := http.NewRequest("GET", fmt.Sprintf("/data/%d", 1<<i), nil)
			getResp, err := client.Do(getReq)
			defer getResp.Body.Close()
			if err != nil {
				log.Fatalf("FATAL: Error uploading: %v", err)
			}
			read := new(bytes.Buffer)
			size, err := io.Copy(read, getResp.Body)
			fmt.Printf("Result %d\n", size)
		}
	} else {
		t, _ := uhttp.NewTransport(addr)
		defer t.Close()
		client := &http.Client{Transport: t}

		resp, err := client.Get("/")
		fmt.Printf("%v %v\n", resp, err)
		defer resp.Body.Close()
		body, _ := ioutil.ReadAll(resp.Body)
		fmt.Printf("%s\n", body)
	}
}
