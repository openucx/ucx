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
	w.Write(globalData[:size])
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

		serve, _ := uhttp.StartServer(addr, http.DefaultServeMux)
		serve()
	} else if doPut {
		obj := make([]byte, object_size)
		rand.Read(obj)
		fileobj := bytes.NewReader(obj)

		req, _ := http.NewRequest("PUT", "/data", fileobj)
		req.Header.Set("Content-Length", strconv.FormatUint(object_size, 10))
		dateHdr := time.Now().UTC().Format("20060102T150405Z")
		req.Header.Set("X-Amz-Date", dateHdr)
		t, _ := uhttp.NewClient(addr)
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
		fileobj := bytes.NewReader(obj)

		t, _ := uhttp.NewClient(addr)
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
		size, err := io.Copy(read, getResp.Body)
		fmt.Printf("Result %d %t\n", size, bytes.Equal(obj, read.Bytes()))
	} else if doGet {
		t, _ := uhttp.NewClient(addr)
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
		t, _ := uhttp.NewClient(addr)
		defer t.Close()
		client := &http.Client{Transport: t}

		resp, err := client.Get("/")
		fmt.Printf("%v %v\n", resp, err)
		defer resp.Body.Close()
		body, _ := ioutil.ReadAll(resp.Body)
		fmt.Printf("%s\n", body)
	}
}
