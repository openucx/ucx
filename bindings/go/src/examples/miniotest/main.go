package main

import (
	"bytes"
	"context"
	"fmt"
	"flag"
	"log"
	"io"
	"math/rand"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"

	"github.com/openucx/ucx/bindings/go/src/ucx/http"

	"code.cloudfoundry.org/bytefmt"
)

func client(ucx bool, addr string) (*minio.Client, error) {
	accessKeyID := "minioadmin"
	secretAccessKey := "minioadmin"
	useSSL := false

	if !ucx {
		return minio.New(addr, &minio.Options{
			Creds:  credentials.NewStaticV4(accessKeyID, secretAccessKey, ""),
			Secure: useSSL,
		})
	}

	transport, err := http.NewTransport(addr)
	if err != nil {
		log.Fatalf("http.NewClient %v", err)
	}

	return minio.New(addr, &minio.Options{
		Creds:  credentials.NewStaticV4(accessKeyID, secretAccessKey, ""),
		Secure: useSSL,
		Transport: transport,
	})
}

func main() {
	var ucx bool
	var sizeArg string
	var addr string

	flag.BoolVar(&ucx, "u", false, "")
	flag.StringVar(&sizeArg, "z", "1M", "Size of objects in bytes with postfix K, M, and G")
	flag.StringVar(&addr, "a", "2.1.3.34:3334", "address")
	flag.Parse()

	object_size, err := bytefmt.ToBytes(sizeArg);
	if err != nil {
		log.Fatalf("Invalid -z argument for object size: %v", err)
	}

	minioClient, err := client(ucx, addr)
	if err != nil {
		log.Fatalf("minio.New %v", err)
	}

	bucketName := "mybucket"
	location := "us-east-1"

	err = minioClient.MakeBucket(context.Background(), bucketName, minio.MakeBucketOptions{Region: location})
	if err != nil {
		// Check to see if the bucket already exists
		exists, errBucketExists := minioClient.BucketExists(context.Background(), bucketName)
		if errBucketExists == nil && exists {
			log.Printf("Bucket %s already exists\n", bucketName)
		} else {
			log.Fatalf("minio.BucketExists %v", errBucketExists)
		}
	} else {
		log.Printf("Successfully created bucket %s\n", bucketName)
	}

	objectName := "myfile.txt"
	contentType := "text/plain"
	fileContent := make([]byte, object_size)
	rand.Read(fileContent)


	reader := bytes.NewReader([]byte(fileContent))
	objectSize := int64(len(fileContent))

	info, err := minioClient.PutObject(context.Background(), bucketName, objectName, reader, objectSize, minio.PutObjectOptions{ContentType: contentType})
	if err != nil {
		log.Fatalln(err)
	}

	log.Printf("Successfully uploaded %s of size %d\n", objectName, info.Size)

	object, err := minioClient.GetObject(context.Background(), bucketName, objectName, minio.GetObjectOptions{})
	if err != nil {
		log.Fatalln(err)
	}
	defer object.Close()

	var buf bytes.Buffer
	if _, err = io.Copy(&buf, object); err != nil {
		log.Fatalln(err)
	}

	if bytes.Equal(fileContent, buf.Bytes()) {
		fmt.Printf("Downloaded ok\n")
	}
}
