/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#include <ucs/type/status.h>
#include <ucs/async/async.h>
#include <uct/api/uct.h>
#include <mpi.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

static int holder = 1;
static uct_context_h context;
static uct_iface_attr_t iface_attr; /* Interface attributes: capabilities and limitations */
static uct_iface_config_t *iface_config; /* Defines interface configuration options */
static uct_iface_h iface; /* Communication interface context */
static uct_worker_h worker; /* Workers represent allocated resources in a communication thread */

/* Callback for active message */
static ucs_status_t hello_world(void *arg, void *data, size_t length, void *desc)
{
	printf("Hello World!!!\n");fflush(stdout);
	holder = 0;

	return UCS_OK;
}

/* Checks if the device and transports are supported by UCX */
static ucs_status_t resource_supported(char *dev_name, char *tl_name)
{
	ucs_status_t status;
		
	/* Read transport-specific interface configuration */
	status = uct_iface_config_read(context, tl_name, NULL, NULL, &iface_config);
	if (UCS_OK != status) {
		fprintf(stderr, "Failed to setup iface_config.\n");fflush(stderr);
		goto error;
	}	 

	/* Open communication interface */
	status = uct_iface_open(worker, tl_name, dev_name, 0, iface_config, &iface);
	uct_iface_config_release(iface_config);
	if (UCS_OK != status) {
		fprintf(stderr, "Failed to open iface.\n");fflush(stderr);
		goto error;
	} 

	/* Get interface attributes */
	status = uct_iface_query(iface, &iface_attr);
	if (UCS_OK != status) {
		fprintf(stderr, "Failed to query iface.\n");fflush(stderr);
		goto iface_close;
	}	 
	
	/* Check if current device and transport support short active messages */
	if (iface_attr.cap.flags & UCT_IFACE_FLAG_AM_SHORT) {
		return UCS_OK;
	}

	return UCS_ERR_UNSUPPORTED;
	
iface_close:
	uct_iface_close(iface);
error:
	return status;
}

/* Compare function used for qsort */
static int cmp_res(const void *p1, const void *p2)
{
	uint64_t lat_1 = ((uct_resource_desc_t *) p1)->latency;
	uint64_t lat_2 = ((uct_resource_desc_t *) p2)->latency;

	if (lat_1 <= lat_2) {
		return -1;
	} else {
		return 1;
	}
}

/* Device and transport to be used are determined by minimum latency */
static ucs_status_t dev_tl_lookup(uct_resource_desc_t *resources, unsigned num_resources)
{
	int i;
	ucs_status_t rt;

	qsort(resources, num_resources, sizeof(uct_resource_desc_t), cmp_res);

	for (i=0; i<num_resources; i++) {
		rt = resource_supported(resources[i].dev_name, resources[i].tl_name);
		if (UCS_OK == rt) {
			break;
		}
	}

	return rt;
}

int main(int argc, char **argv)
{
	MPI_Status mpi_status;
	char dev_name[UCT_DEVICE_NAME_MAX]; /* Device name */
	char tl_name[UCT_TL_NAME_MAX]; /* Transport name */
	int partner = 0;
	int rank = 0;
	int size = 0;
	struct sockaddr *ep_addr; /* Endpoint address */
	struct sockaddr *iface_addr; /* Interface address */
	ucs_async_context_t async; /* Async event context manages times and fd notifications */
	ucs_status_t status; /* status codes for UCS */
	ucs_thread_mode_t thread_mode = UCS_THREAD_MODE_SINGLE; /* Specifies thread sharing mode of an object */
	uct_ep_h ep; /* Remote endpoint */
	uct_resource_desc_t *resources; /* Resource descriptor is an object representing the network resources */
	unsigned num_resources; /* Number of resource objects created */
	void *arg;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if (size < 2) {
		fprintf(stderr, "Failed to create enough mpi processes.\n");fflush(stderr);	
	}
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (0 == rank) { 
		partner = 1; 
	} else if (1 == rank) { 
		partner = 0; 
	} else { 
		MPI_Finalize(); 
		return 0; 
	}

	/* Initialize context */
	uct_init(&context);	
	status = ucs_async_context_init(&async, UCS_ASYNC_MODE_THREAD);
	if (UCS_OK != status) {
		fprintf(stderr, "Failed to init async context.\n");fflush(stderr);
		goto out;
	}	 

	/* Query for available resources */
	status = uct_query_resources(context, &resources, &num_resources);
	if (UCS_OK != status) {
		fprintf(stderr, "Failed to query resources.\n");fflush(stderr);
		goto out;
	}

	/* Create a worker object */ 
	status = uct_worker_create(context, &async, thread_mode, &worker);
	if (UCS_OK != status) {
		fprintf(stderr, "Failed to create worker.\n");fflush(stderr);
		goto out;
	}	 

	/* The device and tranport names are determined by latency */
	dev_tl_lookup(resources, num_resources);
	if (UCS_OK != status) {
		fprintf(stderr, "No supported hardware was found.\n");fflush(stderr);
		goto out_destroy_worker;
	}
	uct_release_resource_list(resources);

	iface_addr = calloc(1, iface_attr.iface_addr_len);
	ep_addr = calloc(1, iface_attr.ep_addr_len);
	if ((NULL == iface_addr) || (NULL == ep_addr)) { 
		goto out_destroy_worker;
	}

	/* Get interface address */
	status = uct_iface_get_address(iface, iface_addr);
	if (UCS_OK != status) {
		fprintf(stderr, "Failed to get interface address.\n");fflush(stderr);
		goto out_free;
	}	 
	
	if (iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
		/* Create new endpoint */
		status = uct_ep_create(iface, &ep);	
		if (UCS_OK != status) {
			fprintf(stderr, "Failed to create endpoint.\n");fflush(stderr);
			goto out_free_ep;
		}	 	
		/* Get endpoint address */
		status = uct_ep_get_address(ep, ep_addr);	
		if (UCS_OK != status) {
			fprintf(stderr, "Failed to get endpoint address.\n");fflush(stderr);
			goto out_free_ep;
		}	 	
	}

	/* Communicate interface and endpoint addresses to corresponding process */
	MPI_Send(iface_addr, iface_attr.iface_addr_len, MPI_BYTE, partner, 0, MPI_COMM_WORLD);
	MPI_Recv(iface_addr, iface_attr.iface_addr_len, MPI_BYTE, partner, 0, MPI_COMM_WORLD, &mpi_status);
	MPI_Send(ep_addr, iface_attr.ep_addr_len, MPI_BYTE, partner, 0, MPI_COMM_WORLD);
	MPI_Recv(ep_addr, iface_attr.ep_addr_len, MPI_BYTE, partner, 0, MPI_COMM_WORLD, &mpi_status);

	if (iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
		/* Connect endpoint to a remote endpoint */
		status = uct_ep_connect_to_ep(ep, ep_addr);
	} else if (iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
		/* Create an endpoint which is connected to a remote interface */
		status = uct_ep_create_connected(iface, iface_addr, &ep);
	} else status = UCS_ERR_UNSUPPORTED;
	if (UCS_OK != status) {
		fprintf(stderr, "Failed to connect endpoint\n");fflush(stderr);
		goto out_free_ep;
	}

	uint8_t id = 0; /* Tag for active message */
	/*Set active message handler for the interface */
	status = uct_iface_set_am_handler(iface, id, hello_world, arg);
	if (UCS_OK != status) {
		fprintf(stderr, "Failed to set callback.\n");fflush(stderr);
		goto out_free_ep;
	}	 	
	
	if (0 == rank) {
		uint64_t header;
		char payload[8];
		unsigned length = sizeof(payload);
		/* Send active message to remote endpoint */
		status = uct_ep_am_short(ep, id, header, payload, length);  		
	} else if (1 == rank) {
		while (holder) { 
			/* Explicitly progress any outstanding active message requests */
			uct_worker_progress(worker);
		}
	}

out_free_ep:
	uct_ep_destroy(ep);
out_free:
	free(iface_addr);
	free(ep_addr);
out_destroy_worker:
	uct_worker_destroy(worker);
out_cleanup_async:
	ucs_async_context_cleanup(&async);
out:
	uct_cleanup(context);
	MPI_Finalize();
	return 0;
}
