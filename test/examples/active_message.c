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

static int rank;
static int holder = 1;
static ucs_async_context_t async; /* Async event context manages times and fd notifications */
static uct_iface_attr_t iface_attr; /* Interface attributes: capabilities and limitations */
static uct_iface_config_t *iface_config; /* Defines interface configuration options */
static uct_iface_h iface; /* Communication interface context */
static uct_pd_h pd; /* Protection domain */
static uct_worker_h worker; /* Workers represent allocated resources in a communication thread */

/* Callback for active message */
static ucs_status_t hello_world(void *arg, void *data, size_t length, void *desc)
{
	printf("Hello World!!!\n");fflush(stdout);
	holder = 0;

	return UCS_OK;
}

/* Checks if the device and transports are supported by UCX */
static ucs_status_t resource_supported(char *dev_name, char *tl_name, int kill_iface)
{
	ucs_status_t status;
		
	/* Read transport-specific interface configuration */
	status = uct_iface_config_read(tl_name, NULL, NULL, &iface_config);
	if (UCS_OK != status) {
		fprintf(stderr, "Failed to setup iface_config.\n");fflush(stderr);
		goto error0;
	}	 

	/* Open communication interface */
	status = uct_iface_open(pd, worker, tl_name, dev_name, 0, iface_config, &iface);
	uct_iface_config_release(iface_config);
	if (UCS_OK != status) {
		fprintf(stderr, "Failed to open temporary interface.\n");fflush(stderr);
		goto error0;
	} 

	/* Get interface attributes */
	status = uct_iface_query(iface, &iface_attr);
	if (UCS_OK != status) {
		fprintf(stderr, "Failed to query iface.\n");fflush(stderr);
		goto error_iface0;
	}	 
	
	/* Check if current device and transport support short active messages */
	if (iface_attr.cap.flags & UCT_IFACE_FLAG_AM_SHORT) {
		if (kill_iface) {
			uct_iface_close(iface);
		}
		return UCS_OK;
	}

	return UCS_ERR_UNSUPPORTED; 

error_iface0:	
	uct_iface_close(iface);
error0:
	return status;
}

/* Device and transport to be used are determined by minimum latency */
static ucs_status_t dev_tl_lookup()
{
	int i;
	int j;
	uint64_t min_latency = UINT64_MAX;
	int pd_index = -1;
	int tl_index = -1;
	ucs_status_t status;
	uct_pd_resource_desc_t *pd_resources; /* Protection domain resource descriptor */
	uct_tl_resource_desc_t *tl_resources; /*Communication resource descriptor */
	unsigned num_pd_resources; /* Number of protected domain */
	unsigned num_tl_resources; /* Number of transport resources resource objects created */

	status = uct_query_pd_resources(&pd_resources, &num_pd_resources);
	if (UCS_OK != status) {
		fprintf(stderr, "Failed to query for protected domain resources.\n");	
		goto out1;
	}

	/* Iterate through protected domain resources */
	for (i = 0; i < num_pd_resources; ++i) {
		status = uct_pd_open(pd_resources[i].pd_name, &pd);
		if (UCS_OK != status) {
			fprintf(stderr, "Failed to open protected domain.\n"); fflush(stderr);
			goto release1;
		}

		status = uct_pd_query_tl_resources(pd, &tl_resources, &num_tl_resources);	
		if (UCS_OK != status) {
			fprintf(stderr, "Failed to query transport resources.\n"); fflush(stderr);
			uct_pd_close(pd);
			goto release1;
		}
		
		/* Go through each available transport resource for a particular protected domain 
		 * and keep track of the fastest latency */
		for (j = 0; j < num_tl_resources; ++j) {
			status = resource_supported(tl_resources[j].dev_name, tl_resources[j].tl_name, 1);	
			if (UCS_OK == status) {
				if (tl_resources[j].latency < min_latency) {
					min_latency = tl_resources[j].latency;
					pd_index = i;
					tl_index = j;
				}
			}
		}

		uct_release_tl_resource_list(tl_resources);
		uct_pd_close(pd);
	}

	/* Check if any valid device/transport found */
	if ((-1 == pd_index) || (-1 == tl_index)) {
		uct_release_pd_resource_list(pd_resources);
		return UCS_ERR_UNSUPPORTED;
	}

	/* IMPORTANT: Certain functions that operate on an interface rely on a pointer to the protection domain that created it */
	/* Reopen new protection domain and */
	status = uct_pd_open(pd_resources[pd_index].pd_name, &pd);
	if (UCS_OK != status) {
		fprintf(stderr, "Failed to open final protected domain.\n"); fflush(stderr);
		goto release1;
	}

	/* Open new tranport resources */
	status = uct_pd_query_tl_resources(pd, &tl_resources, &num_tl_resources);
	if (UCS_OK != status) {
		fprintf(stderr, "Failed to query final transport resources.\n"); fflush(stderr);
		uct_pd_close(pd);
		goto release1;
	}

	/* Call resource_supported() again to set the interface */
	status = resource_supported(tl_resources[tl_index].dev_name, tl_resources[tl_index].tl_name, 0);
	if (UCS_OK != status) {
		fprintf(stderr, "Failed to initialize final interface.\n"); fflush(stderr);
		uct_pd_close(pd);
		return status;	
	}

	printf("Using %s with %s.\n", tl_resources[tl_index].dev_name, tl_resources[tl_index].tl_name);fflush(stdout);

	uct_release_tl_resource_list(tl_resources);
release1:
	uct_release_pd_resource_list(pd_resources);
out1:
	return status;
}

int main(int argc, char **argv)
{
	MPI_Status mpi_status;
	int partner;
	int size;
	struct sockaddr *ep_addr; /* Endpoint address */
	struct sockaddr *iface_addr; /* Interface address */
	ucs_status_t status; /* status codes for UCS */
	ucs_thread_mode_t thread_mode = UCS_THREAD_MODE_SINGLE; /* Specifies thread sharing mode of an object */
	uct_ep_h ep; /* Remote endpoint */
	void *arg;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if (size < 2) {
		fprintf(stderr, "Failed to create enough mpi processes.\n");fflush(stderr);	
		return 1;
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
	status = ucs_async_context_init(&async, UCS_ASYNC_MODE_THREAD);
	if (UCS_OK != status) {
		fprintf(stderr, "Failed to init async context.\n");fflush(stderr);
		goto out;
	}	 

	/* Create a worker object */ 
	status = uct_worker_create(&async, thread_mode, &worker);
	if (UCS_OK != status) {
		fprintf(stderr, "Failed to create worker.\n");fflush(stderr);
		goto out_cleanup_async;
	}	 

	/* The device and tranport names are determined by latency */
	status = dev_tl_lookup();
	if (UCS_OK != status) {
		fprintf(stderr, "Failed to find supported device and transport\n");fflush(stderr);
		goto out_destroy_worker;
	}

	iface_addr = calloc(1, iface_attr.iface_addr_len);
	ep_addr = calloc(1, iface_attr.ep_addr_len);
	if ((NULL == iface_addr) || (NULL == ep_addr)) { 
		goto out_destroy_iface;
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
			goto out_free;
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
	/*Set active message handler */
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
out_destroy_iface:
	uct_iface_close(iface);
	uct_pd_close(pd);
out_destroy_worker:
	uct_worker_destroy(worker);
out_cleanup_async:
	ucs_async_context_cleanup(&async);
out:
	MPI_Finalize();
	return 0;
}
