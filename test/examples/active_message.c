#include <ucs/type/status.h>
#include <ucs/async/async.h>
#include <uct/api/uct.h>
#include <ucp/api/ucp.h>
#include <mpi.h>
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

static ucs_status_t hello_world(void *arg, void *data, size_t length, void *desc){
	fprintf(stderr, "Hello World!!!\n");fflush(stdout);
	exit(0);
}

int main(int argc, char **argv){
	MPI_Status mpi_status;
	char dev_name[UCT_DEVICE_NAME_MAX] = "mthca0:1";/*UCX expects dev_name to be "device:port"*/
	char tl_name[UCT_TL_NAME_MAX] = "ud";/*Transport name*/
	int i = 0;
	int message = 0;
	int partner = 0;
	int rank = 0;
	int size = 0;
	struct sockaddr *ep_addr;
	struct sockaddr *iface_addr;
	ucs_async_context_t async;
	ucs_status_t status;
	ucs_thread_mode_t thread_mode = UCS_THREAD_MODE_SINGLE;
	uct_context_h context;
	uct_ep_h ep;
	uct_iface_attr_t iface_attr;
	uct_iface_config_t *iface_config;
	uct_iface_h iface;
	uct_worker_h worker;
	void *arg;

	fprintf(stderr,"ActiveMessage start (before mpi)\n");fflush(stderr);

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	partner = (0 == rank) ? 1 : 0;

/*<<<<<<<<<<<<<<<<<<<<<<<DEBUG
	int holder=1;
	char name[_SC_HOST_NAME_MAX];
	gethostname(name, sizeof(name));
	printf("ssh %s gdb -p %d\n", name , getpid());
	while(holder){}
//<<<<<<<<<<<<<<<<<<<<<<<DEBUG*/

	uct_init(&context);	
	status = ucs_async_context_init(&async, UCS_ASYNC_MODE_THREAD);
	if(status != UCS_OK){
		fprintf(stderr, "Failed to init async context.\n");
		goto out;
	}	 

	status = uct_worker_create(context, &async, thread_mode, &worker);
	if(status != UCS_OK){
		fprintf(stderr, "Failed to create worker.\n");
		goto out_cleanup_async;
	}	 

	status = uct_iface_config_read(context, tl_name, NULL, NULL, &iface_config);
	if(status != UCS_OK){
		fprintf(stderr, "Failed to setup iface_config.\n");
		goto out_destroy_worker;
	}	 

	status = uct_iface_open(worker, tl_name, dev_name, 0, iface_config, &iface);
	uct_iface_config_release(iface_config);
	if(status != UCS_OK){
		fprintf(stderr, "Failed to open iface.\n");
		goto out_destroy_worker;
	} 

	status = uct_iface_query(iface, &iface_attr);
	if(status != UCS_OK){
		fprintf(stderr, "Failed to query iface.\n");
		goto out_destroy_worker;
	}	 
	
	iface_addr = calloc(1, iface_attr.iface_addr_len);
	ep_addr = calloc(1, iface_attr.ep_addr_len);
	if((NULL == iface_addr) || (NULL == ep_addr)){ 
		goto out_destroy_worker;
	}

	status = uct_iface_get_address(iface, iface_addr);
	if(status != UCS_OK){
		fprintf(stderr, "Failed to get interface address.\n");
		goto out_free;
	}	 
	
	//only one endpoint needed for now
	if(iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP){
		status = uct_ep_create(iface, &ep);	
		if(status != UCS_OK){
			fprintf(stderr, "Failed to create endpoint.\n");
			goto out_free_ep;
		}	 	
		status = uct_ep_get_address(ep, ep_addr);	
		if(status != UCS_OK){
			fprintf(stderr, "Failed to get endpoint address.\n");
			goto out_free_ep;
		}	 	
	}

	fprintf(stderr,"MPI messages for connect\n");fflush(stderr);
	//communicate the iface_addr and ep_addr to corresponding process
	MPI_Send(iface_addr, iface_attr.iface_addr_len, MPI_BYTE, partner, 0, MPI_COMM_WORLD);
	MPI_Recv(iface_addr, iface_attr.iface_addr_len, MPI_BYTE, partner, 0, MPI_COMM_WORLD, &mpi_status);
	MPI_Send(ep_addr, iface_attr.ep_addr_len, MPI_BYTE, partner, 0, MPI_COMM_WORLD);
	MPI_Recv(ep_addr, iface_attr.ep_addr_len, MPI_BYTE, partner, 0, MPI_COMM_WORLD, &mpi_status);

	fprintf(stdout,"Before EP connect !!!\n");fflush(stderr);
	if(iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP){
		status = uct_ep_connect_to_ep(ep, ep_addr);
	}else if(iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE){
		status = uct_ep_create_connected(iface, iface_addr, &ep);
	}else status = UCS_ERR_UNSUPPORTED;
	if(status != UCS_OK){
		fprintf(stderr, "Failed to connect endpoint\n");
	}
	fprintf(stdout,"After EP connect: %d\n", rank);fflush(stderr);


	uint8_t id = 0;
	status = uct_iface_set_am_handler(iface, id, hello_world, arg);
	if(status != UCS_OK){
		fprintf(stderr, "Failed to set callback.\n");
		goto out_free_ep;
	}	 	
	
	if(0 == rank){
		uint64_t header;
		char payload[8];
		unsigned length = sizeof(payload);
		fprintf(stderr,"Sending AM\n");fflush(stderr);
		status = uct_ep_am_short(ep, id, header, payload, length);  		
		fprintf(stderr, "Message Sent status=%d\n", status);
	}else if(1 == rank){
		fprintf(stderr, "Beginning to call uct_worker_progress\n");
		while(1){ 
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
