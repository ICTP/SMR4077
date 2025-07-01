## Introduction to MPI Programming in C

This is an introductory lecture to MPI (Message Passing Interface) programming using C. We will start from the absolute basics.

### What is MPI?

`MPI` stands for Message Passing Interface and is fundamentally a specification/standard that defines a set of functions, data types, and communication protocols for parallel programming. The MPI standard itself is not executable code - it's just a detailed description of how message passing operations should behave, what parameters they should accept, and what results they should produce. MPI implementations are the actual software libraries that translate these standardized function calls into real network communications and system operations.

Most common open source MPI implementations are [MPICH](https://www.mpich.org/) and [OpenMPI](https://www.open-mpi.org/).

---

### Hello, World!

Let's look at a simple "Hello World" MPI program:

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    printf("Hello world from rank %d out of %d processes\n", world_rank, world_size);

    MPI_Finalize();
    return 0;
}
```

---

#### Explanation 

- `MPI_Init`: 

This function initializes the MPI environment. It **must** be called before any other MPI functions. It takes command-line arguments `argc` and `argv`, passing them to all processes. *From MPI standard 2, it's actually safe to initialize with `NULL`, `NULL` if you don't need them as by the standard implementations cannot rely on the command line arguments anymore* (however, you never know if you run into some special implementation, so it's a good practice to still pass them).


- `MPI_Finalize`: This function cleans up the MPI environment. It **must** be called after all MPI communication is finished, and no MPI calls can be made after it. Bad things might happen if you don't call it. 


These are two functions that **must** be present in any MPI program.

- `MPI_Comm_size`: 

The total number of the processes that the program runs with. We usually need to know it from inside the program, so the function `MPI_Comm_size` provides us this number. It expects a pointer, so you need to pass the variable with an `&`.

The first variable here, `MPI_COMM_WORLD` is the communicator that includes all processes. A communicator is a group of processes that take part in some communication. `MPI_COMM_WORLD` is the "default" one, that includes all the processes. We will not see other communicators during this school.


- `MPI_Comm_rank`: Within a given communicator, each process gets a unique identifier - its "rank" and the processes will do different things only based on this rank. 
**You will almost always have to use the "world size" and "rank" in your programs.**


**Note:** All processes run the **same** code! (but encounter conditions based on their rank)



### Compiling and Running

To compile an MPI program in C, use `mpicc`: (first make sure it's available, load a module if needed, we can use `module load openmpi` on Leonardo)

```bash
mpicc hello.c -o hello.x
```

To run it with, say, 4 processes:

```bash
mpirun -np 4 ./hello.x
```

You can also use `mpiexec` instead of `mpirun` or `srun` in slurm.


Note1: typically the number you give to it should not be bigger than the number of cores you have (otherwise performace deteriorates), but for testing purposes on a laptop you can use more. With openmpi implementaion you need to explicitly pass --oversubscribe flag, with some implimentations it will be done implicitly.

Note2: you don't *really* need to submit a job to run a "hello world" program. It's ok to run things on login node if they don't take too much time or resources (use "common sense").


---

Output example when run with 4 processes:

```bash
Hello world from rank 0 out of 4 processes
Hello world from rank 1 out of 4 processes
Hello world from rank 2 out of 4 processes
Hello world from rank 3 out of 4 processes
```



---

### Communication

The essence of MPI lies in communication between processes (that is, sending and receiving various messages). Remember that each of them has it's own memory space and cannot access other's memory - that's why they need to communicate. 

Let's start with basics: blocking point to point communication


## MPI_Send and MPI_Recv

```c
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
```

 - `buf`: Pointer to the buffer where the data to be sent is stored.
 - `count`: Number of elements in the buffer to be sent.
 -  `datatype`: The type of data being sent. *MPI has its own datatypes.* Common `MPI_Datatype` values are:
    - `MPI_INT`: Integer data type.
    - `MPI_DOUBLE`: Double-precision floating-point data type.
    - `MPI_FLOAT`: Single-precision floating-point data type.
    - `MPI_CHAR`: Character data type.
    - `MPI_BYTE`: Allows you to send raw bytes, useful for arbitrary data.
    - the rest you can google when you need them. It is also possible to create your own datatypes, we will see that later
 - `dest`: Rank of the destination process in the communicator.
 - `tag`: Message tag (an integer that can be used to identify different messages). Sometimes your algorithms require using them, sometimes you can just out "0" everywhere.
 - `comm`: Communicator, typically MPI_COMM_WORLD (but can be something created by user).

The `MPI_Recv` is similar:

```c
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
``` 
  - `buf`: Pointer to the buffer where the received data will be stored.
  - `count`: Number of elements expected in the buf.
  - `datatype`: The type of data being received. 
  - `source`: Rank of the source process from which to receive the message.
  - `tag`: **Message tag used to match the sender and receiver.**
  - `comm`: Communicator
  - `status`: A pointer to an MPI_Status object that provides information about the received message (e.g., the source and tag). Very often, you can use `MPI_STATUS_IGNORE`.


  
### Example: Sending an Integer

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int data;
    if (rank == 0) {
        data = 123;
        MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1 received data: %d\n", data);
    }

    MPI_Finalize();
    return 0;
}
```


---

### Dangers of MPI - Deadlocks 

**Deadlock:** MPI programs can easily run into deadlocks (when nothing is happening and the program is not doing anything anymore) if two processes are both waiting to receive data but no one is sending. It is a very annoying bug "in real life"...

Example of dangerous code:

```c
#include <stdio.h>
#include <stdlib.h>

//BAD CODE EXAMPLE!!!
//DO NOT COPY ON ACCIDENT INTRO OTHER PLACES!!!
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int size = 100;
    int* vec = (int*)malloc(size * sizeof(int));
    int* vec2 = (int*)malloc(size * sizeof(int));

    if (world_rank == 0) {
        // Process 0 tries to receive first, then send
        MPI_Recv(vec2, size, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(vec, size, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if (world_rank == 1) {
        // Process 1 also tries to receive first, then send
        MPI_Recv(vec2, size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(vec, size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    free(vec);
    free(vec2);

    MPI_Finalize();
    return 0;
}
```

Both processes can wait forever! Always plan who sends and who receives first.

Now let's look at something that is even more dangerous:

```c

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

//BAD CODE EXAMPLE!!!
//DO NOT COPY ON ACCIDENT INTRO OTHER PLACES!!!
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int size = 100; 
    int* vec = (int*)malloc(size * sizeof(int));
    int* vec2 = (int*)malloc(size * sizeof(int));

    if (world_rank == 0) {
        // Process 0 sends first, then receives
        MPI_Send(vec, size, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(vec2, size, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (world_rank == 1) {
        // Process 1 also sends first, then receives
        MPI_Send(vec, size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(vec2, size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    free(vec);
    free(vec2);

    MPI_Finalize();
    return 0;
}

```

Surprise! It works. Why? Because there is an internal buffer and the MPI actually sends the message immediately instead of waiting if the message fits into that buffer. If we change `100` to, say, `5000` it won't work anymore. So this bug is more dangerous because it might look like everything is working on small testing examples, but it will hang when you run "for real". 

---

How do we avoid deadlocks? For example in this case we can do:

### MPI_Sendrecv

Safe way to send and receive at the same time.

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int send_data = rank;
    int recv_data = -1;
    if (rank == 0) {
        MPI_Sendrecv(&send_data, 1, MPI_INT, 1, 0, &recv_data, 1, MPI_INT, 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank == 1) {
        MPI_Sendrecv(&send_data, 1, MPI_INT, 0, 0, &recv_data, 1, MPI_INT, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    printf("Process %d received %d\n", rank, recv_data);

    MPI_Finalize();
    return 0;
}
```

So, here we've encountered a new function - `MPI_Sendrecv`. Let's look at it's [documentation](https://www.mpich.org/static/docs/v4.3.0/) together.



---

Another way to not have deadlocks like the above ones is to have "non-blocking" communication (not that it's any easier to write, you'll just face different types of bugs, but there might be advantages with overlapping communication and computation).

## Non-Blocking Communication

You can send/receive without waiting!

### MPI_Isend and MPI_Irecv

- Start send/receive.
- Later, use `MPI_Wait` to complete it.

`MPI_Isend` and `MPI_Irecv` functions' signatures are very similar to their blocking counterparts, so I'm not copying the explanations here. The only difference is:

- `request`: a "handle" to an `MPI_Request` object, which is used to track the status of the non-blocking operation.

Example:

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int send_data = rank;
    int recv_data = -1;

    MPI_Request req_send, req_recv;

    if (rank == 0) {
        MPI_Isend(&send_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &req_send);
        MPI_Irecv(&recv_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &req_recv);
    } else if (rank == 1) {
        MPI_Isend(&send_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &req_send);
        MPI_Irecv(&recv_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &req_recv);
    }

    //HERE WE CAN POTENTIALLY DO OTHER THINGS
    
    MPI_Wait(&req_send, MPI_STATUS_IGNORE);
    MPI_Wait(&req_recv, MPI_STATUS_IGNORE);

    printf("Process %d received %d\n", rank, recv_data);

    MPI_Finalize();
    return 0;
}
```

Let's send something bigger than one integer and measure the times to prove that the non-blocking function calls return immediately


```c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const size_t num_doubles = 100 * 1024 * 1024; 
    double* send_data = (double*)malloc(num_doubles * sizeof(double));
    double* recv_data = (double*)malloc(num_doubles * sizeof(double));

    // Initialize send data
    for (size_t i = 0; i < num_doubles; i++) {
        send_data[i] = (double)rank + i * 0.000001;
    }

    MPI_Request req_send, req_recv;
    double t_start_isendrecv, t_end_isendrecv;
    double t_start_wait, t_end_wait;

    // Start non-blocking communication
    t_start_isendrecv = MPI_Wtime();
    if (rank == 0) {
        MPI_Isend(send_data, num_doubles, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &req_send);
        MPI_Irecv(recv_data, num_doubles, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &req_recv);
    } else if (rank == 1) {
        MPI_Isend(send_data, num_doubles, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &req_send);
        MPI_Irecv(recv_data, num_doubles, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &req_recv);
    }
    t_end_isendrecv = MPI_Wtime();

    // Wait for communication to complete
    t_start_wait = MPI_Wtime();
    MPI_Wait(&req_send, MPI_STATUS_IGNORE);
    MPI_Wait(&req_recv, MPI_STATUS_IGNORE);
    t_end_wait = MPI_Wtime();


    printf("Process %d: MPI_Isend/Irecv took %.6f seconds\n", rank, t_end_isendrecv - t_start_isendrecv);
    printf("Process %d: MPI_Wait took %.6f seconds\n", rank, t_end_wait - t_start_wait);

    free(send_data);
    free(recv_data);

    MPI_Finalize();
    return 0;
}

```

---

### Collective Communication


Collective communication functions involve by *all* processes in a communicator. The simplest one is 

 `MPI_Barrier`  this makes the processes wait until **all** of them reach this point.

```c
MPI_Barrier(MPI_COMM_WORLD);
```

#### broadcast

`MPI_Bcast` is used to broadcast data from one process to all other processes in the communicator.

```c
int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
```
  - `buffer`: The data to be broadcast. On the root process, this contains the data to be sent; on other processes, it is the buffer where the data will be received.
  - `count`: Number of elements in the buffer.
  - `datatype`: The datatype of the elements in the buffer (e.g., MPI_INT, MPI_DOUBLE).
  - `root`: The rank of the root process (the one that sends the data).
  - `comm`: The communicator over which the broadcast occurs (e.g., MPI_COMM_WORLD).

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int data;
    if (world_rank == 0) {
        // Root process sets the data
        data = 100;
        printf("Process 0 broadcasting data: %d\n", data);
    }

    // Broadcast the data from process 0 to all processes
    MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // All processes now have the data
    printf("Process %d has data: %d\n", world_rank, data);

    MPI_Finalize();
    return 0;
}

```

#### Gather

```c
int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
```
  - `sendbuf`: The starting address of the send buffer (the data being sent from each process).
  - `sendcount`: The number of elements sent from each process.
  - `sendtype`: The datatype of the elements being sent
  - `recvbuf`: The starting address of the receive buffer (where the gathered data will be stored on the root process).
  - `recvcount`: The number of elements to receive from each process (this must be the same for all processes).
  - `recvtype`: The datatype of the elements being received.
  - `root`: The rank of the root process.
 
```c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int send_data = world_rank;

    // Only the root process allocates the receive buffer
    int* recv_data = NULL;
    if (world_rank == 0) {
        recv_data = (int*)malloc(world_size * sizeof(int));
    }

    // Gather data from all processes to the root
    MPI_Gather(&send_data, 1, MPI_INT, recv_data, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Root process prints the gathered data
    if (world_rank == 0) {
        printf("Process 0 gathered data: ");
        for (int i = 0; i < world_size; ++i) {
            printf("%d ", recv_data[i]);
        }
        printf("\n");
        free(recv_data);
    }

    MPI_Finalize();
    return 0;
}

```
---

#### MPI_Reduce

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int local_value = world_rank;

    int global_sum = 0;

    // Reduce all local values by summing them to root process (rank 0)
    MPI_Reduce(&local_value, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Root process prints the result
    if (world_rank == 0) {
        printf("The sum of all ranks is: %d\n", global_sum);
    }

    MPI_Finalize();
    return 0;
}

```


---
It's easy to write bugs with collective calls, for example by calling them under some conditional operators. This is actually "undefined behavior", it might hang or it might give wrong results.

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int data;
    if (world_rank == 0) {
        // Root process sets the data
        data = 100;
        printf("Process 0 broadcasting data: %d\n", data);
    }

    // DO NOT DO THIS
    if (world_rank == 0) {
        MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    // All processes now have the data
    printf("Process %d has data: %d\n", world_rank, data);

    MPI_Finalize();
    return 0;
}


```

---

### Extra topics:

#### MPI_PROC_NULL

You can send to (or receive from) "nothing". In this case the operation just won't happen, but you can write more general code and avoid extra "ifs". You will see a useful example in the next course about mpi, but so far we want to see how it works and remember it exists.

This "nothing" is called `MPI_PROC_NULL` and can be used as the source or destination in functions.


```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int send_data = rank;
    int recv_data = 777;

    // Define neighbors without periodic boundary conditions
    int next_rank = (rank == world_size - 1) ? MPI_PROC_NULL : rank + 1;
    int prev_rank = (rank == 0) ? MPI_PROC_NULL : rank - 1;

    MPI_Sendrecv(&send_data, 1, MPI_INT, next_rank, 0,
                 &recv_data, 1, MPI_INT, prev_rank, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("Process %d has recv_data %d\n", rank, recv_data);

    MPI_Finalize();
    return 0;
}

}
```


#### MPI_Datatype

Users can create their own "datatypes" by combining some elements of "basic types". There are a few functions for that, but lets consider just one of them - `MPI_Type_vector`.

It is used to define a datatype for regularly spaced blocks of data. This is useful when you want to send non-contiguous data from memory, such as sending every N-th element of an array (a strided array) without copying the data to a temporary contiguous buffer.

here is the "signature":

```c
int MPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype, MPI_Datatype *newtype);
```

 - `count`: Number of blocks.
 - `blocklength`: Number of elements in each block.
 - `stride`: Number of elements between the start of each block.
 - `oldtype`: The type of elements in the original array (e.g., MPI_INT, MPI_DOUBLE).
 - `newtype`: The new datatype that MPI will create based on the vector pattern.

After defining the new type with MPI_Type_vector, you need to commit it using MPI_Type_commit() so that it can be used in communication functions.


Let's send every 2nd element of an array:

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    const int array_size = 10;
    int data[array_size];

    if (rank == 0) {
        // Initialize data with 0 to 9
        for (int i = 0; i < array_size; i++) {
            data[i] = i;
        }

        // Define vector type: send every second element (i.e., elements at indices 0, 2, 4, 6, 8)
        MPI_Datatype vector_type;
        MPI_Type_vector(5, 1, 2, MPI_INT, &vector_type);
        MPI_Type_commit(&vector_type);

        // Send using custom datatype
        MPI_Send(data, 1, vector_type, 1, 0, MPI_COMM_WORLD);

        MPI_Type_free(&vector_type);
    } else if (rank == 1) {
        // Initialize to sentinel value
        for (int i = 0; i < array_size; i++) {
            data[i] = -1;
        }

        // Receive 5 elements normally into the first 5 positions of `data`
        MPI_Recv(data, 5, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Print the result
        printf("Process 1 received data: ");
        for (int i = 0; i < array_size; i++) {
            printf("%d ", data[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}

```


---


### EXERCISES:

### Session 1:

- Exercise 0: run the examples in the lecture and make sure you understand them.

- Exercise 1:
  
   Estimate π using the integral:

    $$
     \pi \approx 4 \int_0^1 \frac{1}{1 + x^2} \, dx
    $$

Split the interval `[0,1]` into `n = 1,000,000,000` subintervals, and compute the sum in parallel using `MPI_Reduce`.
Once it's working, try to run the scaling of it on Leonardo.
    
    
**If you are done and want to do extra:**

 - Exercise 2: find the internal buffer size of MPI_Send of your MPI implementation.

 - Exercise 3: Write "hello world" type programs (minimal examples that explain what they do) for the following functions:

   - `MPI_Scatter`
   - `MPI_Allgather`
   - `MPI_Allreduce`

- Exercise 4: Using `MPI_Type_vector`, exchange every 3rd element of vectors between 2 processes.


### Session 2: (more detailed explanations on "blackboard")

Write an mpi program that initializes and prints identity matrix. 

You have to

- pretend that your matrix doesn't fit into the memory of one process
- calculate the number of rows that each process gets and allocate the memory accordingly (use `calloc`)
- figure out the index correspondence between a "global" and "local" part to know where to put the ones
- send parts to process 0 to print them in order (be careful with the sizes here) 







