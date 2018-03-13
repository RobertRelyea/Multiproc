//Edited by Robert Relyea
//11/4/2016 -- Implementation of Static Strip Allocation
//This file contains the code that the master process will execute.

#include <iostream>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>

#include "RayTrace.h"
#include "master.h"

void masterMain(ConfigData* data)
{
    //Depending on the partitioning scheme, different things will happen.
    //You should have a different function for each of the required 
    //schemes that returns some values that you need to handle.
    
    //Allocate space for the image on the master.
    float* pixels = new float[3 * data->width * data->height];
    
    //Execution time will be defined as how long it takes
    //for the given function to execute based on partitioning
    //type.
    double renderTime = 0.0, startTime, stopTime;

	//Add the required partitioning methods here in the case statement.
	//You do not need to handle all cases; the default will catch any
	//statements that are not specified. This switch/case statement is the
	//only place that you should be adding code in this function. Make sure
	//that you update the header files with the new functions that will be
	//called.
	//It is suggested that you use the same parameters to your functions as shown
	//in the sequential example below.
    switch (data->partitioningMode)
    {
        case PART_MODE_NONE:
            //Call the function that will handle this.
            startTime = MPI_Wtime();
            masterSequential(data, pixels);
            stopTime = MPI_Wtime();
            break;
        case PART_MODE_STATIC_STRIPS_VERTICAL:
            startTime = MPI_Wtime();
            masterStripsV(data, pixels);
            stopTime = MPI_Wtime();
            break;
        case PART_MODE_STATIC_CYCLES_VERTICAL:
            startTime = MPI_Wtime();
            masterCyclesV(data, pixels);
            stopTime = MPI_Wtime();
            break;
        case PART_MODE_STATIC_BLOCKS:
            startTime = MPI_Wtime();
            masterBlocks(data, pixels);
            stopTime = MPI_Wtime();
            break;
        case PART_MODE_DYNAMIC:
        	startTime = MPI_Wtime();
            masterDynamic(data, pixels);
            stopTime = MPI_Wtime();
            break;
        default:
            std::cout << "This mode (" << data->partitioningMode;
            std::cout << ") is not currently implemented." << std::endl;
            break;
    }

    renderTime = stopTime - startTime;
    std::cout << "Execution Time: " << renderTime << " seconds" << std::endl << std::endl;

    //After this gets done, save the image.
    std::cout << "Image will be save to: ";
    std::string file = generateFileName(data);
    std::cout << file << std::endl;
    savePixels(file, pixels, data);

    //Delete the pixel data.
    delete[] pixels; 
}

void masterSequential(ConfigData* data, float* pixels)
{
    //Start the computation time timer.
    double computationStart = MPI_Wtime();

    //Render the scene.
    for( int i = 0; i < data->height; ++i )
    {
        for( int j = 0; j < data->width; ++j )
        {
            int row = i;
            int column = j;

            //Calculate the index into the array.
            int baseIndex = 3 * ( row * data->width + column );

            //Call the function to shade the pixel.
            shadePixel(&(pixels[baseIndex]),row,j,data);
        }
    }

    //Stop the comp. timer
    double computationStop = MPI_Wtime();
    double computationTime = computationStop - computationStart;

    //After receiving from all processes, the communication time will
    //be obtained.
    double communicationTime = 0.0;

    //Print the times and the c-to-c ratio
	//This section of printing, IN THIS ORDER, needs to be included in all of the
	//functions that you write at the end of the function.
    std::cout << "Total Computation Time: " << computationTime << " seconds" << std::endl;
    std::cout << "Total Communication Time: " << communicationTime << " seconds" << std::endl;
    double c2cRatio = communicationTime / computationTime;
    std::cout << "C-to-C Ratio: " << c2cRatio << std::endl;
}

void masterCyclesV(ConfigData* data, float* pixels)
{
    MPI_Status status;

    //Start the computation time timer.
    double computationStart = MPI_Wtime();

    for(int cycle = data->mpi_rank * data->cycleSize; cycle < data->width; cycle += data->cycleSize * data->mpi_procs){
        for(int column = cycle; (column - cycle < data->cycleSize) && (column < data->width); column++){
            for(int row = 0; row < data->height; row++){
                //Calculate the index into the array.
                int baseIndex = 3 * ( row * data->width + column);
                shadePixel(&(pixels[baseIndex]),row,column,data);   
            }
        }
    }

    //Stop the comp. timer
    double computationStop = MPI_Wtime();
    double computationTime = computationStop - computationStart;









    // Start the comm. timer
    double communicationStart = MPI_Wtime();


    for(int i = 1; i < data->mpi_procs; i++){// Gather data from each processor
	    //Allocate buffer for receiving data

	    int recv_cols = 0;

        for(int cycle = i * data->cycleSize; cycle < data->width; cycle += data->cycleSize * data->mpi_procs){
            for(int column = cycle; (column - cycle < data->cycleSize) && (column < data->width); column++){
                recv_cols++;
            }
        }

	    int recv_buf_size = 3 * recv_cols * data->height;

	    float* recv_buf = new float[recv_buf_size];
        double comm_recv_buf = 0.0;

        MPI_Recv(recv_buf, recv_buf_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status); // Update Tag to be nicer
        MPI_Recv(&comm_recv_buf, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status); // Get computation time from each processor.

        if(comm_recv_buf > computationTime){ // Get largest computation time.
            computationTime = comm_recv_buf; // The maximum comp time is the overall comp time.
        }

        int bundle_column = 0; // The column we are currently looking at in the bundle

        for(int cycle = i * data->cycleSize; cycle < data->width; cycle += data->cycleSize * data->mpi_procs){
            for(int column = cycle; (column - cycle < data->cycleSize) && (column < data->width); column++){
                for(int row = 0; row < data->height; row++){
                    //Calculate the index into the array.
                    int bundleIndex = 3 * ( bundle_column * data->height + row );
                    int baseIndex = 3 * ( row * data->width + column);

                    //Copy Pixel
                    pixels[baseIndex] = recv_buf[bundleIndex];
                    pixels[baseIndex + 1] = recv_buf[bundleIndex + 1];
                    pixels[baseIndex + 2] = recv_buf[bundleIndex + 2];
                }
                bundle_column++;
            }
        }

        delete[] recv_buf;
	}

    //After receiving from all processes, the communication time will
    //be obtained.
    double communicationStop = MPI_Wtime();
    double communicationTime = communicationStop - communicationStart;

    //Print the times and the c-to-c ratio
    //This section of printing, IN THIS ORDER, needs to be included in all of the
    //functions that you write at the end of the function.
    std::cout << "Total Computation Time: " << computationTime << " seconds" << std::endl;
    std::cout << "Total Communication Time: " << communicationTime << " seconds" << std::endl;
    double c2cRatio = communicationTime / computationTime;
    std::cout << "C-to-C Ratio: " << c2cRatio << std::endl;
}

void masterStripsV(ConfigData* data, float* pixels)
{
    //Start the computation time timer.
    double computationStart = MPI_Wtime();

    MPI_Status status;

    int my_cols = (data->width)/(data->mpi_procs);

    if(data->mpi_rank < (data->width % data->mpi_procs)){ // Implies remainder, 
        my_cols++;                 // Increment to eliminate remainder
    }

    //Render the scene.
    for(int column = 0; column < my_cols; column++){ // Each assigned col
        for(int row = 0; row < data->width; row++){           // Each pixel in col

            //Calculate the index into the array.
            int baseIndex = 3 * ( row * data->width + column );

            shadePixel(&(pixels[baseIndex]),row,column,data);   // Store RGB info for pixel
        }
    }



    //Stop the comp. timer
    double computationStop = MPI_Wtime();
    double computationTime = computationStop - computationStart;

    // Start the comm. timer
    double communicationStart = MPI_Wtime();

    int current_col = my_cols;

    for(int i = 1; i < data->mpi_procs; i++){// Gather data from each processor

        //Allocate buffer for receiving data
        int recv_cols = (data->width)/(data->mpi_procs);



        if(i < (data->width % data->mpi_procs)){ // Implies remainder, 
            recv_cols++;                 // Increment to eliminate remainder
        }

        int recv_buf_size = 3 * recv_cols * data->height;

        double comm_recv_buf = 0.0;

        float* recv_buf = new float[recv_buf_size];

        MPI_Recv(recv_buf, recv_buf_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status); // Update Tag to be nicer
        MPI_Recv(&comm_recv_buf, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status); // Get computation time from each processor.

        if(comm_recv_buf > computationTime){ // Get largest computation time.
            computationTime = comm_recv_buf; // The maximum comp time is the overall comp time.
        }

        for(int column = 0; column < recv_cols; column++){ //Vertical Data col index
            for(int row = 0; row < (data->height); row++){ //Vertical Data row index

                int recvIndex = 3 * ( row * recv_cols + column ); // Base index for received image
                int baseIndex = 3 * ( row * data->width + current_col ); // Base index for final image

                //Copy Pixel
                pixels[baseIndex] = recv_buf[recvIndex];
                pixels[baseIndex + 1] = recv_buf[recvIndex + 1];
                pixels[baseIndex + 2] = recv_buf[recvIndex + 2];
            }
            current_col++;

        }

        delete[] recv_buf;
    }



    //After receiving from all processes, the communication time will
    //be obtained.
    double communicationStop = MPI_Wtime();
    double communicationTime = communicationStop - communicationStart;

    //Print the times and the c-to-c ratio
    //This section of printing, IN THIS ORDER, needs to be included in all of the
    //functions that you write at the end of the function.
    std::cout << "Total Computation Time: " << computationTime << " seconds" << std::endl;
    std::cout << "Total Communication Time: " << communicationTime << " seconds" << std::endl;
    double c2cRatio = communicationTime / computationTime;
    std::cout << "C-to-C Ratio: " << c2cRatio << std::endl;
}


void masterBlocks(ConfigData* data, float* pixels)
{
	//Start the computation time timer.
    double computationStart = MPI_Wtime();

	MPI_Status status;

    int my_x = data->width / ((int)sqrt(data->mpi_procs));
    int my_y = data->height / ((int)sqrt(data->mpi_procs));

    int x_remainder = data->width % ((int)sqrt(data->mpi_procs));
    int y_remainder = data->height % ((int)sqrt(data->mpi_procs));

    int start_x = (data->mpi_rank / ((int)sqrt(data->mpi_procs))) * my_x;
    int start_y = (data->mpi_rank % ((int)sqrt(data->mpi_procs))) * my_y;

    //Check y remainder
    if(y_remainder){
        if((data->mpi_rank % ((int)sqrt(data->mpi_procs))) == ((int)sqrt(data->mpi_procs)) - 1){ // Left Edge Proc
            my_y += y_remainder;
        }
    }
    //Check x remainder
    if(x_remainder){
        if(data->mpi_rank / ((int)sqrt(data->mpi_procs)) == ((int)sqrt(data->mpi_procs)) - 1){ // Bottom proc
            my_x += x_remainder;
        }
    }

    int end_x = start_x + my_x;
    int end_y = start_y + my_y;


    
    //Render the scene.
    for(int column = start_y; column < end_y; column++){ // Each assigned col
        for(int row = start_x; row < end_x; row++){           // Each pixel in col

            //Calculate the index into the array.
            int baseIndex = 3 * ( row * data->width + column );

            shadePixel(&(pixels[baseIndex]),row,column,data);   // Store RGB info for pixel
        }
    }

    //Stop the comp. timer
    double computationStop = MPI_Wtime();
    double computationTime = computationStop - computationStart;

    //After receiving from all processes, the communication time will
    //be obtained.
    double communicationTime = 0.0;


    for(int i = 1; i < data->mpi_procs; i++){// Gather data from each processor

        // Calculate how much data we will receive from this process
        int recv_x = data->width / ((int)sqrt(data->mpi_procs));
        int recv_y = data->height / ((int)sqrt(data->mpi_procs));

        int recv_start_x = (i % ((int)sqrt(data->mpi_procs))) * recv_x;
        int recv_start_y = (i / ((int)sqrt(data->mpi_procs))) * recv_y;

        //Check y remainder
        if(y_remainder){
            if(i % ((int)sqrt(data->mpi_procs)) == ((int)sqrt(data->mpi_procs)) - 1){ // Left edge proc
                recv_y += y_remainder;
            }
        }
        //Check x remainder
        if(x_remainder){
            if((i / ((int)sqrt(data->mpi_procs))) == ((int)sqrt(data->mpi_procs)) - 1){ // Bottom proc
                recv_x += x_remainder;
            }
        }

        end_x = recv_start_x + recv_x;
        end_y = recv_start_y + recv_y;


        //Allocate buffer for receiving data

        int recv_buf_size = 3 * recv_x * recv_y;

        float* recv_buf = new float[recv_buf_size];

        MPI_Recv(recv_buf, recv_buf_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status); // Update Tag to be nicer

        for(int column = recv_start_y; column < end_y; column++){ //Vertical Data col index
            for(int row = recv_start_x; row < end_x; row++){ //Vertical Data row index

                int recvIndex = 3 * ( (row - recv_start_x) * recv_x + (column - recv_start_y) ); // Base index for received image
                int baseIndex = 3 * ( row * data->width + column ); // Base index for final image

                //Copy Pixel
                pixels[baseIndex] = recv_buf[recvIndex];
                pixels[baseIndex + 1] = recv_buf[recvIndex + 1];
                pixels[baseIndex + 2] = recv_buf[recvIndex + 2];
            }

        }

        delete[] recv_buf;
    }


    //Print the times and the c-to-c ratio
	//This section of printing, IN THIS ORDER, needs to be included in all of the
	//functions that you write at the end of the function.
    std::cout << "Total Computation Time: " << computationTime << " seconds" << std::endl;
    std::cout << "Total Communication Time: " << communicationTime << " seconds" << std::endl;
    double c2cRatio = communicationTime / computationTime;
    std::cout << "C-to-C Ratio: " << c2cRatio << std::endl;

}

// Initialize central task queue
void dInitQueue(queuePointer queue){
	queue->top = NULL;
	queue->bottom = NULL;
}

// Add a work unit to the queue
void dAddQueue(queuePointer queue, workPointer work){
	// Add the work to the bottom of the queue
	if(queue->bottom == NULL){ // Queue is empty, add the only work node
		queue->top = work;
		queue->bottom = work;
	}else{ // Queue is not empty
		queue->bottom->next = work;
		work->prev = queue->bottom;
		work->next = NULL;
		queue->bottom = work;
	}
}

// Take the next work unit off the queue
void dPopQueue(queuePointer queue, workPointer* work){
	*work = queue->top;
	queue->top = queue->top->next;
}

// See if the queue has any more work
int dQueueEmpty(queuePointer queue){
	return queue->top == NULL;
}





void masterDynamic(ConfigData* data, float* pixels){

	MPI_Status status;

	int slaves[data->mpi_procs]; // Array for keeping track of slave status

	int cur_col = 0; // Our current index in the image.
	int cur_row = 0;

	int slave_flag = 0;
	int done = 0;

	// Initialize task queue
	queuePointer task_queue = (queuePointer)malloc(sizeof(struct centralQueue));
	dInitQueue(task_queue);

	// Enqueue all work units
	while(!done){
		if(dCheckWork(data, &cur_col, &cur_row, data->dynamicBlockWidth, data->dynamicBlockHeight)){
			// Work unit to enqueue
			// Allocate memory for work unit
			workPointer work = (workPointer)malloc(sizeof(struct workNode));
			// Assign parameters for work unit
			work->start_col = cur_col;
			work->start_row = cur_row;
			work->work_width = data->dynamicBlockWidth;
			work->work_height = data->dynamicBlockHeight;
			// Enqueue work unit
			dAddQueue(task_queue, work);
			cur_row += data->dynamicBlockHeight;
		}else{ // All work units enqueued
			done = 1;
		}
	}

	    // Check for leftovers
    int leftovers = dCheckLeftovers(data);
    if(leftovers){
    	int new_width = 0;
    	int new_height = 0;
    	if(leftovers == LEFTOVERS_RIGHT){ // Leftovers on right image border
    		// std::cout << "Leftovers found on right border" << std::endl;

    		// Generate new work unit size for leftover region
    		int new_width = 0;
    		int new_height = 0;
    		dNewWorkUnitRight(data, &new_width, &new_height);

    		// Iterate through leftover region
    		done = 0;
    		cur_col = data->width - (data->width % data->dynamicBlockWidth); // Origin of right leftover region
    		cur_row = 0;
    		while(!done){
    			if(dCheckWork(data, &cur_col, &cur_row, new_width, new_height)){
					// Work unit to enqueue
					// Allocate memory for work unit
					workPointer work = (workPointer)malloc(sizeof(struct workNode));
					// Assign parameters for work unit
					work->start_col = cur_col;
					work->start_row = cur_row;
					work->work_width = new_width;
					work->work_height = new_height;
					// Enqueue work unit
					dAddQueue(task_queue, work);
					cur_row += new_height;
				}else{ // All work units enqueued
					done = 1;
				}
    		}
    	}else if(leftovers == LEFTOVERS_BOTTOM){ // Leftovers on bottom image border

    		// Generate new work unit size for leftover region
    		int new_width = 0;
    		int new_height = 0;
    		dNewWorkUnitBottom(data, &new_width, &new_height);

    		// Iterate through leftover region
    		done = 0;
    		cur_col = 0; // Origin of right leftover region
    		cur_row = data->height - (data->height % data->dynamicBlockHeight);
    		int right_bound = data->width;
    		while(!done){
    			if(dCheckWorkH(data, &cur_col, &cur_row, new_width, new_height, right_bound)){
					// Work unit to enqueue
					// Allocate memory for work unit
					workPointer work = (workPointer)malloc(sizeof(struct workNode));
					// Assign parameters for work unit
					work->start_col = cur_col;
					work->start_row = cur_row;
					work->work_width = new_width;
					work->work_height = new_height;
					// Enqueue work unit
					dAddQueue(task_queue, work);
					cur_col += new_width;
				}else{ // All work units enqueued
					done = 1;
				}
    		}


    	}else{ // Leftovers on both the right and bottom image borders

    		// Generate new work unit size for right leftover region
    		int new_width = 0;
    		int new_height = 0;
    		dNewWorkUnitRight(data, &new_width, &new_height);

    		// Iterate through right leftover region
    		done = 0;
    		cur_col = data->width - (data->width % data->dynamicBlockWidth); // Origin of right leftover region
    		cur_row = 0;
    		while(!done){
    			if(dCheckWork(data, &cur_col, &cur_row, new_width, new_height)){
					// Work unit to enqueue
					// Allocate memory for work unit
					workPointer work = (workPointer)malloc(sizeof(struct workNode));
					// Assign parameters for work unit
					work->start_col = cur_col;
					work->start_row = cur_row;
					work->work_width = new_width;
					work->work_height = new_height;
					// Enqueue work unit
					dAddQueue(task_queue, work);
					cur_row += new_height;
				}else{ // All work units enqueued
					done = 1;
				}
    		}

    		

    		// Iterate through bottom leftover region
    		dNewWorkUnitBottom(data, &new_width, &new_height);
    		done = 0;
    		cur_col = 0; // Origin of right leftover region
    		cur_row = data->height - (data->height % data->dynamicBlockHeight);
    		int right_bound = data->width - (data->width % data->dynamicBlockWidth);
    		while(!done){
    			if(dCheckWorkH(data, &cur_col, &cur_row, new_width, new_height, right_bound)){
					// Work unit to enqueue
					// Allocate memory for work unit
					workPointer work = (workPointer)malloc(sizeof(struct workNode));
					// Assign parameters for work unit
					work->start_col = cur_col;
					work->start_row = cur_row;
					work->work_width = new_width;
					work->work_height = new_height;
					// Enqueue work unit
					// std::cout << "Enqueueing work c: " << work->start_col << "\tr: " << work->start_row << "\t addr: " << work << std::endl;
					dAddQueue(task_queue, work);
					cur_col += new_width;
				}else{ // All work units enqueued
					done = 1;
				}
    		}

    	}
    }


	// Assign all queued work
	workPointer work = NULL;
	while(!dQueueEmpty(task_queue)){
		
		int buff[4] = {0, 0, 0, 0};
		int slave = dProbeSlaves(data, &slave_flag, buff);
        if(slave > 0){
            if(slave_flag == SLAVE_WORK_REQUEST){ // Assign this slave some work off the queue
            	dPopQueue(task_queue, &work); // Grab work from queue
            	dAssignWork(slave, work->start_col, work->start_row, work->work_width, work->work_height, slaves);
            	free(work);
                
            }else if(slave_flag == SLAVE_WORK_FINISHED){ // Slave finished work unit
                dReceiveWork(data, slave, pixels, slaves, buff);
            }
        }// Nothing new with the slaves
		
	}


    // Wait for all remaining working slaves to finish and collect data.
    dFinalize(data, pixels, slaves);
	
    //After receiving from all processes, the communication time will
    //be obtained.
    double communicationTime = 0.0;
    double computationTime = 0.0;

    //Print the times and the c-to-c ratio
	//This section of printing, IN THIS ORDER, needs to be included in all of the
	//functions that you write at the end of the function.
    std::cout << "Total Computation Time: " << computationTime << " seconds" << std::endl;
    std::cout << "Total Communication Time: " << communicationTime << " seconds" << std::endl;
    double c2cRatio = communicationTime / computationTime;
    std::cout << "C-to-C Ratio: " << c2cRatio << std::endl; 
    free(task_queue);
}

void dInitSlaves(ConfigData *data, int* slaves){
	// Iterate through all slave statuses
	for(int i = 1; i < data->mpi_procs; i++){
		slaves[i] = UNEMPLOYED; // Set everyone to unemployed :(
	}
}


int dProbeSlaves(ConfigData *data, int* slave_flag, int* buff){
	MPI_Status status;
	MPI_Recv(buff, 4, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	int slave = (int)status.MPI_SOURCE;
	*slave_flag = (int)status.MPI_TAG;

	if(*slave_flag == SLAVE_WORK_REQUEST){// Slave looking for work
		return slave;
	}else if(*slave_flag == SLAVE_WORK_FINISHED){// Slave done with work
		return slave;
	}else{
		// We've got an issue :(
		std::cout << "Problem with slave #" << slave << std::endl;
	}
	return -1;

}



int dCheckWork(ConfigData *data, int* cur_col, int* cur_row, int width, int height){

	// Check if the next piece of work is within image bounds
	if(*cur_col + width <= data->width){ // Fits in columns
		if(*cur_row + height <= data->height){ // Fits in current line of work.
			return 1; // Good to go!
		}else if(*cur_col + (width * 2) <= data->width ){ // Can we get another line of work?
			*cur_col += width;
			*cur_row = 0;
			return 1; // Apply changes to starting coords now, then good to go!
		}else{ // Cannot fit any more pieces of work
				// do something about the leftovers
				// work is complete
			return 0;
		}
	}else{ // Cannot fit any more pieces of work
		// do something about the leftovers
		// work is complete
		return 0;
	}
}

int dCheckWorkH(ConfigData *data, int* cur_col, int* cur_row, int width, int height, int right_bound){

	// Check if the next piece of work is within image bounds
	if(*cur_col + width <= right_bound){ // Fits in columns
		if(*cur_row + height <= data->height){ // Fits in current line of work.
			return 1; // Good to go!
		}else if(*cur_col + (width * 2) <= data->width ){ // Can we get another line of work?
			*cur_row += height;
			*cur_col = 0;
			return 1; // Apply changes to starting coords now, then good to go!
		}else{ // Cannot fit any more pieces of work
				// do something about the leftovers
				// work is complete
			return 0;
		}
	}else{ // Cannot fit any more pieces of work
		// do something about the leftovers
		// work is complete
		return 0;
	}
}

void dAssignWork(int slave_rank, int start_col, int start_row, int width, int height, int* slaves){
	int work_info[4] = {start_col, start_row, width, height};
	// Let the slave know where to render
	MPI_Send(work_info, 4, MPI_INT, slave_rank, MASTER_WORK_INFO, MPI_COMM_WORLD);

	// Update slave status
	if(start_col >= 0){
		slaves[slave_rank] = WORKING;
	}else{
		slaves[slave_rank] = RETIRED;
	}
	
}

void dReceiveWork(ConfigData *data, int slave_rank, float* pixels, int* slaves, int* buff){
	MPI_Status status;

	int start_col = buff[0];
	int start_row = buff[1];
	int width = buff[2];
	int height = buff[3];

    // Prepare a buffer for the slave's work
	float* work_pixels = new float[3 * width * height];
	// Receive slave's work
	MPI_Recv(work_pixels, 3 * width * height, MPI_FLOAT, slave_rank, SLAVE_WORK_DATA, MPI_COMM_WORLD, &status);
	// Set slave to unemployed
	slaves[slave_rank] = UNEMPLOYED;

	// Iterate through all pixels in the work
	for(int col = start_col; col < start_col + width; col++){
		for(int row = start_row; row < start_row + height; row++){

			int workIndex = 3 * ( (row - start_row) * width + (col - start_col) ); // Base index for received work
            int baseIndex = 3 * ( row * data->width + col ); // Base index for final image

            //Copy Pixel
            pixels[baseIndex] = work_pixels[workIndex];
            pixels[baseIndex + 1] = work_pixels[workIndex + 1];
            pixels[baseIndex + 2] = work_pixels[workIndex + 2];
		}
	}
	delete[] work_pixels;
}

int dCheckLeftovers(ConfigData *data){
	if(data->width % data->dynamicBlockWidth){ // There are leftovers on the right border
		if(data->height % data->dynamicBlockHeight){ // Leftovers also on bottom border
			return LEFTOVERS_BOTH;
		}
		return LEFTOVERS_RIGHT; 
	}else if(data->height % data->dynamicBlockHeight){
		return LEFTOVERS_BOTTOM;
	}
	return 0; // No leftovers
}


void dNewWorkUnitRight(ConfigData *data, int* new_width, int* new_height){
	// Calculate new height
	for(int i = 10; i > 0; i--){
		*new_height = i;
		if(data->height % i == 0){
			break;
		}
	}

	*new_width = data->width % data->dynamicBlockWidth;
}

void dNewWorkUnitBottom(ConfigData *data, int* new_width, int* new_height){

	// Calculate new width
	for(int i = 10; i > 0; i--){
		*new_width = i;
		if((data->width - (data->width % data->dynamicBlockWidth))% i == 0){
			break;
		}
	}

	*new_height = data->height % data->dynamicBlockHeight;
}

void dFinalize(ConfigData *data, float* pixels, int* slaves){
    int finalized = 0;
    int flag = 0;


    int debug_counter = 0;


    MPI_Status status;
    while(!finalized){
    	finalized = 1; // Will be set to zero if an active and working slave is found
        for(int slave = 1; slave < data->mpi_procs; slave++){ // Iterate through all slaves
            if(slaves[slave] != RETIRED){ // If the slave is not retired
    			finalized = 0; // Still have a working slave.
            }
        }
        if(!finalized){
        	// See if the slave has anything to send or receive
	    	int buff[4];
	    	int flag = 0;

	        int slave = dProbeSlaves(data, &flag, buff);

	        if(flag == SLAVE_WORK_REQUEST){         
	                // Tell the slave to stop
	                dAssignWork(slave, -1, -1, data->dynamicBlockWidth, data->dynamicBlockHeight, slaves);
	                slaves[slave] = RETIRED;
	        }else if(flag == SLAVE_WORK_FINISHED){ // Slave finished work unit
	            dReceiveWork(data, slave, pixels, slaves, buff);
	            slaves[slave] = UNEMPLOYED;
	        }else if(flag == MASTER_WORK_INFO){ // Slave is waiting for work info
	        	// Get them to stop
	        	int work_info[4] = {-1, -1, 0, 0};
	        	MPI_Send(work_info, 4, MPI_INT, slave, MASTER_WORK_INFO, MPI_COMM_WORLD);
	        	slaves[slave] = RETIRED;
	        } // Nothing new with the slave
        }
    }
}