//Edited by Robert Relyea
//11/4/2016 -- Implementation of Static Strip Allocation
//This file contains the code that the master process will execute.

#include <iostream>
#include <mpi.h>
#include <math.h>

#include "RayTrace.h"
#include "slave.h"


void slaveMain(ConfigData* data)
{
    //Depending on the partitioning scheme, different things will happen.
    //You should have a different function for each of the required 
    //schemes that returns some values that you need to handle.
    switch (data->partitioningMode)
    {
        case PART_MODE_NONE:
            //The slave will do nothing since this means sequential operation.
            break;
        case PART_MODE_STATIC_STRIPS_VERTICAL:
            //The slave will render pixels according to static cyclic horizontal assignment
            slaveStripsV(data);
            break;
        case PART_MODE_STATIC_CYCLES_VERTICAL:
            slaveCyclesV(data);
            break;
        case PART_MODE_STATIC_BLOCKS:
            slaveBlocks(data);
        case PART_MODE_DYNAMIC:
            slaveDynamic(data);
            break;
        default:
            std::cout << "This mode (" << data->partitioningMode;
            std::cout << ") is not currently implemented." << std::endl;
            break;
    }
}

void slaveCyclesV(ConfigData* data){
    //Start the computation time timer.
    double computationStart = MPI_Wtime();

    int my_cols = 0; // Number of columns this process will render.
    int bundle_column = 0; // The column inside our bundle.

    // Calculate number of columns this process will render
    for(int cycle = data->mpi_rank * data->cycleSize; cycle < data->width; cycle += data->cycleSize * data->mpi_procs){
        for(int column = cycle; (column - cycle < data->cycleSize) && (column < data->width); column++){
            my_cols++;
        }
    }

    // Allocate memory for our columns
    float* my_pixels = new float[3 * data->height * my_cols];

    for(int cycle = data->mpi_rank * data->cycleSize; cycle < data->width; cycle += data->cycleSize * data->mpi_procs){
        for(int column = cycle; (column - cycle < data->cycleSize) && (column < data->width); column++){
            for(int row = 0; row < data->height; row++){
                //Calculate the index into the array.
                int baseIndex = 3 * ( bundle_column * data->width + row );
                shadePixel(&(my_pixels[baseIndex]),row,column,data);   
            }
            bundle_column++;
        }
    }

    //Stop the comp. timer
    double computationStop = MPI_Wtime();
    double computationTime = computationStop - computationStart;

    MPI_Send(my_pixels, 3 * data->width * my_cols, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(&computationTime, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

    delete[] my_pixels;

}



void slaveStripsV(ConfigData* data){

    //Start the computation time timer.
    double computationStart = MPI_Wtime();

    int my_cols = (data->width)/(data->mpi_procs);

    if(data->mpi_rank < (data->width % data->mpi_procs)){ // Implies remainder, 
        my_cols++;                 // Increment to eliminate remainder
    }


    int start_col = data->mpi_rank * my_cols - 1; // Starting column in overall picture
    if(data->width % data->mpi_procs){ // if remainder
        if(data->mpi_rank - (data->width % data->mpi_procs) < 0){ // if one of the procs with remainder added
            start_col++;
        }
        else{ // Not one of the procs with remainder added
            start_col += (data->width % data->mpi_procs) + 1;
        }
    }


    float* my_pixels = new float[3 * data->height * my_cols];

     //Render the scene.
    int my_column = 0;
    for(int column = start_col; column < start_col + my_cols; column++){ // Each assigned col
        
        for(int row = 0; row < data->height; row++){           // Each pixel in col

            //Calculate the index into the array.
            int baseIndex = 3 * ( row * my_cols + my_column);
            shadePixel(&(my_pixels[baseIndex]),row,column,data);   // Store RGB info for pixel
        }
        my_column++;
    }
    //Stop the comp. timer
    double computationStop = MPI_Wtime();
    double computationTime = computationStop - computationStart;

    MPI_Send(my_pixels, 3 * data->width * my_cols, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(&computationTime, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);


    delete[] my_pixels;


    

}


void slaveBlocks(ConfigData* data){
    //Start the computation time timer.
    double computationStart = MPI_Wtime();

    MPI_Status status;

    int my_x = data->width / ((int)sqrt(data->mpi_procs));
    int my_y = data->height / ((int)sqrt(data->mpi_procs));

    int x_remainder = data->width % ((int)sqrt(data->mpi_procs));
    int y_remainder = data->height % ((int)sqrt(data->mpi_procs));

    int start_x = (data->mpi_rank % ((int)sqrt(data->mpi_procs))) * my_x;
    int start_y = (data->mpi_rank / ((int)sqrt(data->mpi_procs))) * my_y;

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

    float* my_pixels = new float[3 * my_x * my_y];


    std::cout << data->mpi_rank << ":\tstart_x: " << start_x << "\tend_x: " << end_x << "\tstart_y: " << start_y <<  "\tend_y: " << end_y << std::endl;

    //Render the scene.
    for(int column = start_y; column < end_y; column++){ // Each assigned col
        for(int row = start_x; row < end_x; row++){           // Each pixel in col

            //Calculate the index into the array.
            int baseIndex = 3 * ( (row - start_x) * my_x + (column - start_y) );

            shadePixel(&(my_pixels[baseIndex]),row,column,data);   // Store RGB info for pixel
        }
    }


    // Send info to master

    MPI_Send(my_pixels, 3 * my_x * my_y, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

    delete[] my_pixels;

}

void slaveDynamic( ConfigData *data ){
    int working = 1;
    int start_col = 0;
    int start_row = 0;
    int width = 0;
    int height = 0;

    while(working > 0){ // Work until there is nothing else to work on
        dRequestWork(data);

        // Receive work info
        dParseWorkInfo( data , &start_col, &start_row, &width, &height);

        if(start_col < 0){ // No more work
            working = -1;
        }else{
            // Allocate pixels for rendering
            float* my_pixels = new float[3 * width * height];

            // Iterate through all pixels in work unit
            for(int row = start_row; row < start_row + height; row++){
                for(int col = start_col; col < start_col + width; col++){
                    // Calculate index in my_pixels
                    int baseIndex = 3 * ( (row - start_row) * width + (col - start_col) );

                    shadePixel(&(my_pixels[baseIndex]),row,col,data);   // Store RGB info for pixel
                    
                }
            }

            // Send the work info to the master so they can allocate memory for pixels
            dSendWorkInfo(data, start_col, start_row, width, height);

            // Send the pixels to the master
             MPI_Send(my_pixels, 3 * width * height, MPI_FLOAT, 0, SLAVE_WORK_DATA, MPI_COMM_WORLD);

            // Free up allocated memory
            delete[] my_pixels;
        }

    }
    
}

void dRequestWork( ConfigData *data ){
    int blank[4];
    MPI_Send(&blank, 4, MPI_INT, 0, SLAVE_WORK_REQUEST, MPI_COMM_WORLD);
}

void dParseWorkInfo( ConfigData *data, int* start_col, int* start_row, int* width, int* height){
    MPI_Status status;
    // Buffer for work info
    // {start_col, start_row, width, height}
    int work_info[4];
    MPI_Recv(&work_info, 4, MPI_INT, 0, MASTER_WORK_INFO, MPI_COMM_WORLD, &status);

    *start_col = work_info[0];
    *start_row = work_info[1];
    *width = work_info[2];
    *height = work_info[3];
}

void dSendWorkInfo( ConfigData *data, int start_col, int start_row, int width, int height){
    // Buffer for work info
    // {start_col, start_row, width, height}
    int work_info[4] = {start_col, start_row, width, height};
    MPI_Send(&work_info, 4, MPI_INT, 0, SLAVE_WORK_FINISHED, MPI_COMM_WORLD);
}