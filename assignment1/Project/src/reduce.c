#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main( int argc, char **argv )
{
  int numtasks, rank, rc;
  rc = MPI_Init( &argc, &argv );
  if (rc != MPI_SUCCESS) {
    printf( "Error starting MPI.\n" );
    MPI_Abort( MPI_COMM_WORLD, rc );
  }

  MPI_Comm_size( MPI_COMM_WORLD, &numtasks );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  printf("Number of tasks = %d\nMy rank = %d\n", numtasks, rank );

  double bob = 1.3;
  double robby = 0.4;
  double reduc;

  if (rank == 0) {
    printf("sleeping\n");
    sleep( 5 );
    printf("done sleeping\n");

    MPI_Allreduce( &bob, &reduc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD ); 

    printf("Task 0: First reduction = %f\n", reduc);

    MPI_Allreduce( &robby, &reduc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD ); 
    printf("Task 0: Second reduction = %f\n", reduc);
  } else {
    printf("Task %d reducing first...\n", rank);
    MPI_Allreduce( &bob, &reduc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD ); 
    printf("Task %d done reducing first, result = %f\n", rank, reduc);

    printf("Task %d reducing second...\n", rank);
    MPI_Allreduce( &robby, &reduc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD ); 
    printf("Task %d done reducing second, result = %f\n", rank, reduc);
  }

  MPI_Finalize();
}
