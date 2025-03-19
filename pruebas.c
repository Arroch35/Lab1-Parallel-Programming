#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(){
    int num_training_patterns=1934;
    int ranpat[num_training_patterns];
    #pragma omp parallel for num_threads(4)
    for (int p = 0; p < num_training_patterns; p++){
        ranpat[p] = p;
        //printf("1: Soc %d en la iteració (%d)\n", omp_get_thread_num(), p);
        }
    
}