#include<openacc.h>
#include<stdio.h>
#include<time.h>
#include<sys/time.h>
#include<string.h>
#include<stdlib.h>

double seconds(){
    struct timeval tmp;
    double sec;
    gettimeofday( &tmp, (struct timezone *)0 );
    sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
    return sec;
}


int main(int argc, char* argv[]){
    
    if(argc<2){
	    printf("\nUsage: ./executable n_rects\n");
            return 1;
    }

    long int n_rects=atoi(argv[1]);
    
    double base=1./n_rects;
    double pi=0;

    double t_in=seconds();
//Put your pragmas here
    for( long int i =0; i<n_rects; i++){
       double x=base*(1.*i-0.5);
       double fx=1/(1.+x*x);
       pi+=fx;
    }
    pi*=4*base;
    double t_end=seconds();

    printf("The estimate for pi is: %lf\n The time elapsed is %lf\n", pi, t_end-t_in);

    return 0;
}
