#include<stdio.h>
#include<stdlib.h>

int main()
{

        int n=1000000000;

        int * a = ( int * )  malloc( n * sizeof(int) );
        int * b = ( int * )  malloc( n * sizeof(int) );
        int * c = ( int * )  malloc( n * sizeof(int) );

#pragma acc parallel loop
        for(int i=0; i<n; i++){
                a[i]=3;
                b[i]=4;
        }

#pragma acc parallel loop
        for(int j=0; j<n; j++){
                a[j] = 2*a[j];
                b[j] = b[j]+1;
        }

#pragma acc parallel loop
        for(int k=0; k<n; k++){
                c[k] = a[k]+b[k];
        }
	
      
        for(int l=0; l<10; l++){
                printf("c[%d]=%d\n",l, c[l]);
           }
        printf("\n");

        return 0;
}
