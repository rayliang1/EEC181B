
#include <stdio.h>

int main()
{
    float alpha = 0.1;
    float beta = 0.01;
    //float e = 0.23279564738785163;  // b_u[3] = b_i[3] = 0.02327956 <= original Python value
                                   
                                    
    float e = 1.468562238120115; // b_u[3] = b_i[3] = 0.14685622 <= original Python value
    
    float *b_u = (float*)malloc(5 * sizeof(float));
	  float *b_i = (float*)malloc(4 * sizeof(float));

    int i;
	  for(i=0; i<5; i++)
		  b_u[i] = 0;
	  for(i=0; i<4; i++)
		  b_i[i] = 0;
    
	  b_u[3] = b_u[3] + alpha * (e - beta * b_u[3]);							// need to check i usage
	  b_i[3] = b_i[3] + alpha * (e - beta * b_i[3]);							// need to check j usage
	
	  printf("%0.9f\n", b_u[3]);  
	  printf("%0.9f\n", b_i[3]);  
    return 0;
}
