// IMPORTANT: USE THIS P AND Q CALCULATION IN TRAIN.C
// THIS IS FIXED P AND Q CALCULATION, BUT IN 1D VECTOR (AS OPPOSED TO 2D VECTOR IN ORIGINAL TRAIN)


#include <stdio.h>
#include <math.h>

int main(){
    float alpha = 0.1;
    float beta = 0.01;
    
    
    float e = 1.468562238120115;
    float *P_i = (float*)malloc(2*sizeof(float));
    P_i[0] = 0.33309937;
    P_i[1] = -0.08719953;
    
    float *Q = (float*)malloc(2*sizeof(float));
    Q[0] = -0.6867654;
    Q[1] = 0.10357725;
    
    float *P = (float*)malloc(2*sizeof(float));
    P[0] = 0.33309937;
    P[1] = -0.08719953;
    
    int x;
    for(x=0; x<2; x++){
	P[x] = P[x] + alpha * (e * Q[x] - beta * P[x]);
	P_i[x] = P[x];
        printf("%0.9f\n",P[x]);
    }   // P[0] = 0.23191049, P[1] = -0.07190137    <= original Python values
    
    printf("\n");
    for(x=0; x<2; x++){
	Q[x] = Q[x] + alpha * (e * P_i[x] - beta * Q[x]);
	printf("%0.9f\n",Q[x]);
	//printf("%0.9f * %0.9f = ", e, P_i[x]);
	//printf("%0.9f\n", e*P_i[x]);
    }   // Q[0] = -0.65202113, Q[1] = 0.09291451    <= original Python values
    
    return 0;
}
