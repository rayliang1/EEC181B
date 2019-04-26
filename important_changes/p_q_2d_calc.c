// CODE IS CORRECT

#include <stdio.h>
#include <math.h>

void disp(float** A, int row, int col){
	/*
	Function to print the contents of the matrices (2D maximum)
	Note the type float and double pointer (needs 2D arrays, use col = 1 or row = 1 for 1D vector)
	*/
    int i, j;
    for(i=0; i<row; i++){
        for(j=0; j<col; j++)
            printf("%f\t", A[i][j]);
        printf("\n");
    }
}	// Tested

int main(){
    float alpha = 0.1;
    float beta = 0.01;
    
    float e = 0.23279564738785163;
    
    int i;
    float **P = (float**)malloc(5*sizeof(float*));
    for(i = 0; i<5; i++)
        P[i] = (float*)malloc(2*sizeof(float));
    
    float **Q = (float**)malloc(4*sizeof(float*));
    for(i=0; i<4; i++)
        Q[i] = (float*)malloc(2*sizeof(float));
        
    P[0][0] = -0.92699867; P[0][1] = 0.35551501;
    P[1][0] = -0.96932012; P[1][1] = -0.76431675;
    P[2][0] = -0.22435221; P[2][1] = -0.7446414;
    P[3][0] = -1.00114266; P[3][1] = 0.24829553;
    P[4][0] = -0.0374667;  P[4][1] = -0.41356484;
    
    Q[0][0] = -0.08878777; Q[0][1] = -0.14341404;
    Q[1][0] = 0.07627995;  Q[1][1] = -0.19536302;
    Q[2][0] = 0.38885663;  Q[2][1] = -0.03886003;
    Q[3][0] = -0.87069894; Q[3][1] = 0.50858641;
    
    float* P_i = (float*)malloc(2 * sizeof(float));
	int x;
	for(x=0; x<2; x++)
		P_i[x] = P[3][x];
	
	for(x=0; x<2; x++){												
		P[3][x] = P[3][x] + alpha * (e * Q[3][x] - beta * P[3][x]);		
		P_i[x] = P[3][x];
		printf("%0.9f\n", P[3][x]);
	}   // P[3,0] = -1.02041101, P[3,1] = 0.2598869     <=  original Python values
	
	printf("\n");
	for(x=0; x<2; x++){
		Q[3][x] = Q[3][x] + alpha * (e * P_i[x] - beta * Q[3][x]);
		printf("%0.9f\n",Q[3][x]);
	}	// Q[3,0] = -0.89358296, Q[3,1] = 0.51412788    <= original Python values
    
    printf("\n");
    disp(P,5,2);
    printf("\n");
    disp(Q,4,2);
    
    /*  P and Q matrices from Python, should match:
    P = 
        [-0.92699867,  0.35551501]
        [-0.96932012, -0.76431675]
        [-0.22435221, -0.7446414 ]
        [-1.02041101,  0.2598869 ]
        [-0.0374667 , -0.41356484]
        
    Q = 
        [-0.08878777, -0.14341404]
        [ 0.07627995, -0.19536302]
        [ 0.38885663, -0.03886003]
        [-0.89358296,  0.51412788]
    */
    
    return 0;
}
