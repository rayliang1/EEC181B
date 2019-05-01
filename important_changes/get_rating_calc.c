// NOW CORRECT
// PREVIOUS ERROR WAS IN INCORRECT INDEX USE IN A, B ASSIGNMENT AT LINE 58


#include <stdio.h>
#include <math.h>

float b;
float* b_u;
float* b_i;
float** P;
float** Q;
float alpha = 0.1;
float beta = 0.01;
float b = 2.769230769230769;

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

float vectordot(float* A, float* B, int lenA, int lenB){
	/*
	Function to compute the result of dot product between two vectors (row and column)
	Returns float number for dot product result
	lenA and lenB are sizes of the vectors A and B
	*/
	if(lenA != lenB)
		return 0;
	else{
		int i=0; float sum = 0;
		for(i=0; i<lenA; i++)
			sum = sum + A[i]*B[i];
		return sum;
	}
}	// Tested

float get_rating(int i, int j, int Prows, int Pcols, int Qrows, int Qcols){
	/*
	Function to retrieve prediction entry (rating entry) from final prediction 2D array
	Input arguments: b, b_u, b_i biases computed in int main()
					 i := user
					 j := item (movie, object, etc)
					 P, Q resultant biases
	Returns the floating point entry at row i column j of the final prediction matrix
	*/
	int k, lenA = Pcols, lenB = Qcols;
	float *A = (float*) malloc(lenA * sizeof(float));
	float *B = (float*) malloc(lenB * sizeof(float));
	for(k=0; k<lenA; k++){		// exporting P and Q into different 1D vectors A and B
		A[k] = P[i][k];
		B[k] = Q[j][k];
	}
	
	float prediction = b + b_u[i] + b_i[j] + vectordot(A,B,lenA,lenB);
	printf("dotprod = %0.9f\n\n", vectordot(A,B,lenA,lenB));
	return prediction;
}	// Tested

int main(){
    //float e = 0.23279564738785163;
    // Variables b_u, b_i, b, samples, samples_x, samples_y are ok
    int i;
	b_u = (float*)malloc(5 * sizeof(float));
	b_i = (float*)malloc(4 * sizeof(float));
	for(i=0; i<5; i++)
		b_u[i] = 0;
	for(i=0; i<4; i++)
		b_i[i] = 0;
		
		
    P = (float**)malloc(5*sizeof(float*));
    for(i = 0; i<5; i++)
    	P[i] = (float*)malloc(2*sizeof(float));
    
    Q = (float**)malloc(4*sizeof(float*));
    for(i=0; i<4; i++)
    	Q[i] = (float*)malloc(2*sizeof(float));
        
    P[0][0] = -0.45385641; P[0][1] = 0.12240304;
    P[1][0] = 0.10962749;  P[1][1] = -0.55365739;
    P[2][0] = -0.21193127; P[2][1] = 0.32004102;
    P[3][0] = 0.50767246;  P[3][1] = 0.40353577;
    P[4][0] = -0.21690251; P[4][1] = -1.116791;
    
    Q[0][0] = -0.3499104;  Q[0][1] = 0.17541916;
    Q[1][0] = -0.62551354; Q[1][1] = 0.15131101;
    Q[2][0] = 0.03058709;  Q[2][1] = -0.59956074;
    Q[3][0] = 0.39453033;  Q[3][1] = -0.22296095;
    
    printf("b = %0.9f\n", b);               // b should be 2.769230769230769
    printf("b_u[3] = %0.9f\n", b_u[3]);     // b_u[3] = b_i[3] = 0
    printf("b_i[3] = %0.9f\n", b_i[3]);
    //printf("dotprod = %0.9f\n\n", vectordot())                    // dotprod = 0.11031946267490311 <= in Python
    printf("get_rating(3,3) = %0.9f\n",get_rating(3,3,5,2,4,2));    // get_rating(3,3) = 2.8795502319056725 <= in Python
    return 0;
}
