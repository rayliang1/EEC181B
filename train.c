// update: Changed P, Q assignment with lines taken from important_changes
// output: still nan

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// Global Variables, should be computed by the program:
// Variables below need to be global because they will be modified by sgd() function
float b;
float* b_u;
float* b_i;
float** P;
float** Q;

// User defined input arguments:
int R_rows, R_cols, K, iterations;
float alpha, beta;
float **R;

float *samples;
int *samples_x;
int *samples_y;


struct trainingprocess{
	/*
	Structure will be used for train() function
	Stores the iteration number and the mean squared error at that iteration
	*/
	int iteration_num;
	float mse_value;
};

void deletematrix(float** matrix, int row, int col){
	/*
	TODO: Deallocates memory allocated for 2D matrix (malloc)
	*/
	
}


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

void shuffle(int *array, size_t n){
	/* 
	https://stackoverflow.com/questions/6127503/shuffle-array-in-c
	Arrange the N elements of ARRAY in random order.
	Only effective if N is much smaller than RAND_MAX;
	if this may not be the case, use a better random
	number generator. 
	*/
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}	// Tested

void shufflesamples(int len){
	/*
	Replacement function for numpy.random.shuffle(self.samples)
	This function shuffles the order of self.samples in Python or samples, samples_x, and samples_y in C
	Input argument len is the size of samples_x, samples_y, and samples 1D array
	*/
	int* idx = (int*)malloc(len * sizeof(int));
	int i, newindex = 0;
	for(i=0; i<len; i++)
		idx[i] = i;
	
	shuffle(idx,len);
	
	float* samples_copy = (float*)malloc(len*sizeof(float));
	int* samples_x_copy = (int*)malloc(len*sizeof(int));
	int* samples_y_copy = (int*)malloc(len*sizeof(int));
	
	// Storing shuffled samples, samples_x, samples_y in new respective 1D pointer vectors
	for(i=0; i<len; i++){
		int index = idx[i];
		samples_copy[newindex] = samples[index];
		samples_x_copy[newindex] = samples_x[index];
		samples_y_copy[newindex] = samples_y[index];
		newindex++;
	}
	
	// Moving shuffled variables back into samples, samples_x, samples_y
	for(i=0; i<len; i++){
		samples[i] = samples_copy[i];
		samples_x[i] = samples_x_copy[i];
		samples_y[i] = samples_y_copy[i];
	}
	//free(idx);
	//free(samples_copy);
	//free(samples_x_copy);
	//free(samples_y_copy);
}	// Tested

float** matrixdot(float** A, float** B, int rowA, int colA, int rowB, int colB){
	/*
	Function to compute the result of matrix multiplication (dot product)
	Returns resulting matrix in double pointer form
	Return type is float for prediction entry
	rowA is the number of rows of matrix A, colA is the number of cols of matrix A
	*/
	if(colA!=rowB)
		return NULL;
	else{
	    int i, j, k=0;
		float sum=0;
		float **res = (float **)malloc(rowA * sizeof(float*));
		for(i=0; i<rowA; i++)
		    res[i] = (float *)malloc(colB * sizeof(float));
		    
		for(i=0; i<rowA; i++){
			for(j=0; j<colB; j++){
				while(k<colA){
					sum = sum + A[i][k]*B[k][j];
					k++;
				}
				res[i][j] = sum;
				sum = 0;
				k = 0;
			}
		}
		return res;
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
	//free(A);
	//free(B);
	return prediction;
}	// Tested

void sgd(int iter, int Prows, int Pcols, int Qrows, int Qcols){
	/*
	Computes stoch gradient descent
	Input argument: iter <- iterator found in "for i, j, r in self.samples:" in python code
	There is no output argument, function updates bias vectors (global variables)
	*/
	int k;
	float r; int i, j;
	for(k=0; k<iter; k++){
		// Unpack the variables 'i, j, and r in self.samples' -> unpack samples_x, samples_y, and samples in C
		// Store in 3 1D matrices (alternative, store in 1 2D matrix with one column representing samples, etc.)
		i = samples_x[k];
		j = samples_y[k];
		r = samples[k];
		float prediction = get_rating(i, j, Prows, Pcols, Qrows, Qcols);		// need to check i and j usage
		float e = r - prediction;
		
		b_u[i] = b_u[i] + alpha * (e - beta * b_u[i]);							// need to check i usage
		b_i[j] = b_i[j] + alpha * (e - beta * b_i[j]);							// need to check j usage
		
		float* P_i = (float*)malloc(Pcols * sizeof(float));
		int x;
		for(x=0; x<Pcols; x++)
			P_i[x] = P[i][x];
		
		// Done per column; cannot operate on both columns. Check the math for the 2 lines below:
		for(x=0; x<Pcols; x++){		
			P[i][x] = P[i][x] + alpha * (e * Q[j][x] - beta * P[i][x]);
			P_i[x] = P[i][x];
		}
		for(x=0; x<Pcols; x++)
			Q[j][x] = Q[j][x] + alpha * (e * P_i[x] - beta * Q[j][x]);
		
		// free(P_i);
	}
	
}	// Needs more testing, some variables off


////////////////////////////////////////// RAYMOND'S FUNCTIONS

float** full_matrix(){

    int i,j;

    float **bcomb = (float**)malloc(4 * sizeof(float*));				// bcomb is: self.b + self.b_u[:,np.newaxis] + self.b_I[np.newaxis:,]
	for(i=0; i<5; i++) {
		bcomb[i] = (float *)malloc(5 * sizeof(float));
	}
    //declaring a 5by5 matrix bcomb used to calculate the b_u etc
    
    float **Qtrans = (float **)malloc(4 * sizeof(int*));				// Q transpose
	for(i=0; i<5; i++) {
		Qtrans[i] = (float *)malloc(2 * sizeof(int));
	}
    //transing q, declaring Q trans
    
    for (i = 0;i<4;i++)
    {
        for (j = 0; j<5;j++){
            bcomb[j][i] = b_u[j] + b_i[i] + b;		// should be b instead of 2.7
            //adding bias of 0.224, don't know why they make it smaller, this is self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] 
        }
    }

    for (i = 0; i<4; i++) 
    {
        for (j =0;j<2;++j)  {
            Qtrans[j][i] = Q[i][j];
        }
    }
   //transing Q 
    float **QdotP = (float **)malloc(5 * sizeof(float*));			// self.P.dot(self.Q.T)
	for(i=0; i<5; i++) {
		QdotP[i] = (float *)malloc(5 * sizeof(float));
	}
	QdotP = matrixdot(P,Qtrans,5,2,2,4);
    //Q and P multiplied together, a 4by5 matrix
    
    float **fullmatrix = (float **)malloc(R_rows * sizeof(float*));
	for(i=0; i<5; i++) {
		fullmatrix[i] = (float *)malloc(R_cols * sizeof(float));
	}
    // declaring room for full matrix
    for (i = 0; i<5; i++) {
        for (j=0;j<4;j++) {
            fullmatrix[i][j] = QdotP[i][j] + bcomb[i][j];
        }
    }
    // https://scontent.fsac1-2.fna.fbcdn.net/v/t1.15752-0/p50x50/56879323_278578126395723_2395177873703960576_n.png?_nc_cat=110&_nc_ht=scontent.fsac1-2.fna&oh=b5ab5e040eb412c44a8da97de063e7d1&oe=5D333273 //adding everything together
    return fullmatrix;
}	// Tested by Raymond

float mse() {
	/*
	Computes mean squared error, does not take in input arguments
	Returns a floating number, the mse
	*/
    float error = 0;
    int i,j;
    float **predicted = full_matrix();
    //assigs predictions from fullmatrix
    for(i=0; i<R_rows; i++) {
        for(j=0; j<R_cols; j++) {
            if(R[i][j] != 0)
                error = error + pow((R[i][j] - predicted[i][j]),2);         
        }
    }
    return sqrt(error);
}	// Tested by Raymond

////////////////////////////////////////// END OF RAYMOND'S FUNCTIONS

void train(){
	/*
	TODO: Modify the function so that it returns array of 'struct trainingprocess'
		  Check struct usage (syntax)
	Train function in python code, calls mse and sgd
	*/
	
	// TODO: NEED TO INITIALIZE P AND Q USING GAUSSIAN DISTRIBUTION
	
	// Temporary P and Q values taken from Python test
	int i, j, k=0;
	
	P = (float**)malloc(5*sizeof(float*));
	for(i=0;i<5;i++)
	    P[i] = (float*)malloc(2*sizeof(float));
	Q = (float**)malloc(4*sizeof(float*));
	for(i=0;i<4;i++)
	    Q[i] = (float*)malloc(2*sizeof(float));
	    
	P[0][0]=-0.10647869; P[0][1]=1.29320198;
	P[1][0]=-0.17975839; P[1][1]=0.90424228;
	P[2][0]=0.37239126;  P[2][1]=-1.27146798;
	P[3][0]=0.72194025;  P[3][1]=-0.75267798;
	P[4][0]=0.05120293;  P[4][1]=-0.71838542;
	
	Q[0][0]=-1.52001292; Q[0][1]=1.31483326;
	Q[1][0]=1.10224395;  Q[1][1]=1.04025647;
	Q[2][0]=1.11982426;  Q[2][1]=-0.27210834;
	Q[3][0]=-0.09565345; Q[3][1]=-1.50853652;
	
	// Variables b_u, b_i, b, samples, samples_x, samples_y are ok
	b_u = (float*)malloc(R_rows * sizeof(float));
	b_i = (float*)malloc(R_cols * sizeof(float));

	for(i=0; i<R_rows; i++)
		b_u[i] = 0;
	for(i=0; i<R_cols; i++)
		b_i[i] = 0;
	
	// Assume matrix R is 2D
	float sum = 0; int tally = 0;
	for(i=0; i<R_rows; i++){
		for(j=0; j<R_cols; j++){
		    if(R[i][j]>0){
    			sum = sum + R[i][j];
    			tally++;
		    }
		}
	}
	b = sum/tally;					// compute average of row*col elements inside 2D array R
	// b = 2.769;
	
	// self.samples: samples_x represent 'num_users' or 'i'
	//				 samples_y represent 'num_items' or 'j'
	//				 samples contain nonzero entries in R 2D array
	
	samples = (float*)malloc(R_rows*R_cols*sizeof(float));
	samples_x = (int*)malloc(R_rows*R_cols*sizeof(int));
	samples_y = (int*)malloc(R_rows*R_cols*sizeof(int));
	
	for(i=0; i<R_rows; i++){
	    for(j=0; j<R_cols; j++){
	        if(R[i][j]>0){
	            samples[k] = R[i][j];
	            samples_x[k] = i;
	            samples_y[k] = j;
	            k++;					// pass k into void sgd() for iterator
	        }
	    }
	}
	
	struct trainingprocess training_process[iterations];
	for(i=0; i<iterations; i++){
		shufflesamples(k);
		sgd(k,5,2,4,2);
		training_process[i].iteration_num = i;
		training_process[i].mse_value = mse();
		printf("Iteration: %d\t Error: %.4f\n", i, mse());	
		/*
		if((i+1)%10 == 0)
			printf("Iteration: %d\t Error = %.4f\n", i+1, mse());
		*/
	}
	// return training_process;
}	// Incorrect values from incorrect sgd()

int main(){
	// This main function tests the function train
	
	// Initialize all USER DEFINED VARIABLES/INPUTS
	R_rows = 5;
	R_cols = 4;
	alpha = 0.1;
	beta = 0.01;
	K = 2;
	iterations = 20;
	
	int R_iterator;
	R = (float **)malloc(R_rows * sizeof(float*));
	for(R_iterator=0; R_iterator<R_rows; R_iterator++)
    	R[R_iterator] = (float*)malloc(R_cols * sizeof(float));

    R[0][0] = 5;	R[0][1] = 3;	R[0][2] = 0;	R[0][3] = 1;
    R[1][0] = 4;	R[1][1] = 0;	R[1][2] = 0;	R[1][3] = 1;    
    R[2][0] = 1;	R[2][1] = 1;	R[2][2] = 0;	R[2][3] = 5;
    R[3][0] = 1;	R[3][1] = 0;	R[3][2] = 0;	R[3][3] = 4;
    R[4][0] = 0;	R[4][1] = 1;	R[4][2] = 5;	R[4][3] = 4;
	
	train();
	float** out_matrix = (float**)malloc(R_rows * sizeof(float*));
	for(R_iterator=0; R_iterator<R_rows; R_iterator++)
		out_matrix[R_iterator] = (float*)malloc(R_cols * sizeof(float));
	
	out_matrix = full_matrix();
	printf("\n\n");
	disp(out_matrix,R_rows,R_cols);
	
	return 0;
}


/*
int main(){
	// This main function tests the function sgd
	
	// Initialize 2D array R beforehand
	int R_iterator;
	R = (float **)malloc(R_rows * sizeof(float*));
	for(R_iterator=0; R_iterator<R_rows; R_iterator++)
    	R[R_iterator] = (float*)malloc(R_cols * sizeof(float));

    R[0][0] = 5;	R[0][1] = 3;	R[0][2] = 0;	R[0][3] = 1;
    R[1][0] = 4;	R[1][1] = 0;	R[1][2] = 0;	R[1][3] = 1;    
    R[2][0] = 1;	R[2][1] = 1;	R[2][2] = 0;	R[2][3] = 5;
    R[3][0] = 1;	R[3][1] = 0;	R[3][2] = 0;	R[3][3] = 4;
    R[4][0] = 0;	R[4][1] = 1;	R[4][2] = 5;	R[4][3] = 4;
	
	// Values for variables below retrieved from Python code after running code
    int i, j, k=0;
	b = 2.769230769230769;
	b_u = (float*)malloc(5*sizeof(float));
	b_i = (float*)malloc(4*sizeof(float));
	b_u[0] = 0.19883971; b_u[1] = -0.39881915; b_u[2] = 0.31547987; b_u[3] = 0.16384996; b_u[4] = 0.23911689;
	b_i[0] = 0.16125589; b_i[1] = -1.20730111; b_i[2] = 1.62633722; b_i[3] = -0.0185079;
	
	P = (float**)malloc(5*sizeof(float*));
	for(i=0;i<5;i++)
	    P[i] = (float*)malloc(2*sizeof(float));
	Q = (float**)malloc(4*sizeof(float*));
	for(i=0;i<4;i++)
	    Q[i] = (float*)malloc(2*sizeof(float));
	    
	P[0][0]=-0.10647869; P[0][1]=1.29320198;
	P[1][0]=-0.17975839; P[1][1]=0.90424228;
	P[2][0]=0.37239126;  P[2][1]=-1.27146798;
	P[3][0]=0.72194025;  P[3][1]=-0.75267798;
	P[4][0]=0.05120293;  P[4][1]=-0.71838542;
	
	Q[0][0]=-1.52001292; Q[0][1]=1.31483326;
	Q[1][0]=1.10224395;  Q[1][1]=1.04025647;
	Q[2][0]=1.11982426;  Q[2][1]=-0.27210834;
	Q[3][0]=-0.09565345; Q[3][1]=-1.50853652;
	
	// self.samples: samples_x represent 'num_users' or 'i'
	//				 samples_y represent 'num_items' or 'j'
	//				 samples contain nonzero entries in R 2D array
	
	samples = (float*)malloc(R_rows*R_cols*sizeof(float));
	samples_x = (int*)malloc(R_rows*R_cols*sizeof(int));
	samples_y = (int*)malloc(R_rows*R_cols*sizeof(int));
	
	for(i=0; i<R_rows; i++){
	    for(j=0; j<R_cols; j++){
	        if(R[i][j]>0){
	            samples[k] = R[i][j];
	            samples_x[k] = i;
	            samples_y[k] = j;
	            k++;					// pass k into void sgd() for iterator
	        }
	    }
	}
	
	disp(R,5,4);					// test displaying contents of R 2D array
	sgd(k,5,2,4,2);					// sgd function call
	
	// print bias vectors and P, Q
	for(i=0; i<5; i++){
	    printf("%f\t",b_u[i]);
	}
	printf("\n");
	for(i=0; i<4; i++){
	    printf("%f\t",b_i[i]);
	}
	printf("\n\n");
	disp(P,5,2);
	printf("\n\n");
	disp(Q,4,2);
	return 0;
}
*/

/*
int main(){
	// This main function tests the function get_rating, values are slightly off:
	// In Python, with >> mf.get_rating(1,2) = 3.5493991670252867
	// In C, with >> get_rating(1,2,5,2,4,2) = 3.750697
	
    // Values for variables below retrieved from Python code
    int i;
	b = 2.769230769230769;
	b_u = (float*)malloc(5*sizeof(float));
	b_i = (float*)malloc(4*sizeof(float));
	b_u[0] = 0.19883971; b_u[1] = -0.39881915; b_u[2] = 0.31547987; b_u[3] = 0.16384996; b_u[4] = 0.23911689;
	b_i[0] = 0.16125589; b_i[1] = -1.20730111; b_i[2] = 1.62633722; b_i[3] = -0.0185079;
	
	P = (float**)malloc(5*sizeof(float*));
	for(i=0;i<5;i++)
	    P[i] = (float*)malloc(2*sizeof(float));
	Q = (float**)malloc(4*sizeof(float*));
	for(i=0;i<4;i++)
	    Q[i] = (float*)malloc(2*sizeof(float));
	    
	P[0][0]=-0.10647869; P[0][1]=1.29320198;
	P[1][0]=-0.17975839; P[1][1]=0.90424228;
	P[2][0]=0.37239126;  P[2][1]=-1.27146798;
	P[3][0]=0.72194025;  P[3][1]=-0.75267798;
	P[4][0]=0.05120293;  P[4][1]=-0.71838542;
	
	Q[0][0]=-1.52001292; Q[0][1]=1.31483326;
	Q[1][0]=1.10224395;  Q[1][1]=1.04025647;
	Q[2][0]=1.11982426;  Q[2][1]=-0.27210834;
	Q[3][0]=-0.09565345; Q[3][1]=-1.50853652;
	
	// printing get_rating() results for all x, y coordinates in self.samples
	printf("%f\n",get_rating(2,1,5,2,4,2));
	printf("%f\n",get_rating(4,3,5,2,4,2));
	printf("%f\n",get_rating(4,2,5,2,4,2));
	printf("%f\n",get_rating(0,1,5,2,4,2));
	printf("%f\n",get_rating(1,3,5,2,4,2));
	printf("%f\n",get_rating(4,1,5,2,4,2));
	printf("%f\n",get_rating(2,0,5,2,4,2));
	printf("%f\n",get_rating(2,3,5,2,4,2));
	printf("%f\n",get_rating(0,3,5,2,4,2));
	printf("%f\n",get_rating(3,0,5,2,4,2));
	printf("%f\n",get_rating(3,3,5,2,4,2));
	printf("%f\n",get_rating(1,0,5,2,4,2));
	printf("%f\n",get_rating(0,0,5,2,4,2));
	return 0;
}
*/
