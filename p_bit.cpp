#include "p_bit.h"
#include "C:\Users\ari\AppData\Roaming\Xilinx\Vitis\init_x.h"
#include "C:\Users\ari\Documents\noise_vals.h"

//#include "iostream"
//#include <fstream>

//#define DEBUG
#ifdef DEBUG
using std::cout;
using std::endl;
float x1_debug[150][65];
float x2_debug[150][65];
#endif

AHC::AHC(){
#pragma HLS INLINE

// Partition local variables to improve bandwidth
#pragma HLS ARRAY_PARTITION variable=J complete dim=1
#pragma HLS ARRAY_PARTITION variable=x complete
#pragma HLS ARRAY_PARTITION variable=MVM_out complete

	// Initialize the AHC solver
	this->num_time_steps = 256;

}

// setSpins
// This function update the lastSpins based on the current spin values
// This is used to determine the sign of the spins in the MVM
void AHC::setSpins(){
	#pragma HLS INLINE off
	setSpins_loop:
	for(int i=0; i<N; i++){
		#pragma HLS PIPELINE
		if(this->x[i] > 0){
			this->lastSpins[i] = 1;
		}
		else if(this->x[i] < 0){
			this->lastSpins[i] = -1;
		}
		else{
			this->lastSpins[i] = 0;
		}
	}
}

// Matrix vector product
void AHC::Mat_Vec_Mal()
{
	#pragma HLS INLINE off
	// Matrix vector product
	// MVM = (J).dot(np.sign(x))
	MVM_init:
	for (int row = 0; row < N; row++) {
		#pragma HLS PIPELINE
		this->MVM_out[row] = 0.0;
	}

	// using column method MVM = \sum_j J[:][j] * x[j]
	MVM_outer:
		// for each element in x
	for(int j = 0; j < N; j++){
		#pragma HLS PIPELINE II=1
		MVM_inner:
		for(int i=0; i<N; i++){
			#pragma HLS UNROLL
			// multiply with each element on i th column of J
			if(this->lastSpins[j] == 1){
				this->MVM_out[i] += this->J[i][j];
			}
			else if(this->lastSpins[j] == -1){
				this->MVM_out[i] += -(this->J[i][j]);
			}
			else {
				// this->MVM_out[i] += 0;
				continue;
			}
		}
	}
}

// Calculates the Ising energy
void AHC::IsingEnergy(){
	#pragma HLS INLINE off
	data_type_e energy = 0.0;
	IsingEnergy_loop:
	for(int i=0; i<N; i++){
		#pragma HLS PIPELINE
		data_type_e temp;
		if(this->lastSpins[i]==1){
			energy += -((this->MVM_out[i]))/2;
		}
		else if(this->lastSpins[i]==-1){
			energy += (this->MVM_out[i])/2;
		}
	}

	if(energy < this->bestEnergy){
		this->bestEnergy = energy;
		for(int k=0; k<N; k++){
			this->bestSpins[k] = this->lastSpins[k];
		}
	}
}

// Update the spins and error vectors
void AHC::update(int run, int timestep){
	#pragma HLS INLINE off

	data_type_e temp;
	update_spin_and_error:
	for(int i=0; i<N; i++){
		#pragma HLS PIPELINE
		temp = this->MVM_out[i] + noise_values[run*timestep][i];
		if(temp > 0)
			this->x[i] = 1;
		else {
			this->x[i] = -1;
		}
	}

}

void AHC::reset(){
	#pragma HLS INLINE off
	// Reset MVM
	reset_MVM:
	for(int i = 0; i < N; i++){
		#pragma HLS PIPELINE
		this->MVM_out[i] = 0.0;
	}
}

void AHC::updateX(data_type_x x_init[N],int run)
{
	initialize_vectors:for(int i=0; i<N; i++){

		this->x[i] = init_x[run][i];
	}
}

void AHC::updateJ(data_type_J J_init[N][N]){
	this->bestEnergy = 10;	// reset bestEnergy

	// reset bestSpins
	initialize_matrix_X:
	for (int i=0; i<N; i++){
		this->bestSpins[i] = 0;
	}

	initialize_matrix_J:
	for(int i=0; i<N; i++){
		for(int j=0;j<N;j++){
            this->J[i][j] = J_init[i][j];
        }
    }
}

void AHC::ahc_solver(
	data_type_x x_init[N], int run){

	updateX(x_init, run);

	setSpins();	// initialize vectors
	// Mat_Vec_Mal();

	Mat_Vec_Mal();
	TIME_STEP_LOOP:
	for(int time_step=0; time_step < num_time_steps; time_step++){
		update(run, time_step);
		IsingEnergy();

		setSpins();
		Mat_Vec_Mal();
	}
}

data_type_e AHC::bestEnergySpins(
	spin_sign bestSpins[N]
){
	for(int i=0; i<N; i++){
		bestSpins[i] = this->bestSpins[i];
	}

	return this->bestEnergy;
}
//----------------------------------------------------------
// Top function
//----------------------------------------------------------



void dut(hls::stream<bit32_t> &strm_in, hls::stream<bit32_t> &strm_out) {
	data_type_J J_in[N][N];
	data_type_x x_in[N];
	spin_sign bestSpinsOut[N];
	data_type_e bestEnergy;

	bit32_t input_l;
	bit32_t output_energy;
	bit32_t output_spin;

	static AHC ahc_instance;


	// read J matrix
	for (int i = 0; i < N; i++) {
		#pragma HLS pipeline off
		for (int j = 0; j < N; j++) {
			#pragma HLS pipeline
			input_l = strm_in.read();
			data_type_J J_receive;
			J_receive(MAX_WIDTH-1,0) = input_l(MAX_WIDTH-1,0);
			J_in[i][j] = J_receive;
			// cout << "J " << J_receive << endl;
		}
	}
	ahc_instance.updateJ(J_in);

	// run num_anneals sets of X
	for (int x_iter=0; x_iter<num_anneals; x_iter++){
		#pragma HLS pipeline off
		// read x_init
		for (int i = 0; i < N; i++) {
			#pragma HLS pipeline
			input_l = strm_in.read();
			data_type_x X_receive;
			X_receive(MAX_WIDTH-1,0) = input_l(MAX_WIDTH-1,0);
			x_in[i] = X_receive;
			// cout << "x " << X_receive << endl;
		}
		ahc_instance.ahc_solver(x_in, x_iter);
	}


	// return the best energy
	bestEnergy = ahc_instance.bestEnergySpins(bestSpinsOut);
	output_energy(MAX_WIDTH-1,0) = bestEnergy(MAX_WIDTH-1,0);
	strm_out.write(output_energy);

	// write out the result
	for (int i = 0; i < 8; i++) {
		// pack 8 spins into one transmit
		output_spin(1,0)   = reinterpret_cast<bit2_t&>(bestSpinsOut[8*i  ]);
		output_spin(3,2)   = reinterpret_cast<bit2_t&>(bestSpinsOut[8*i+1]);
		output_spin(5,4)   = reinterpret_cast<bit2_t&>(bestSpinsOut[8*i+2]);
		output_spin(7,6)   = reinterpret_cast<bit2_t&>(bestSpinsOut[8*i+3]);
		output_spin(9,8)   = reinterpret_cast<bit2_t&>(bestSpinsOut[8*i+4]);
		output_spin(11,10) = reinterpret_cast<bit2_t&>(bestSpinsOut[8*i+5]);
		output_spin(13,12) = reinterpret_cast<bit2_t&>(bestSpinsOut[8*i+6]);
		output_spin(15,14) = reinterpret_cast<bit2_t&>(bestSpinsOut[8*i+7]);
		strm_out.write(output_spin);
	}
	output_spin(1,0)   = reinterpret_cast<bit2_t&>(bestSpinsOut[64]);
	strm_out.write(output_spin);
}

