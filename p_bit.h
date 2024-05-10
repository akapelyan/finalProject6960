#include "ap_fixed.h"
#include "ap_int.h"
#include <hls_stream.h>
#include <ap_axi_sdata.h>

// control to use floating point or fixed point data types
#define USE_FLOAT 0 // 1 for floating point, 0 for fixed point
#define USE_6775_FPGA 1

// define the width of the fixed point data types
#ifndef MAX_WIDTH
#define MAX_WIDTH 16
#endif

#ifndef intBits
#define intBits 4
#endif

#ifndef fracBits
#define fracBits 16
#endif

#define N 65

#define num_anneals 1

#if USE_6775_FPGA
    typedef ap_uint<32> bit32_t;
    typedef ap_uint<MAX_WIDTH> bit_Width_t;
    typedef ap_uint<2>  bit2_t;
#endif

#if USE_FLOAT
    typedef float data_type_J;
    typedef float data_type_x;
    typedef float data_type_e;
    typedef float spin_sign  ;
#else
    typedef ap_fixed<MAX_WIDTH, intBits> data_type_J;
    typedef ap_int<2> data_type_x;
    typedef ap_fixed<MAX_WIDTH, intBits> data_type_e;
    typedef ap_int<2> spin_sign;
#endif

// Maximize the number of MIMO channels N we can decode per FPGA,
// subject to a constraint on the maximum allowable error rate.

#define sign_bit(x) (x > 0) ? 1 : ((x==0.0 | x==0) ? 0 : -1)

class AHC{
    public:
        AHC();

        data_type_e bestEnergySpins(spin_sign bestSpins[N]);

        void ahc_solver(
			data_type_x x_init[N],int run
		);

        void Mat_Vec_Mal();
        void update(int run, int timestep);
        void setSpins();
        void IsingEnergy();

        void reset();
        //void ahc_solver(data_type_x x_init[N]);
        void writeDebug(int index);
        void updateX(data_type_x x_init[N], int run);
        void updateJ(data_type_J J_init[N][N]);

    private:

        int num_time_steps;

        data_type_J J[N][N];
        spin_sign x[N];

        data_type_e MVM_out[N];

        data_type_e bestEnergy = 10;
        spin_sign bestSpins[N];
        spin_sign lastSpins[N];
};



#if USE_6775_FPGA
void dut(hls::stream<bit32_t> &strm_in, hls::stream<bit32_t> &strm_out);
#else
void dut(data_type_J J_matrix[N][N], data_type_x x_init[N], spin_sign bestSpinsOut[N], data_type_e bestEnergyOut);
#endif
