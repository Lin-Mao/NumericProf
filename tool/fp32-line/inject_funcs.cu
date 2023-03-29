#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the instr_info_t structure */
#include "common.h"

__device__
static 
__forceinline__ 
ExceptionType_t _FPC_FP32_IS_INF(uint32_t reg_val) {
    uint32_t exponent, mantissa; 
    exponent = reg_val << 1;
    exponent = exponent >> 24; 
    mantissa = reg_val << 9; 
    mantissa = mantissa >> 9;
    if(exponent == (uint32_t)(255) && mantissa == (uint32_t)(0)){
        return E_INF;
    }
    return NONE; 
}

__device__
static 
__forceinline__ 
ExceptionType_t _FPC_FP32_IS_NAN(uint32_t reg_val) {
    uint32_t exponent, mantissa; 
    exponent = reg_val << 1;
    exponent = exponent >> 24; 
    mantissa = reg_val << 9; 
    mantissa = mantissa >> 9;
    if(exponent == (uint32_t)(255) && mantissa != (uint32_t)(0)){
        return E_NAN;
    }
    return NONE; 
}

__device__
static 
__forceinline__ 
ExceptionType_t _FPC_FP32_IS_SUBNORMAL(uint32_t reg_val) {
    uint32_t exponent, mantissa; 
    exponent = reg_val << 1;
    exponent = exponent >> 24; 
    mantissa = reg_val << 9; 
    mantissa = mantissa >> 9;
    if(exponent == (uint32_t)(0) && mantissa != (uint32_t)(0)){
        return E_SUB;
    }
    return NONE; 
}

__device__
static 
__forceinline__ 
ExceptionType_t _FPC_FP32_IS_0(uint32_t reg_val) {
    if(_FPC_FP32_IS_INF(reg_val)!=0||_FPC_FP32_IS_NAN(reg_val)!=0){
        return E_DIV0;
    }
    return NONE;
}

__device__
static 
__forceinline__ 
ExceptionType_t _CHECK_FP32(uint32_t reg_val) {
    ExceptionType_t result;
    result = _FPC_FP32_IS_INF(reg_val);
    if (result != NONE) {
        return result;
    }

    result = _FPC_FP32_IS_NAN(reg_val);
    if (result != NONE) {
        return result;
    }

    result = _FPC_FP32_IS_SUBNORMAL(reg_val);
    if (result != NONE) {
        return result;
    }

    result = _FPC_FP32_IS_0(reg_val);
    if (result != NONE) {
        return result;
    }

    return NONE;
}


extern "C" __device__ __noinline__ void check_exceptions(int pred, int opcode_id,
                                                       uint64_t grid_launch_id,
                                                       uint64_t pchannel_dev,
                                                       uint32_t data,
                                                       uint32_t pc) {
    /* if thread is predicated off, return */
    if (!pred) {
        return;
    }

    int active_mask = __ballot_sync(__activemask(), 1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;

    instr_info_t ii;

    ExceptionType_t exception = _CHECK_FP32(data);
    for (int i = 0; i < 32; i++) {
        ii.exception[i] = (ExceptionType_t) __shfl_sync(active_mask, exception, i);
    }

    for (int i = 0; i < 32; i++) {
        ii.data[i] = (uint32_t) __shfl_sync(active_mask, data, i);
    }

    int4 cta = get_ctaid();
    ii.grid_launch_id = grid_launch_id;
    ii.cta_id_x = cta.x;
    ii.cta_id_y = cta.y;
    ii.cta_id_z = cta.z;
    ii.warp_id = get_warpid();
    ii.opcode_id = opcode_id;
    ii.pc = pc;

    /* first active lane pushes information on the channel */
    if (first_laneid == laneid) {
        ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
        channel_dev->push(&ii, sizeof(instr_info_t));
    }
}

