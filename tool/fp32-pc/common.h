#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <vector>

typedef enum ExceptionType {
    NONE = 0,
    E_NAN = 1,
    E_INF = 2,
    E_SUB = 3,
    E_DIV0 = 4,
    NUM_EXCE_TYPES = 5,
}ExceptionType_t;

typedef struct {
    uint64_t grid_launch_id;
    int cta_id_x;
    int cta_id_y;
    int cta_id_z;
    int warp_id;
    int opcode_id;
    uint32_t pc;
    uint32_t data[32];
    ExceptionType_t exception[32];
} instr_info_t;

// Store the instr_info with exceptions
typedef struct {
    std::vector<int> lane_id;
    instr_info_t * ii;
} exc_result_t;

#endif // COMMON_H