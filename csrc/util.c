#include "util.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>

// Box-Muller transform for normal distribution
float randn_float() {
    static int has_spare = 0;
    static float spare;
    
    if (has_spare) {
        has_spare = 0;
        return spare;
    }
    
    has_spare = 1;
    
    float u, v, s;
    do {
        u = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        v = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);
    
    s = sqrtf(-2.0f * logf(s) / s);
    spare = v * s;
    return u * s;
}