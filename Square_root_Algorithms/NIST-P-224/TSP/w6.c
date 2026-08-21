#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include <immintrin.h>
#include <relic/relic_core.h>
#include "relic/relic_md.h"

#define w   6
#define we  64       /* 2^w = 64 */
#define n   96
#define nw  16       /* ceil(96/6) */

int countM = 0;
int countS = 0;

#define REPEAT  100000
#define WARMUP  (REPEAT / 4)

unsigned long long RDTSC_start_clk, RDTSC_end_clk;
double RDTSC_clk[REPEAT];
double RDTSC_clk_min, RDTSC_clk_median, RDTSC_clk_max;
double min_v, l1;
int RDTSC_MEASURE_ITERATOR;
int iii, jjj, ttt;

static inline unsigned long long get_Clks(void)
{
    unsigned int lo, hi;
    __asm__ volatile (
        "cpuid\n\t"
        "rdtsc\n\t"
        : "=a"(lo), "=d"(hi)
        :
        : "rbx", "rcx", "memory"
    );
    return ((unsigned long long)hi << 32) | lo;
}

#define MEASURE(x) \
{ \
    for (RDTSC_MEASURE_ITERATOR = 0; RDTSC_MEASURE_ITERATOR < WARMUP; RDTSC_MEASURE_ITERATOR++) \
        { x }; \
    for (RDTSC_MEASURE_ITERATOR = 0; RDTSC_MEASURE_ITERATOR < REPEAT; RDTSC_MEASURE_ITERATOR++) \
    { \
        RDTSC_start_clk = get_Clks(); \
        { x }; \
        RDTSC_end_clk = get_Clks(); \
        RDTSC_clk[RDTSC_MEASURE_ITERATOR] = (double)(RDTSC_end_clk - RDTSC_start_clk); \
    } \
    for (iii = 0; iii < REPEAT; iii++) { \
        min_v = RDTSC_clk[iii]; \
        for (jjj = iii + 1; jjj < REPEAT; jjj++) { \
            if (min_v > RDTSC_clk[jjj]) { min_v = RDTSC_clk[jjj]; ttt = jjj; } \
        } \
        l1 = RDTSC_clk[ttt]; RDTSC_clk[ttt] = RDTSC_clk[iii]; RDTSC_clk[iii] = l1; \
    } \
    RDTSC_clk_min    = RDTSC_clk[0]; \
    RDTSC_clk_median = RDTSC_clk[REPEAT / 2]; \
    RDTSC_clk_max    = RDTSC_clk[REPEAT - 1]; \
}

/* =========================================================================
 * precomputation
 * ========================================================================= */
void precomputation(fp_t g, fp_t h, fp_t hh,
                    fp_t gw[nw][we], fp_t rll[we], fp_t fll[we], fp_t gpp[n])
{
    bn_t temp, one;
    bn_null(temp); bn_new(temp);
    bn_null(one);  bn_new(one);
    bn_set_dig(one, 1);
    for (int v = 0; v < we; v++) {
        bn_set_dig(temp, v);
        fp_exp(rll[v], h,  temp);
        fp_exp(fll[v], hh, temp);
    }
    for (int i = 0; i < nw; i++) {
        for (int j = 0; j < we; j++) {
            bn_lsh(temp, one, i * w);
            bn_mul_dig(temp, temp, j);
            fp_exp(gw[i][j], g, temp);
        }
    }
    fp_copy(gpp[0], g);
    for (int i = 1; i < n; i++)
        fp_sqr(gpp[i], gpp[i - 1]);
    bn_free(temp);
    bn_free(one);
}


static void SELECT(fp_t a0, fp_t a1, bool ctl, fp_t out)
{
    if (ctl) fp_copy(out, a1);
    else     fp_copy(out, a0);
}




static void GPOW_i0_e47(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 47) - 1;
    uint64_t wm = (1 << w) - 1;
    fp_copy(t, gw[0][e & wm]); e >>= w;
    fp_mul(t, t, gw[1][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[2][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[3][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[4][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[5][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[6][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[7][e & wm]); //countM++;
}


static void GPOW_i24_e48(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 48) - 1;
    uint64_t wm = (1 << w) - 1;
    fp_copy(t, gw[4][e & wm]); e >>= w;
    fp_mul(t, t, gw[5][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[6][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[7][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[8][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[9][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[10][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[11][e & wm]); //countM++;
}


static void GPOW_i47_e24(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 24) - 1;
    e <<= 5;
    uint64_t wm = (1 << w) - 1;
    fp_copy(t, gw[7][e & wm]); e >>= w;
    fp_mul(t, t, gw[8][e & wm]);  //countM++; 
e >>= w;
    fp_mul(t, t, gw[9][e & wm]);  //countM++; 
e >>= w;
    fp_mul(t, t, gw[10][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[11][e & wm]); //countM++;
}


static void GPOW_i48_e24(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 24) - 1;
    uint64_t wm = (1 << w) - 1;
    fp_copy(t, gw[8][e & wm]); e >>= w;
    fp_mul(t, t, gw[9][e & wm]);  //countM++; 
e >>= w;
    fp_mul(t, t, gw[10][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[11][e & wm]); //countM++;
}


static void GPOW_i60_e24(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 24) - 1;
    uint64_t wm = (1 << w) - 1;
    fp_copy(t, gw[10][e & wm]); e >>= w;
    fp_mul(t, t, gw[11][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[12][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[13][e & wm]); //countM++;
}


static void GPOW_i71_e12(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 12) - 1;
    e <<= 5;
    uint64_t wm = (1 << w) - 1;
    fp_copy(t, gw[11][e & wm]); e >>= w;
    fp_mul(t, t, gw[12][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[13][e & wm]); //countM++;
}


static void GPOW_i72_e12(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 12) - 1;
    uint64_t wm = (1 << w) - 1;
    fp_copy(t, gw[12][e & wm]); e >>= w;
    fp_mul(t, t, gw[13][e & wm]); //countM++;
}


static void GPOW_i78_e12(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 12) - 1;
    uint64_t wm = (1 << w) - 1;
    fp_copy(t, gw[13][e & wm]); e >>= w;
    fp_mul(t, t, gw[14][e & wm]); //countM++;
}


static void GPOW_i83_e6(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 6) - 1;
    e <<= 5;
    uint64_t wm = (1 << w) - 1;
    fp_copy(t, gw[13][e & wm]); e >>= w;
    fp_mul(t, t, gw[14][e & wm]); //countM++;
}


static void GPOW_i84_e6(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 6) - 1;
    uint64_t wm = (1 << w) - 1;
    fp_copy(t, gw[14][e & wm]);
}


void DLPpow2ext(fp_t u_A, fp_t out_t,
                fp_t gw[nw][we], int rlll[1024], fp_t rll[we], fp_t fll[we], fp_t gpp[n])
{
    uint64_t one_k = 1;
    int _ii, _tmp;

    /* ---- All fp_t temporaries ---- */
    fp_t h0_A, hlp_A, f_A;
    fp_t h0_PA, hlp_PA;
    fp_t h0_PAH, hlp_PAH;
    fp_t h0_PAHH;
    fp_t h1_PAHH, u2_PAHH;
    fp_t h1_PAH, hlp1_PAH, u2_PAH;
    fp_t h0_PAHL;
    fp_t h1_PAHL, u2_PAHL;
    fp_t h1_PA, hlp1_PA, u2_PA;
    fp_t h0_PAL;
    fp_t h0_PALH;
    fp_t h1_PALH, u2_PALH;
    fp_t h1_PAL, u2_PAL;
    fp_t h0_PALL;
    fp_t h1_PALL, u2_PALL;
    fp_t hlp1_A, f_A_sq;
    fp_t u_X0, h0_X0;
    fp_t h0_X0P, hlp_X0P;
    fp_t h0_X0PH;
    fp_t h1_X0PH, u2_X0PH;
    fp_t h1_X0P, hlp1_X0P, u2_X0P;
    fp_t h0_X0PL;
    fp_t h1_X0PL, u2_X0PL;
    fp_t f_X0, f_X0_sq;
    fp_t u_X1, hlp_X1, h0_X1;
    fp_t h0_X1P;
    fp_t h1_X1P, u2_X1P;
    fp_t f_X1, hlp1_X1, f_X1_sq;
    fp_t u_X2, h0_X2;
    fp_t f_X2, f_X2_sq;
    fp_t u_X3, z_X3;
    fp_t t_X2, t_X1, t_X0, t_A;

    fp_null(h0_A);    fp_new(h0_A);
    fp_null(hlp_A);   fp_new(hlp_A);
    fp_null(f_A);     fp_new(f_A);
    fp_null(h0_PA);   fp_new(h0_PA);
    fp_null(hlp_PA);  fp_new(hlp_PA);
    fp_null(h0_PAH);  fp_new(h0_PAH);
    fp_null(hlp_PAH); fp_new(hlp_PAH);
    fp_null(h0_PAHH); fp_new(h0_PAHH);
    fp_null(h1_PAHH); fp_new(h1_PAHH);
    fp_null(u2_PAHH); fp_new(u2_PAHH);
    fp_null(h1_PAH);  fp_new(h1_PAH);
    fp_null(hlp1_PAH);fp_new(hlp1_PAH);
    fp_null(u2_PAH);  fp_new(u2_PAH);
    fp_null(h0_PAHL); fp_new(h0_PAHL);
    fp_null(h1_PAHL); fp_new(h1_PAHL);
    fp_null(u2_PAHL); fp_new(u2_PAHL);
    fp_null(h1_PA);   fp_new(h1_PA);
    fp_null(hlp1_PA); fp_new(hlp1_PA);
    fp_null(u2_PA);   fp_new(u2_PA);
    fp_null(h0_PAL);  fp_new(h0_PAL);
    fp_null(h0_PALH); fp_new(h0_PALH);
    fp_null(h1_PALH); fp_new(h1_PALH);
    fp_null(u2_PALH); fp_new(u2_PALH);
    fp_null(h1_PAL);  fp_new(h1_PAL);
    fp_null(u2_PAL);  fp_new(u2_PAL);
    fp_null(h0_PALL); fp_new(h0_PALL);
    fp_null(h1_PALL); fp_new(h1_PALL);
    fp_null(u2_PALL); fp_new(u2_PALL);
    fp_null(hlp1_A);  fp_new(hlp1_A);
    fp_null(f_A_sq);  fp_new(f_A_sq);
    fp_null(u_X0);    fp_new(u_X0);
    fp_null(h0_X0);   fp_new(h0_X0);
    fp_null(h0_X0P);  fp_new(h0_X0P);
    fp_null(hlp_X0P); fp_new(hlp_X0P);
    fp_null(h0_X0PH); fp_new(h0_X0PH);
    fp_null(h1_X0PH); fp_new(h1_X0PH);
    fp_null(u2_X0PH); fp_new(u2_X0PH);
    fp_null(h1_X0P);  fp_new(h1_X0P);
    fp_null(hlp1_X0P);fp_new(hlp1_X0P);
    fp_null(u2_X0P);  fp_new(u2_X0P);
    fp_null(h0_X0PL); fp_new(h0_X0PL);
    fp_null(h1_X0PL); fp_new(h1_X0PL);
    fp_null(u2_X0PL); fp_new(u2_X0PL);
    fp_null(f_X0);    fp_new(f_X0);
    fp_null(f_X0_sq); fp_new(f_X0_sq);
    fp_null(u_X1);    fp_new(u_X1);
    fp_null(hlp_X1);  fp_new(hlp_X1);
    fp_null(h0_X1);   fp_new(h0_X1);
    fp_null(h0_X1P);  fp_new(h0_X1P);
    fp_null(h1_X1P);  fp_new(h1_X1P);
    fp_null(u2_X1P);  fp_new(u2_X1P);
    fp_null(f_X1);    fp_new(f_X1);
    fp_null(hlp1_X1); fp_new(hlp1_X1);
    fp_null(f_X1_sq); fp_new(f_X1_sq);
    fp_null(u_X2);    fp_new(u_X2);
    fp_null(h0_X2);   fp_new(h0_X2);
    fp_null(f_X2);    fp_new(f_X2);
    fp_null(f_X2_sq); fp_new(f_X2_sq);
    fp_null(u_X3);    fp_new(u_X3);
    fp_null(z_X3);    fp_new(z_X3);
    fp_null(t_X2);    fp_new(t_X2);
    fp_null(t_X1);    fp_new(t_X1);
    fp_null(t_X0);    fp_new(t_X0);
    fp_null(t_A);     fp_new(t_A);

   

    fp_copy(h0_A, u_A);
    for (int _j = 0; _j < 48; _j++) {
        if (_j == 24) fp_copy(hlp_A, h0_A);
        fp_sqr(h0_A, h0_A);
    }
    //countS += 48;


    
    fp_copy(h0_PA, h0_A);
    for (int _j = 0; _j < 24; _j++) {
        if (_j == 12) fp_copy(hlp_PA, h0_PA);
        fp_sqr(h0_PA, h0_PA);
    }
    //countS += 24;


    
    fp_copy(h0_PAH, h0_PA);
    for (int _j = 0; _j < 12; _j++) {
        if (_j == 6) fp_copy(hlp_PAH, h0_PAH);
        fp_sqr(h0_PAH, h0_PAH);
    }
    //countS += 12;


    
    fp_copy(h0_PAHH, h0_PAH);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_PAHH, h0_PAHH);
    //countS += 6;
    
	bn_t tmp_bn;
	dig_t d;
	bn_null(tmp_bn);bn_new(tmp_bn);
    fp_prime_back(tmp_bn, h0_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3ff;
    _tmp=rlll[d];

    uint64_t c_PAH = (uint64_t)_tmp;   

    
    GPOW_i84_e6(h1_PAHH, (one_k << 6) - c_PAH, gw);
    SELECT(h1_PAHH, gpp[84 + 6], c_PAH == 0, h1_PAHH);
    
    //countM++;
    fp_mul(u2_PAHH, h0_PAH, h1_PAHH);   
    
    fp_prime_back(tmp_bn, u2_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3ff;
    _tmp=rlll[d];

    uint64_t d_PAH = (uint64_t)_tmp;   
    c_PAH = c_PAH + ((d_PAH - 1) % 64) * 64;  

    
    GPOW_i72_e12(h1_PAH,   (one_k << 12) - c_PAH, gw);
    SELECT(h1_PAH,   gpp[72 + 12], c_PAH == 0, h1_PAH);
    GPOW_i78_e12(hlp1_PAH, (one_k << 12) - c_PAH, gw);
    SELECT(hlp1_PAH, gpp[78 + 12], c_PAH == 0, hlp1_PAH);
    fp_mul(hlp_PAH, hlp_PAH, hlp1_PAH); //countM++;
    //countM++;
    fp_mul(u2_PAH, h0_PA, h1_PAH);   

    
    fp_copy(h0_PAHL, hlp_PAH);   
    /* hlp_PAHL = None */
    fp_prime_back(tmp_bn, h0_PAHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3ff;
    _tmp=rlll[d];

    uint64_t c_PAHL = (uint64_t)_tmp;   

    GPOW_i84_e6(h1_PAHL, (one_k << 6) - c_PAHL, gw);
    SELECT(h1_PAHL, gpp[84 + 6], c_PAHL == 0, h1_PAHL);
    
    //countM++;
    fp_mul(u2_PAHL, u2_PAH, h1_PAHL);   
    fp_prime_back(tmp_bn, u2_PAHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3ff;
    _tmp=rlll[d];

    uint64_t d_PAHL = (uint64_t)_tmp;   /* >> 0 */
    uint64_t d_PA   = c_PAHL + ((d_PAHL - 1) % 64) * 64;
    uint64_t c_PA   = c_PAH  + ((d_PA   - 1) % ((one_k << 12))) * (one_k << 12);
    

    
    GPOW_i48_e24(h1_PA,   (one_k << 24) - c_PA, gw);
    SELECT(h1_PA,   gpp[48 + 24], c_PA == 0, h1_PA);
    GPOW_i60_e24(hlp1_PA, (one_k << 24) - c_PA, gw);
    SELECT(hlp1_PA, gpp[60 + 24], c_PA == 0, hlp1_PA);
    fp_mul(hlp_PA, hlp_PA, hlp1_PA); //countM++;
    //countM++;
    fp_mul(u2_PA, h0_A, h1_PA);   

   
    fp_copy(h0_PAL, hlp_PA);   
    

  
    fp_copy(h0_PALH, h0_PAL);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_PALH, h0_PALH);
    //countS += 6;
    fp_prime_back(tmp_bn, h0_PALH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3ff;
    _tmp=rlll[d];

    uint64_t c_PALH = (uint64_t)_tmp;   

    GPOW_i84_e6(h1_PALH, (one_k << 6) - c_PALH, gw);
    SELECT(h1_PALH, gpp[84 + 6], c_PALH == 0, h1_PALH);
    
    //countM++;
    fp_mul(u2_PALH, h0_PAL, h1_PALH);   
    fp_prime_back(tmp_bn, u2_PALH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3ff;
    _tmp=rlll[d];

    uint64_t d_PALH = (uint64_t)_tmp;   /* >> 0 */
    uint64_t c_PAL  = c_PALH + ((d_PALH - 1) % 64) * 64;

   
    GPOW_i72_e12(h1_PAL, (one_k << 12) - c_PAL, gw);
    SELECT(h1_PAL, gpp[72 + 12], c_PAL == 0, h1_PAL);
    
    //countM++;
    fp_mul(u2_PAL, u2_PA, h1_PAL);   

    
    fp_copy(h0_PALL, u2_PAL);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_PALL, h0_PALL);
    //countS += 6;
    fp_prime_back(tmp_bn, h0_PALL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3ff;
    _tmp=rlll[d];

    uint64_t c_PALL = (uint64_t)_tmp;   /* >> 0 */

    GPOW_i84_e6(h1_PALL, (one_k << 6) - c_PALL, gw);
    SELECT(h1_PALL, gpp[84 + 6], c_PALL == 0, h1_PALL);
    
    //countM++;
    fp_mul(u2_PALL, u2_PAL, h1_PALL);   
    fp_prime_back(tmp_bn, u2_PALL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3ff;
    _tmp=rlll[d];

    uint64_t d_PALL = (uint64_t)_tmp;   
    uint64_t d_PAL  = c_PALL + ((d_PALL - 1) % 64) * 64;
    uint64_t d_PA2  = c_PAL  + ((d_PAL  - 1) % ((one_k << 12))) * (one_k << 12);
    uint64_t c_A    = c_PA   + ((d_PA2  - 1) % ((one_k << 24))) * (one_k << 24);

    
    GPOW_i0_e47(f_A,    ((one_k << 48) - c_A + 1) >> 1, gw);
    SELECT(f_A,    gpp[47],        c_A == 0, f_A);
    GPOW_i24_e48(hlp1_A, (one_k << 48) - c_A, gw);
    SELECT(hlp1_A, gpp[24 + 48], c_A == 0, hlp1_A);
    fp_mul(hlp_A, hlp_A, hlp1_A); //countM++;
    //countM++;
    //countS++;

    
    fp_sqr(f_A_sq, f_A);
    fp_mul(u_X0, u_A, f_A_sq);
    fp_copy(h0_X0, hlp_A);   
    

    
    fp_copy(h0_X0P, h0_X0);
    for (int _j = 0; _j < 12; _j++) {
        if (_j == 6) fp_copy(hlp_X0P, h0_X0P);
        fp_sqr(h0_X0P, h0_X0P);
    }
    //countS += 12;

    
    fp_copy(h0_X0PH, h0_X0P);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_X0PH, h0_X0PH);
    //countS += 6;
    fp_prime_back(tmp_bn, h0_X0PH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3ff;
    _tmp=rlll[d];

    uint64_t c_X0P = (uint64_t)_tmp;   /* >> 0 */

    GPOW_i84_e6(h1_X0PH, (one_k << 6) - c_X0P, gw);
    SELECT(h1_X0PH, gpp[84 + 6], c_X0P == 0, h1_X0PH);
    
    //countM++;
    fp_mul(u2_X0PH, h0_X0P, h1_X0PH);   /* u_X0PH = h0_X0P */
    fp_prime_back(tmp_bn, u2_X0PH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3ff;
    _tmp=rlll[d];

    uint64_t d_X0PH = (uint64_t)_tmp;   /* >> 0 */
    c_X0P = c_X0P + ((d_X0PH - 1) % 64) * 64;

    
    GPOW_i72_e12(h1_X0P,   (one_k << 12) - c_X0P, gw);
    SELECT(h1_X0P,   gpp[72 + 12], c_X0P == 0, h1_X0P);
    GPOW_i78_e12(hlp1_X0P, (one_k << 12) - c_X0P, gw);
    SELECT(hlp1_X0P, gpp[78 + 12], c_X0P == 0, hlp1_X0P);
    fp_mul(hlp_X0P, hlp_X0P, hlp1_X0P); //countM++;
    //countM++;
    fp_mul(u2_X0P, h0_X0, h1_X0P);   

    
    fp_copy(h0_X0PL, hlp_X0P);  
   
    fp_prime_back(tmp_bn, h0_X0PL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3ff;
    _tmp=rlll[d];

    uint64_t c_X0PL = (uint64_t)_tmp;   /* >> 0 */

    GPOW_i84_e6(h1_X0PL, (one_k << 6) - c_X0PL, gw);
    SELECT(h1_X0PL, gpp[84 + 6], c_X0PL == 0, h1_X0PL);
    
    //countM++;
    fp_mul(u2_X0PL, u2_X0P, h1_X0PL);   
    fp_prime_back(tmp_bn, u2_X0PL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3ff;
    _tmp=rlll[d];

    uint64_t d_X0PL = (uint64_t)_tmp;   
    uint64_t d_X0P  = c_X0PL + ((d_X0PL - 1) % 64)             * 64;
    uint64_t c_X0   = c_X0P  + ((d_X0P  - 1) % ((one_k << 12))) * (one_k << 12);

    
    GPOW_i47_e24(f_X0, (one_k << 24) - c_X0, gw);
    SELECT(f_X0, gpp[47 + 24], c_X0 == 0, f_X0);

    //countM++;
    //countS++;

    
    fp_sqr(f_X0_sq, f_X0);
    fp_mul(u_X1, u_X0, f_X0_sq);
    
    fp_copy(h0_X1, u_X1);
    for (int _j = 0; _j < 12; _j++) {
        if (_j == 6) fp_copy(hlp_X1, h0_X1);
        fp_sqr(h0_X1, h0_X1);
    }
    //countS += 12;

    fp_copy(h0_X1P, h0_X1);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_X1P, h0_X1P);
    //countS += 6;
    fp_prime_back(tmp_bn, h0_X1P);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3ff;
    _tmp=rlll[d];

    uint64_t c_X1 = (uint64_t)_tmp;  

    GPOW_i84_e6(h1_X1P, (one_k << 6) - c_X1, gw);
    SELECT(h1_X1P, gpp[84 + 6], c_X1 == 0, h1_X1P);
    
    //countM++;
    fp_mul(u2_X1P, h0_X1, h1_X1P);  
    fp_prime_back(tmp_bn, u2_X1P);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3ff;
    _tmp=rlll[d];

    uint64_t d_X1P = (uint64_t)_tmp;   
    c_X1 = c_X1 + ((d_X1P - 1) % 64) * 64;

    
    GPOW_i71_e12(f_X1,   (one_k << 12) - c_X1, gw);
    SELECT(f_X1,   gpp[71 + 12], c_X1 == 0, f_X1);
    GPOW_i78_e12(hlp1_X1, (one_k << 12) - c_X1, gw);
    SELECT(hlp1_X1, gpp[78 + 12], c_X1 == 0, hlp1_X1);
    fp_mul(hlp_X1, hlp_X1, hlp1_X1); //countM++;
    //countM++;
    //countS++;

    
    fp_sqr(f_X1_sq, f_X1);
    fp_mul(u_X2, u_X1, f_X1_sq);
    fp_copy(h0_X2, hlp_X1);   
    fp_prime_back(tmp_bn, h0_X2);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3ff;
    _tmp=rlll[d];

    uint64_t c_X2 = (uint64_t)_tmp;   /* >> (w-lb) = >> 0 */

    
    GPOW_i83_e6(f_X2, (one_k << 6) - c_X2, gw);
    SELECT(f_X2, gpp[83 + 6], c_X2 == 0, f_X2);
   
    //countM++;
    //countS++;

    
    fp_sqr(f_X2_sq, f_X2);
    fp_mul(u_X3, u_X2, f_X2_sq);
    fp_prime_back(tmp_bn, u_X3);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3ff;
    _tmp=rlll[d];

    uint64_t c1_X3 = (uint64_t)_tmp;   
    fp_copy(z_X3, fll[c1_X3]);


    
    fp_mul(t_X2, f_X2, z_X3); //countM++;


    fp_mul(t_X1, f_X1, t_X2); //countM++;


    fp_mul(t_X0, f_X0, t_X1); //countM++;


    fp_mul(out_t,  f_A,  t_X0); //countM++;


    /* ---- free all ---- */
    fp_free(h0_A);    fp_free(hlp_A);    fp_free(f_A);
    fp_free(h0_PA);   fp_free(hlp_PA);
    fp_free(h0_PAH);  fp_free(hlp_PAH);
    fp_free(h0_PAHH); fp_free(h1_PAHH);  fp_free(u2_PAHH);
    fp_free(h1_PAH);  fp_free(hlp1_PAH); fp_free(u2_PAH);
    fp_free(h0_PAHL); fp_free(h1_PAHL);  fp_free(u2_PAHL);
    fp_free(h1_PA);   fp_free(hlp1_PA);  fp_free(u2_PA);
    fp_free(h0_PAL);
    fp_free(h0_PALH); fp_free(h1_PALH);  fp_free(u2_PALH);
    fp_free(h1_PAL);  fp_free(u2_PAL);
    fp_free(h0_PALL); fp_free(h1_PALL);  fp_free(u2_PALL);
    fp_free(hlp1_A);  fp_free(f_A_sq);
    fp_free(u_X0);    fp_free(h0_X0);
    fp_free(h0_X0P);  fp_free(hlp_X0P);
    fp_free(h0_X0PH); fp_free(h1_X0PH);  fp_free(u2_X0PH);
    fp_free(h1_X0P);  fp_free(hlp1_X0P); fp_free(u2_X0P);
    fp_free(h0_X0PL); fp_free(h1_X0PL);  fp_free(u2_X0PL);
    fp_free(f_X0);    fp_free(f_X0_sq);
    fp_free(u_X1);    fp_free(hlp_X1);   fp_free(h0_X1);
    fp_free(h0_X1P);  fp_free(h1_X1P);   fp_free(u2_X1P);
    fp_free(f_X1);    fp_free(hlp1_X1);  fp_free(f_X1_sq);
    fp_free(u_X2);    fp_free(h0_X2);
    fp_free(f_X2);    fp_free(f_X2_sq);
    fp_free(u_X3);    fp_free(z_X3);
    fp_free(t_X2);    fp_free(t_X1);     fp_free(t_X0);    fp_free(t_A);
}

/* =========================================================================
 * sqrt_ext
 * ========================================================================= */
void sqrt_ext(fp_t x, fp_t y, bn_t e_exp,
              fp_t gw[nw][we], int rlll[1024], fp_t rll[we], fp_t fll[we], fp_t gpp[n])
{
    fp_t u, v, w_, t;
    fp_null(u); fp_null(v); fp_null(w_); fp_null(t);
    RLC_TRY {
        fp_new(u); fp_new(v); fp_new(w_); fp_new(t);
        fp_exp(v,  x, e_exp);
        fp_mul(w_, x, v);  //countM++;
        fp_mul(u,  w_, v); //countM++;
        DLPpow2ext(u, t, gw, rlll, rll, fll, gpp);
        fp_mul(y, w_, t);  //countM++;
        //fp_print(x);
        //fp_sqr(y,y);
        //fp_print(y);
    }
    RLC_CATCH_ANY { RLC_THROW(ERR_CAUGHT); }
    RLC_FINALLY   { fp_free(u); fp_free(v); fp_free(w_); fp_free(t); }
}

/* =========================================================================
 * main
 * ========================================================================= */
void main(void)
{
    if (core_init() != RLC_OK) { core_clean(); return; }
    if (fp_param_set_any_pmers() != RLC_OK) { core_clean(); return; }

    int i, j;
    fp_t gw[nw][we], rll[we], fll[we], gpp[n], g, z, b, h, hh, y;
    bn_t tmp, m, e;
	int rlll[1024]={0};

rlll[ 0x1 ]= 0 ;
rlll[ 0x34b ]= 1 ;
rlll[ 0x2d3 ]= 2 ;
rlll[ 0x3e0 ]= 3 ;
rlll[ 0x32f ]= 4 ;
rlll[ 0x2a5 ]= 5 ;
rlll[ 0x2b3 ]= 6 ;
rlll[ 0x2c5 ]= 7 ;
rlll[ 0x1bd ]= 8 ;
rlll[ 0x5 ]= 9 ;
rlll[ 0x362 ]= 10 ;
rlll[ 0x222 ]= 11 ;
rlll[ 0x1b0 ]= 12 ;
rlll[ 0x337 ]= 13 ;
rlll[ 0x37c ]= 14 ;
rlll[ 0x2 ]= 15 ;
rlll[ 0x219 ]= 16 ;
rlll[ 0x38b ]= 17 ;
rlll[ 0x5e ]= 18 ;
rlll[ 0x287 ]= 19 ;
rlll[ 0xb8 ]= 20 ;
rlll[ 0x263 ]= 21 ;
rlll[ 0x10a ]= 22 ;
rlll[ 0x383 ]= 23 ;
rlll[ 0x109 ]= 24 ;
rlll[ 0x269 ]= 25 ;
rlll[ 0x296 ]= 26 ;
rlll[ 0x365 ]= 27 ;
rlll[ 0xac ]= 28 ;
rlll[ 0x272 ]= 29 ;
rlll[ 0x301 ]= 30 ;
rlll[ 0xd3 ]= 31 ;
rlll[ 0x0 ]= 32 ;
rlll[ 0xb6 ]= 33 ;
rlll[ 0x12e ]= 34 ;
rlll[ 0x21 ]= 35 ;
rlll[ 0xd2 ]= 36 ;
rlll[ 0x15c ]= 37 ;
rlll[ 0x14e ]= 38 ;
rlll[ 0x13c ]= 39 ;
rlll[ 0x244 ]= 40 ;
rlll[ 0x3fc ]= 41 ;
rlll[ 0x9f ]= 42 ;
rlll[ 0x1df ]= 43 ;
rlll[ 0x251 ]= 44 ;
rlll[ 0xca ]= 45 ;
rlll[ 0x85 ]= 46 ;
rlll[ 0x3ff ]= 47 ;
rlll[ 0x1e8 ]= 48 ;
rlll[ 0x76 ]= 49 ;
rlll[ 0x3a3 ]= 50 ;
rlll[ 0x17a ]= 51 ;
rlll[ 0x349 ]= 52 ;
rlll[ 0x19e ]= 53 ;
rlll[ 0x2f7 ]= 54 ;
rlll[ 0x7e ]= 55 ;
rlll[ 0x2f8 ]= 56 ;
rlll[ 0x198 ]= 57 ;
rlll[ 0x16b ]= 58 ;
rlll[ 0x9c ]= 59 ;
rlll[ 0x355 ]= 60 ;
rlll[ 0x18f ]= 61 ;
rlll[ 0x100 ]= 62 ;
rlll[ 0x32e ]= 63 ;

    for (i = 0; i < nw; i++)
        for (j = 0; j < we; j++) { fp_null(gw[i][j]); fp_new(gw[i][j]); }
    for (i = 0; i < we; i++) { fp_null(rll[i]); fp_new(rll[i]); }
    for (i = 0; i < we; i++) { fp_null(fll[i]); fp_new(fll[i]); }
    for (i = 0; i < n;  i++) { fp_null(gpp[i]); fp_new(gpp[i]); }
    fp_null(b);  fp_new(b);  fp_null(y);  fp_new(y);
    fp_null(g);  fp_new(g);  fp_null(z);  fp_new(z);
    fp_null(h);  fp_new(h);  fp_null(hh); fp_new(hh);
    bn_null(tmp); bn_new(tmp);
    bn_null(m);   bn_new(m);
    bn_null(e);   bn_new(e);

    fp_rand(b);
    while (fp_is_sqr(b) != 1) fp_rand(b);

//    bn_read_str(tmp, "4", 1, 16);
//    fp_prime_conv(b, tmp);

    bn_read_str(tmp, "b", 1, 16);
    fp_prime_conv(z, tmp);

    bn_read_str(m, "ffffffffffffffffffffffffffffffff", 32, 16);
    fp_exp(g, z, m);

    bn_read_str(e, "7fffffffffffffffffffffffffffffff", 32, 16);

    bn_read_str(tmp, "40000000000000000000000", 24, 16);
    fp_exp(h, g, tmp);

    fp_srt(hh, h);
    fp_inv(hh, hh);
//    fp_print(h);
//    fp_print(hh);

    precomputation(g, h, hh, gw, rll, fll, gpp);

    MEASURE(sqrt_ext(b, y, e, gw, rlll, rll, fll, gpp);)

    printf("RDTSC_clk_min=%f\n",    RDTSC_clk_min);
    printf("RDTSC_clk_median=%f\n", RDTSC_clk_median);
    printf("RDTSC_clk_max=%f\n",    RDTSC_clk_max);
    printf("mult_count=%d\n",  countM);
    printf("sqr_count=%d\n",   countS);

    for (i = 0; i < nw; i++)
        for (j = 0; j < we; j++) fp_free(gw[i][j]);
    for (i = 0; i < we; i++) fp_free(rll[i]);
    for (i = 0; i < we; i++) fp_free(fll[i]);
    for (i = 0; i < n;  i++) fp_free(gpp[i]);
    fp_free(b); fp_free(y); fp_free(g); fp_free(z); fp_free(h); fp_free(hh);
    bn_free(tmp); bn_free(e); bn_free(m);
}
