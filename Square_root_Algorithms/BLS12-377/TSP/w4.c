#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include <immintrin.h>
#include <relic/relic_core.h>
#include "relic/relic_md.h"

#define w   4
#define we  16          
#define n   46
#define nw  12          



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
                    fp_t gw[nw][we], fp_t fll[we], fp_t gpp[n])
{
    bn_t temp, one;
    bn_null(temp); bn_new(temp);
    bn_null(one);  bn_new(one);
    bn_set_dig(one, 1);
    for (int v = 0; v < we; v++) {
        bn_set_dig(temp, v);
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


static void GPOW_i41_e2(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 2) - 1;
    e <<= 1;
    fp_copy(t, gw[10][e & 0xF]);
}


static void GPOW_i39_e3(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 3) - 1;
    e <<= 3;
    fp_copy(t, gw[9][e & 0xF]); e >>= 4;
    fp_mul(t, t, gw[10][e & 0xF]); 
}


static void GPOW_i40_e3(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 3) - 1;
    fp_copy(t, gw[10][e & 0xF]);
}


static void GPOW_i35_e5(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 5) - 1;
    e <<= 3;
    fp_copy(t, gw[8][e & 0xF]); e >>= 4;
    fp_mul(t, t, gw[9][e & 0xF]); //countM++;
}


static void GPOW_i38_e5(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 5) - 1;
    e <<= 2;
    fp_copy(t, gw[9][e & 0xF]); e >>= 4;
    fp_mul(t, t, gw[10][e & 0xF]); //countM++;
}


static void GPOW_i34_e6(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 6) - 1;
    e <<= 2;
    fp_copy(t, gw[8][e & 0xF]); e >>= 4;
    fp_mul(t, t, gw[9][e & 0xF]); //countM++;
}


static void GPOW_i37_e6(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 6) - 1;
    e <<= 1;
    fp_copy(t, gw[9][e & 0xF]); e >>= 4;
    fp_mul(t, t, gw[10][e & 0xF]); //countM++;
}


static void GPOW_i33_e6(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 6) - 1;
    e <<= 1;
    fp_copy(t, gw[8][e & 0xF]); e >>= 4;
    fp_mul(t, t, gw[9][e & 0xF]); //countM++;
}


static void GPOW_i23_e11(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 11) - 1;
    e <<= 3;
    fp_copy(t, gw[5][e & 0xF]); e >>= 4;
    fp_mul(t, t, gw[6][e & 0xF]);
    e >>= 4;
    fp_mul(t, t, gw[7][e & 0xF]);
    e >>= 4;
    fp_mul(t, t, gw[8][e & 0xF]);
}


static void GPOW_i29_e11(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 11) - 1;
    e <<= 1;
    fp_copy(t, gw[7][e & 0xF]); e >>= 4;
    fp_mul(t, t, gw[8][e & 0xF]);
    e >>= 4;
    fp_mul(t, t, gw[9][e & 0xF]);
}


static void GPOW_i0_e22(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 22) - 1;
    fp_copy(t, gw[0][e & 0xF]); e >>= 4;
    fp_mul(t, t, gw[1][e & 0xF]); //countM++; 
    e >>= 4;
    fp_mul(t, t, gw[2][e & 0xF]); //countM++; 
    e >>= 4;
    fp_mul(t, t, gw[3][e & 0xF]); //countM++; 
    e >>= 4;
    fp_mul(t, t, gw[4][e & 0xF]); //countM++; 
    e >>= 4;
    fp_mul(t, t, gw[5][e & 0xF]); //countM++;
}


static void GPOW_i12_e23(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 23) - 1;
    fp_copy(t, gw[3][e & 0xF]); e >>= 4;
    fp_mul(t, t, gw[4][e & 0xF]); //countM++; 
    e >>= 4;
    fp_mul(t, t, gw[5][e & 0xF]); //countM++; 
    e >>= 4;
    fp_mul(t, t, gw[6][e & 0xF]); //countM++; 
    e >>= 4;
    fp_mul(t, t, gw[7][e & 0xF]); //countM++; 
    e >>= 4;
    fp_mul(t, t, gw[8][e & 0xF]); //countM++;
}


static void GPOW_i22_e11(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 11) - 1;
    e <<= 2;
    fp_copy(t, gw[5][e & 0xF]); e >>= 4;
    fp_mul(t, t, gw[6][e & 0xF]); //countM++; 
    e >>= 4;
    fp_mul(t, t, gw[7][e & 0xF]); //countM++; 
    e >>= 4;
    fp_mul(t, t, gw[8][e & 0xF]); //countM++;
}


void DLPpow2ext(fp_t u_A, fp_t out_t,
                fp_t gw[nw][we], fp_t fll[we], fp_t gpp[n], int tab2[16])
{
    const uint64_t one_k = 1;
    int _ii;

    
    fp_t h0_A, hlp_A;
    fp_t h0_PA, hlp_PA;
    fp_t h0_PAH, hlp_PAH;
    fp_t h0_PAHH, h1_PAHH, u2_PAHH;
    fp_t h1_PAH, hlp1_PAH, u2_PAH;
    fp_t h0_PAHL, f_PAHL, u2_PAHL, d_PAHL;
    fp_t h1_PA, hlp1_PA, u2_PA;
    fp_t h0_PAL;
    fp_t h0_PALH, h1_PALH, u2_PALH;
    fp_t f_PAL, h1_PAL;
    fp_t h0_PALL, f_PALL, u2_PALL, d_PALL, d_PAL;
    fp_t f_A, hlp1_A, f_A_sq, h1_A;
    fp_t h0_X;
    fp_t h0_XH, hlp_XH;
    fp_t h0_XHH, h1_XHH, u2_XHH;
    fp_t h1_XH, hlp1_XH, u2_XH;
    fp_t h0_XHL, f_XHL, u2_XHL, d_XHL;
    fp_t f_X, f_X_sq, h1_X;
    fp_t h0_XL, hlp_XL;
    fp_t h0_XLH, h1_XLH, u2_XLH;
    fp_t h1_XL_h, hlp1_XL, u2_XL_h;
    fp_t f_XL, f_XL_sq, h1_XL_product;
    fp_t h0_XLL, f_XLL, u2_XLL, d_XLL, d_XL;
    fp_t d_X, t_A;

#define FN(x) fp_null(x); fp_new(x)
    FN(h0_A); FN(hlp_A);
    FN(h0_PA); FN(hlp_PA);
    FN(h0_PAH); FN(hlp_PAH);
    FN(h0_PAHH); FN(h1_PAHH); FN(u2_PAHH);
    FN(h1_PAH); FN(hlp1_PAH); FN(u2_PAH);
    FN(h0_PAHL); FN(f_PAHL); FN(u2_PAHL); FN(d_PAHL);
    FN(h1_PA); FN(hlp1_PA); FN(u2_PA);
    FN(h0_PAL);
    FN(h0_PALH); FN(h1_PALH); FN(u2_PALH);
    FN(f_PAL); FN(h1_PAL);
    FN(h0_PALL); FN(f_PALL); FN(u2_PALL); FN(d_PALL); FN(d_PAL);
    FN(f_A); FN(hlp1_A); FN(f_A_sq); FN(h1_A);
    FN(h0_X);
    FN(h0_XH); FN(hlp_XH);
    FN(h0_XHH); FN(h1_XHH); FN(u2_XHH);
    FN(h1_XH); FN(hlp1_XH); FN(u2_XH);
    FN(h0_XHL); FN(f_XHL); FN(u2_XHL); FN(d_XHL);
    FN(f_X); FN(f_X_sq); FN(h1_X);
    FN(h0_XL); FN(hlp_XL);
    FN(h0_XLH); FN(h1_XLH); FN(u2_XLH);
    FN(h1_XL_h); FN(hlp1_XL); FN(u2_XL_h);
    FN(f_XL); FN(f_XL_sq); FN(h1_XL_product);
    FN(h0_XLL); FN(f_XLL); FN(u2_XLL); FN(d_XLL); FN(d_XL);
    FN(d_X); FN(t_A);
#undef FN

    
    fp_copy(h0_A, u_A);
    for (int _j = 0; _j < 23; _j++) {
        if (_j == 12) fp_copy(hlp_A, h0_A);
        fp_sqr(h0_A, h0_A);
    }
    //countS += 23;

    
    fp_copy(h0_PA, h0_A);
    for (int _j = 0; _j < 12; _j++) {
        if (_j == 6) fp_copy(hlp_PA, h0_PA);
        fp_sqr(h0_PA, h0_PA);
    }
    //countS += 12;

    
    fp_copy(h0_PAH, h0_PA);
    for (int _j = 0; _j < 6; _j++) {
        if (_j == 3) fp_copy(hlp_PAH, h0_PAH);
        fp_sqr(h0_PAH, h0_PAH);
    }
    //countS += 6;

    
    fp_copy(h0_PAHH, h0_PAH);
    for (int _j = 0; _j < 3; _j++) fp_sqr(h0_PAHH, h0_PAHH);
    //countS += 3;

    
    bn_t tmp_bn;
    dig_t d;
    bn_null(tmp_bn);bn_new(tmp_bn);
    
    
    fp_prime_back(tmp_bn, h0_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    uint64_t c_PAHH = (uint64_t)tab2[d] >> 2;

    
    GPOW_i41_e2(h1_PAHH, (one_k << 2) - c_PAHH, gw);
    SELECT(h1_PAHH, gpp[43], c_PAHH == 0, h1_PAHH);
    fp_mul(u2_PAHH, h0_PAH, h1_PAHH); //countM++;

    
    fp_prime_back(tmp_bn, u2_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    uint64_t e1_PAHH  = (uint64_t)tab2[d] >> 1;
    

    uint64_t e_PAHH    = (e1_PAHH - 1) & 7ULL;
    uint64_t c_PAH_val = c_PAHH + (e_PAHH << 2);  

    
    GPOW_i35_e5(h1_PAH,   (one_k << 5) - c_PAH_val, gw);
    SELECT(h1_PAH,   gpp[40], c_PAH_val == 0, h1_PAH);
    GPOW_i38_e5(hlp1_PAH, (one_k << 5) - c_PAH_val, gw);
    SELECT(hlp1_PAH, gpp[43], c_PAH_val == 0, hlp1_PAH);
    fp_mul(hlp_PAH, hlp_PAH, hlp1_PAH); //countM++;
    //countM++;
    fp_mul(u2_PAH, h0_PA, h1_PAH);

    
    fp_copy(h0_PAHL, hlp_PAH);

    
    fp_prime_back(tmp_bn, h0_PAHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    uint64_t c_PAHL = (uint64_t)tab2[d]>> 1;

    
    GPOW_i39_e3(f_PAHL, (one_k << 3) - c_PAHL, gw);
    SELECT(f_PAHL, gpp[42], c_PAHL == 0, f_PAHL); //countM++;
    fp_sqr(u2_PAHL, f_PAHL); fp_mul(u2_PAHL, u2_PAHL, u2_PAH); //countS++;

    
    fp_prime_back(tmp_bn, u2_PAHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    uint64_t e1_PAHL = (uint64_t)tab2[d] >> 1;
    fp_copy(d_PAHL, fll[e1_PAHL << 1]);
    fp_mul(d_PAHL, d_PAHL, f_PAHL); //countM++;

    uint64_t e_PAHL      = (e1_PAHL - 1) & 7ULL;
    uint64_t e1_PAH_node = c_PAHL + (e_PAHL << 3);   

    
    uint64_t e_PAH_adj = (e1_PAH_node - 1) & 63ULL;
    uint64_t c_PA      = c_PAH_val + (e_PAH_adj << 5);

    
    GPOW_i23_e11(h1_PA,   (one_k << 11) - c_PA, gw);
    SELECT(h1_PA,   gpp[34], c_PA == 0, h1_PA);
    GPOW_i29_e11(hlp1_PA, (one_k << 11) - c_PA, gw);
    SELECT(hlp1_PA, gpp[40], c_PA == 0, hlp1_PA);
    fp_mul(hlp_PA, hlp_PA, hlp1_PA); //countM++;
    //countM++;
    fp_mul(u2_PA, h0_A, h1_PA);

   
    fp_copy(h0_PAL, hlp_PA);

    
    fp_copy(h0_PALH, h0_PAL);
    for (int _j = 0; _j < 3; _j++) fp_sqr(h0_PALH, h0_PALH);
    //countS += 3;

    
    fp_prime_back(tmp_bn, h0_PALH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    uint64_t c_PALH = (uint64_t)tab2[d] >> 1;

    
    GPOW_i40_e3(h1_PALH, (one_k << 3) - c_PALH, gw);
    SELECT(h1_PALH, gpp[43], c_PALH == 0, h1_PALH);
    fp_mul(u2_PALH, h0_PAL, h1_PALH); //countM++;

    
    fp_prime_back(tmp_bn, u2_PALH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    uint64_t e1_PALH  = (uint64_t)tab2[d] >> 1;
    uint64_t e_PALH   = (e1_PALH - 1) & 7ULL;
    uint64_t c_PAL_val = c_PALH + (e_PALH << 3);   /* 6-bit */

    
    GPOW_i33_e6(f_PAL, (one_k << 6) - c_PAL_val, gw);
    SELECT(f_PAL, gpp[39], c_PAL_val == 0, f_PAL); //countM++;
    fp_sqr(h1_PAL, f_PAL); fp_mul(h1_PAL, h1_PAL, u2_PA); //countS++;

    
    fp_copy(h0_PALL, h1_PAL);
    for (int _j = 0; _j < 3; _j++) fp_sqr(h0_PALL, h0_PALL);
    //countS += 3;

    
    fp_prime_back(tmp_bn, h0_PALL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    uint64_t c_PALL = (uint64_t)tab2[d] >> 1;

    
    GPOW_i39_e3(f_PALL, (one_k << 3) - c_PALL, gw);
    SELECT(f_PALL, gpp[42], c_PALL == 0, f_PALL); //countM++;
    fp_sqr(u2_PALL, f_PALL); fp_mul(u2_PALL, u2_PALL, h1_PAL); //countS++;

    
    fp_prime_back(tmp_bn, u2_PALL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    uint64_t e1_PALL = (uint64_t)tab2[d]>> 1;
    fp_copy(d_PALL, fll[e1_PALL << 1]);
    fp_mul(d_PALL, d_PALL, f_PALL); //countM++;

    uint64_t e_PALL      = (e1_PALL - 1) & 7ULL;
    uint64_t e1_PAL_node = c_PALL + (e_PALL << 3);   /* 6-bit */

    fp_mul(d_PAL, d_PALL, f_PAL); //countM++;

    
    uint64_t e_PAL_adj   = (e1_PAL_node - 1) & 63ULL;
    uint64_t e1_PA_node  = c_PAL_val + (e_PAL_adj << 6);

    
    uint64_t e_PA_adj = (e1_PA_node - 1) & 4095ULL;
    uint64_t e0_A     = c_PA + (e_PA_adj << 11);

    
    GPOW_i0_e22(f_A,    ((one_k << 23) - e0_A + 1) >> 1, gw);
    SELECT(f_A,    gpp[22], e0_A == 0, f_A); //countM++;
    fp_sqr(f_A_sq, f_A); fp_mul(h1_A, f_A_sq, u_A); //countS++;

    GPOW_i12_e23(hlp1_A, (one_k << 23) - e0_A, gw);
    SELECT(hlp1_A, gpp[35], e0_A == 0, hlp1_A);
    fp_mul(hlp_A, hlp_A, hlp1_A); //countM++;
    //countM++;

    
    fp_copy(h0_X, hlp_A);

    
    fp_copy(h0_XH, h0_X);
    for (int _j = 0; _j < 6; _j++) {
        if (_j == 3) fp_copy(hlp_XH, h0_XH);
        fp_sqr(h0_XH, h0_XH);
    }
    //countS += 6;

    
    fp_copy(h0_XHH, h0_XH);
    for (int _j = 0; _j < 3; _j++) fp_sqr(h0_XHH, h0_XHH);
    //countS += 3;

    
    fp_prime_back(tmp_bn, h0_XHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    uint64_t c_XHH = (uint64_t)tab2[d] >> 2;

    
    GPOW_i41_e2(h1_XHH, (one_k << 2) - c_XHH, gw);
    SELECT(h1_XHH, gpp[43], c_XHH == 0, h1_XHH);
    fp_mul(u2_XHH, h0_XH, h1_XHH); //countM++;

    
    fp_prime_back(tmp_bn, u2_XHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    uint64_t e1_XHH  = (uint64_t)tab2[d] >> 1;
    uint64_t e_XHH   = (e1_XHH - 1) & 7ULL;
    uint64_t c_XH_val = c_XHH + (e_XHH << 2);   /* 5-bit */

    
    GPOW_i35_e5(h1_XH,   (one_k << 5) - c_XH_val, gw);
    SELECT(h1_XH,   gpp[40], c_XH_val == 0, h1_XH);
    GPOW_i38_e5(hlp1_XH, (one_k << 5) - c_XH_val, gw);
    SELECT(hlp1_XH, gpp[43], c_XH_val == 0, hlp1_XH);
    fp_mul(hlp_XH, hlp_XH, hlp1_XH); //countM++;
    //countM++;
    fp_mul(u2_XH, h0_X, h1_XH);

    
    fp_copy(h0_XHL, hlp_XH);

   
    fp_prime_back(tmp_bn, h0_XHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    uint64_t c_XHL = (uint64_t)tab2[d] >> 1;

    
    GPOW_i39_e3(f_XHL, (one_k << 3) - c_XHL, gw);
    SELECT(f_XHL, gpp[42], c_XHL == 0, f_XHL); //countM++;
    fp_sqr(u2_XHL, f_XHL); fp_mul(u2_XHL, u2_XHL, u2_XH); //countS++;

    
    fp_prime_back(tmp_bn, u2_XHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    uint64_t e1_XHL = (uint64_t)tab2[d] >> 1;
    fp_copy(d_XHL, fll[e1_XHL << 1]);
    fp_mul(d_XHL, d_XHL, f_XHL); //countM++;

    uint64_t e_XHL      = (e1_XHL - 1) & 7ULL;
    uint64_t e1_XH_node = c_XHL + (e_XHL << 3);   /* 6-bit */

    
    uint64_t e_XH_adj = (e1_XH_node - 1) & 63ULL;
    uint64_t c_X      = c_XH_val + (e_XH_adj << 5);

    
    GPOW_i22_e11(f_X, (one_k << 11) - c_X, gw);
    SELECT(f_X, gpp[33], c_X == 0, f_X); //countM++;
    fp_sqr(f_X_sq, f_X); fp_mul(h1_X, f_X_sq, h1_A); //countS++;

    
    fp_copy(h0_XL, h1_X);
    for (int _j = 0; _j < 6; _j++) {
        if (_j == 3) fp_copy(hlp_XL, h0_XL);
        fp_sqr(h0_XL, h0_XL);
    }
    //countS += 6;

    
    fp_copy(h0_XLH, h0_XL);
    for (int _j = 0; _j < 3; _j++) fp_sqr(h0_XLH, h0_XLH);
    //countS += 3;

    
    fp_prime_back(tmp_bn, h0_XLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    uint64_t c_XLH = (uint64_t)tab2[d]>> 1;

    
    GPOW_i40_e3(h1_XLH, (one_k << 3) - c_XLH, gw);
    SELECT(h1_XLH, gpp[43], c_XLH == 0, h1_XLH);
    fp_mul(u2_XLH, h0_XL, h1_XLH); //countM++;

    
    fp_prime_back(tmp_bn, u2_XLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    uint64_t e1_XLH  = (uint64_t)tab2[d] >> 1;
    uint64_t e_XLH   = (e1_XLH - 1) & 7ULL;
    uint64_t c_XL_val = c_XLH + (e_XLH << 3);   /* 6-bit */

    
    GPOW_i34_e6(h1_XL_h,  (one_k << 6) - c_XL_val, gw);
    SELECT(h1_XL_h,  gpp[40], c_XL_val == 0, h1_XL_h);
    GPOW_i37_e6(hlp1_XL, (one_k << 6) - c_XL_val, gw);
    SELECT(hlp1_XL,  gpp[43], c_XL_val == 0, hlp1_XL);
    fp_mul(hlp_XL, hlp_XL, hlp1_XL); //countM++;
    //countM++;
    fp_mul(u2_XL_h, h1_X, h1_XL_h);

    GPOW_i33_e6(f_XL, (one_k << 6) - c_XL_val, gw);
    SELECT(f_XL, gpp[39], c_XL_val == 0, f_XL); //countM++;
    fp_sqr(f_XL_sq, f_XL); fp_mul(h1_XL_product, f_XL_sq, h1_X); //countS++;

    
    fp_copy(h0_XLL, hlp_XL);

    
    fp_prime_back(tmp_bn, h0_XLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    uint64_t c_XLL = (uint64_t)tab2[d] >> 1;

    
    GPOW_i39_e3(f_XLL, (one_k << 3) - c_XLL, gw);
    SELECT(f_XLL, gpp[42], c_XLL == 0, f_XLL); //countM++;
    fp_sqr(u2_XLL, f_XLL); fp_mul(u2_XLL, u2_XLL, h1_XL_product); //countS++;

    
    fp_prime_back(tmp_bn, u2_XLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    uint64_t e1_XLL = (uint64_t)tab2[d]>> 1;
    fp_copy(d_XLL, fll[e1_XLL << 1]);
    fp_mul(d_XLL, d_XLL, f_XLL); //countM++;

    fp_mul(d_XL, d_XLL, f_XL); //countM++;

    
    fp_mul(d_X, d_XL, f_X); //countM++;

    
    fp_mul(out_t, d_X, f_A); //countM++;
    bn_new(tmp_bn);
    /* ── Free all ─────────────────────────────────────────────────────── */
#define FF(x) fp_free(x)
    FF(h0_A); FF(hlp_A); FF(h0_PA); FF(hlp_PA); FF(h0_PAH); FF(hlp_PAH);
    FF(h0_PAHH); FF(h1_PAHH); FF(u2_PAHH);
    FF(h1_PAH); FF(hlp1_PAH); FF(u2_PAH);
    FF(h0_PAHL); FF(f_PAHL); FF(u2_PAHL); FF(d_PAHL);
    FF(h1_PA); FF(hlp1_PA); FF(u2_PA); FF(h0_PAL);
    FF(h0_PALH); FF(h1_PALH); FF(u2_PALH);
    FF(f_PAL); FF(h1_PAL);
    FF(h0_PALL); FF(f_PALL); FF(u2_PALL); FF(d_PALL); FF(d_PAL);
    FF(f_A); FF(hlp1_A); FF(f_A_sq); FF(h1_A); FF(h0_X);
    FF(h0_XH); FF(hlp_XH); FF(h0_XHH); FF(h1_XHH); FF(u2_XHH);
    FF(h1_XH); FF(hlp1_XH); FF(u2_XH);
    FF(h0_XHL); FF(f_XHL); FF(u2_XHL); FF(d_XHL);
    FF(f_X); FF(f_X_sq); FF(h1_X);
    FF(h0_XL); FF(hlp_XL); FF(h0_XLH); FF(h1_XLH); FF(u2_XLH);
    FF(h1_XL_h); FF(hlp1_XL); FF(u2_XL_h);
    FF(f_XL); FF(f_XL_sq); FF(h1_XL_product);
    FF(h0_XLL); FF(f_XLL); FF(u2_XLL); FF(d_XLL); FF(d_XL);
    FF(d_X); FF(t_A);
#undef FF
}


void sqrt_ext(fp_t x, fp_t y, bn_t e_exp,
              fp_t gw[nw][we], fp_t fll[we], fp_t gpp[n], int tab2[16])
{
    fp_t u, v, w_, t;
    fp_null(u); fp_null(v); fp_null(w_); fp_null(t);
    RLC_TRY {
        fp_new(u); fp_new(v); fp_new(w_); fp_new(t);
        //fp_print(x);
        fp_exp(v,  x, e_exp);
        //fp_print(v);
        fp_mul(w_, x, v);
        fp_mul(u,  w_, v);
        DLPpow2ext(u, t, gw, fll, gpp, tab2);
        fp_mul(y, w_, t);
        //fp_print(x);
        //fp_sqr(y, y);
        //fp_print(y);
    }
    RLC_CATCH_ANY { RLC_THROW(ERR_CAUGHT); }
    RLC_FINALLY   { fp_free(u); fp_free(v); fp_free(w_); fp_free(t); }
}


int main(void)
{
    if (core_init() != RLC_OK) {
          core_clean();
          return 1;
    }

    
    if (ep_param_set_any_pairf() != RLC_OK) {
        printf("Curve initialization failed\n");
        core_clean();
        return 1;
    }

    int i, j;
    fp_t gw[nw][we], fll[we], gpp[n], g, z, b, h, hh, y;
    bn_t tmp, m, e;

    for (i = 0; i < nw; i++)
    for (j = 0; j < we; j++) { fp_null(gw[i][j]);}
    for (i = 0; i < we; i++) { fp_null(fll[i]);}
    for (i = 0; i < n;  i++) { fp_null(gpp[i]);}
    fp_null(b);  fp_null(y); 
    fp_null(g);  fp_null(z); 
    fp_null(h);  fp_null(hh); 
    bn_null(tmp); 
    bn_null(m);   
    bn_null(e);   
    
    
    RLC_TRY {
        bn_new(e);fp_new(g);fp_new(b);
        bn_new(m);fp_new(y);fp_new(z);
        bn_new(tmp);fp_new(hh);
        fp_new(h);
        for (i = 0; i < nw; i++)
        for (j = 0; j < we; j++) { fp_new(gw[i][j]); }
        for (i = 0; i < we; i++) { fp_new(fll[i]); }
        for (i = 0; i < n;  i++) { fp_new(gpp[i]); }
        
        int tab2[16]={0};
        
        tab2[ 0x1 ]= 0 ;
        tab2[ 0x37 ]= 1 ;
        tab2[ 0x3a ]= 2 ;
        tab2[ 0x3 ]= 3 ;
        tab2[ 0x23 ]= 4 ;
        tab2[ 0x31 ]= 5 ;
        tab2[ 0x30 ]= 6 ;
        tab2[ 0x1d ]= 7 ;
        tab2[ 0x0 ]= 8 ;
        tab2[ 0xa ]= 9 ;
        tab2[ 0x7 ]= 10 ;
        tab2[ 0x3e ]= 11 ;
        tab2[ 0x1e ]= 12 ;
        tab2[ 0x10 ]= 13 ;
        tab2[ 0x11 ]= 14 ;
        tab2[ 0x24 ]= 15 ;
        
        
        
        fp_rand(b);
        while (fp_is_sqr(b) != 1) fp_rand(b);
    

        
        bn_read_str(tmp,"5",1,16);
        fp_prime_conv(z,tmp);
        
        bn_read_str(m,"6b8e9185f1443ab18ec1701b28524ec688b67cc03d44e3c7bcd88bee82520005c2d7510c00000021423",96,16);
        fp_exp(g,z,m);
        
        bn_read_str(e,"35c748c2f8a21d58c760b80d94292763445b3e601ea271e3de6c45f741290002e16ba88600000010a11",83,16);
        
        
        bn_t a1;
        bn_null(a1);
        bn_new(a1);
        bn_read_str(a1,"40000000000",11,16);
        fp_exp(h,g,a1);

        
        fp_srt(hh, h);
        fp_inv(hh, hh);

        precomputation(g, h, hh, gw,  fll, gpp);

        MEASURE(sqrt_ext(b, y, e, gw, fll, gpp, tab2);)

        printf("RDTSC_clk_min=%f\n",    RDTSC_clk_min);
        printf("RDTSC_clk_median=%f\n", RDTSC_clk_median);
        printf("RDTSC_clk_max=%f\n",    RDTSC_clk_max);
        
    }
    RLC_CATCH_ANY { 
        RLC_THROW(ERR_CAUGHT);
    }
    RLC_FINALLY   {    
        for (i = 0; i < nw; i++)
            for (j = 0; j < we; j++) fp_free(gw[i][j]);
        for (i = 0; i < we; i++) fp_free(fll[i]);
        for (i = 0; i < n;  i++) fp_free(gpp[i]);
        fp_free(b); fp_free(y); fp_free(g); fp_free(z); fp_free(h); fp_free(hh);
        bn_free(tmp); bn_free(e); bn_free(m);
    }
    return 0;
}
