

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include <immintrin.h>
#include <relic/relic_core.h>
#include "relic/relic_md.h"

#define w   2
#define we  4           /* 2^2 */
#define n   37
#define nw  19          /* ceil(37/2) */

int countM = 0;
int countS = 0;
dig_t d;

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
 *
 * Sage:
 *   h  = g^(2^(n-w))   = g^(2^35)
 *   h1 = sqrt(h)
 *   rll[v] = h^v
 *   fll[v] = h1^(-v)
 *   gw[i][j] = g^(j * 2^(i*w))
 *   gpp[0]=g, gpp[i]=gpp[i-1]^2
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

static void GPOW_i33_e2(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 2) - 1;
    e <<= 1;                            /* ri=1 */
    fp_copy(t, gw[16][e & 0x3]); e >>= 2;
    fp_mul(t, t, gw[17][e & 0x3]); //countM++;
}


static void GPOW_i33_e1(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 1) - 1;
    e <<= 1;                            /* ri=1 */
    fp_copy(t, gw[16][e & 0x3]);
}


static void GPOW_i28_e4(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 4) - 1;
    fp_copy(t, gw[14][e & 0x3]); e >>= 2;
    fp_mul(t, t, gw[15][e & 0x3]); //countM++;
}


static void GPOW_i32_e2(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 2) - 1;
    fp_copy(t, gw[16][e & 0x3]);
}


static void GPOW_i34_e2(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 2) - 1;
    fp_copy(t, gw[17][e & 0x3]);
}


static void GPOW_i34_e1(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 1) - 1;
    fp_copy(t, gw[17][e & 0x3]);
}


static void GPOW_i19_e9(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 9) - 1;
    e <<= 1;                            /* ri=1 */
    fp_copy(t, gw[9][e  & 0x3]); e >>= 2;
    fp_mul(t, t, gw[10][e & 0x3]); //countM++; 
e >>= 2;
    fp_mul(t, t, gw[11][e & 0x3]); //countM++; 
e >>= 2;
    fp_mul(t, t, gw[12][e & 0x3]); //countM++; 
e >>= 2;
    fp_mul(t, t, gw[13][e & 0x3]); //countM++;
}


static void GPOW_i0_e17(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 17) - 1;
    fp_copy(t, gw[0][e & 0x3]); e >>= 2;
    fp_mul(t, t, gw[1][e & 0x3]); //countM++; 
e >>= 2;
    fp_mul(t, t, gw[2][e & 0x3]); //countM++; 
e >>= 2;
    fp_mul(t, t, gw[3][e & 0x3]); //countM++; 
e >>= 2;
    fp_mul(t, t, gw[4][e & 0x3]); //countM++; 
e >>= 2;
    fp_mul(t, t, gw[5][e & 0x3]); //countM++; 
e >>= 2;
    fp_mul(t, t, gw[6][e & 0x3]); //countM++; 
e >>= 2;
    fp_mul(t, t, gw[7][e & 0x3]); //countM++; 
e >>= 2;
    fp_mul(t, t, gw[8][e & 0x3]); //countM++;
}


static void GPOW_i10_e18(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 18) - 1;
    fp_copy(t, gw[5][e  & 0x3]); e >>= 2;
    fp_mul(t, t, gw[6][e  & 0x3]); //countM++; 
e >>= 2;
    fp_mul(t, t, gw[7][e  & 0x3]); //countM++; 
e >>= 2;
    fp_mul(t, t, gw[8][e  & 0x3]); //countM++; 
e >>= 2;
    fp_mul(t, t, gw[9][e  & 0x3]); //countM++; 
e >>= 2;
    fp_mul(t, t, gw[10][e & 0x3]); //countM++; 
e >>= 2;
    fp_mul(t, t, gw[11][e & 0x3]); //countM++; 
e >>= 2;
    fp_mul(t, t, gw[12][e & 0x3]); //countM++; 
e >>= 2;
    fp_mul(t, t, gw[13][e & 0x3]); //countM++;
}


static void GPOW_i17_e9(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 9) - 1;
    e <<= 1;                            /* ri=1 */
    fp_copy(t, gw[8][e  & 0x3]); e >>= 2;
    fp_mul(t, t, gw[9][e  & 0x3]); //countM++; 
e >>= 2;
    fp_mul(t, t, gw[10][e & 0x3]); //countM++; 
e >>= 2;
    fp_mul(t, t, gw[11][e & 0x3]); //countM++; 
e >>= 2;
    fp_mul(t, t, gw[12][e & 0x3]); //countM++;
}


static void GPOW_i26_e5(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 5) - 1;
    fp_copy(t, gw[13][e & 0x3]); e >>= 2;
    fp_mul(t, t, gw[14][e & 0x3]); //countM++; 
e >>= 2;
    fp_mul(t, t, gw[15][e & 0x3]); //countM++;
}


static void GPOW_i31_e2(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 2) - 1;
    e <<= 1;                            /* ri=1 */
    fp_copy(t, gw[15][e & 0x3]); e >>= 2;
    fp_mul(t, t, gw[16][e & 0x3]); //countM++;
}


void DLPpow2ext(fp_t u_A, fp_t out_t,
                fp_t gw[nw][we], int rlll[4], fp_t rll[we], fp_t fll[we], fp_t gpp[n])
{
    const uint64_t one_k = 1;

    

    
    fp_t h0_A, hlp_A;

    
    fp_t h0_PA;

    
    fp_t dlp1_h0, dlp1_h0_HH, dlp1_h1_HH, dlp1_u2_HH;
    fp_t dlp1_h1_H, dlp1_u2_H;
    fp_t dlp1_hlp_HL, dlp1_h0_HL, dlp1_h1_HL, dlp1_hlp1_HL, dlp1_u2_HL;
    fp_t dlp1_h0_HLL, dlp1_h1_HLL, dlp1_u2_HLL;

    
    fp_t h1_PA, u2_PA;
    fp_t dlp2_h0, dlp2_h0_HH, dlp2_h1_HH, dlp2_u2_HH;
    fp_t dlp2_h1_H, dlp2_u2_H;
    fp_t dlp2_hlp_HL, dlp2_h0_HL, dlp2_h1_HL, dlp2_hlp1_HL, dlp2_u2_HL;
    fp_t dlp2_h0_HLL, dlp2_h1_HLL, dlp2_u2_HLL;

    
    fp_t f_A, hlp1_A, f_A_sq;

    
    fp_t u_X0, h0_X0;
    fp_t dlp3_h0, dlp3_h0_HH, dlp3_h1_HH, dlp3_u2_HH;
    fp_t dlp3_h1_H, dlp3_u2_H;
    fp_t dlp3_hlp_HL, dlp3_h0_HL, dlp3_h1_HL, dlp3_hlp1_HL, dlp3_u2_HL;
    fp_t dlp3_h0_HLL, dlp3_h1_HLL, dlp3_u2_HLL;
    fp_t f_X0, f_X0_sq;

    
    fp_t u_X1, h0_X1;
    fp_t hlp_X1P, h0_X1P;
    fp_t h1_X1P, hlp1_X1P, u2_X1P;
    fp_t h0_X1PL, h1_X1PL, u2_X1PL;
    fp_t f_X1, f_X1_sq;

    
    fp_t u_X2, hlp_X2, h0_X2;
    fp_t f_X2, hlp1_X2, f_X2_sq;

    
    fp_t u_X3, h0_X3;
    fp_t f_X3, f_X3_sq;

    
    fp_t u_X4, d1_X4;

    
    fp_t t_X3, t_X2, t_X1, t_X0, t_A;

    
    fp_null(h0_A);          fp_new(h0_A);
    fp_null(hlp_A);         fp_new(hlp_A);
    fp_null(h0_PA);         fp_new(h0_PA);

    fp_null(dlp1_h0);       fp_new(dlp1_h0);
    fp_null(dlp1_h0_HH);    fp_new(dlp1_h0_HH);
    fp_null(dlp1_h1_HH);    fp_new(dlp1_h1_HH);
    fp_null(dlp1_u2_HH);    fp_new(dlp1_u2_HH);
    fp_null(dlp1_h1_H);     fp_new(dlp1_h1_H);
    fp_null(dlp1_u2_H);     fp_new(dlp1_u2_H);
    fp_null(dlp1_hlp_HL);   fp_new(dlp1_hlp_HL);
    fp_null(dlp1_h0_HL);    fp_new(dlp1_h0_HL);
    fp_null(dlp1_h1_HL);    fp_new(dlp1_h1_HL);
    fp_null(dlp1_hlp1_HL);  fp_new(dlp1_hlp1_HL);
    fp_null(dlp1_u2_HL);    fp_new(dlp1_u2_HL);
    fp_null(dlp1_h0_HLL);   fp_new(dlp1_h0_HLL);
    fp_null(dlp1_h1_HLL);   fp_new(dlp1_h1_HLL);
    fp_null(dlp1_u2_HLL);   fp_new(dlp1_u2_HLL);

    fp_null(h1_PA);         fp_new(h1_PA);
    fp_null(u2_PA);         fp_new(u2_PA);

    fp_null(dlp2_h0);       fp_new(dlp2_h0);
    fp_null(dlp2_h0_HH);    fp_new(dlp2_h0_HH);
    fp_null(dlp2_h1_HH);    fp_new(dlp2_h1_HH);
    fp_null(dlp2_u2_HH);    fp_new(dlp2_u2_HH);
    fp_null(dlp2_h1_H);     fp_new(dlp2_h1_H);
    fp_null(dlp2_u2_H);     fp_new(dlp2_u2_H);
    fp_null(dlp2_hlp_HL);   fp_new(dlp2_hlp_HL);
    fp_null(dlp2_h0_HL);    fp_new(dlp2_h0_HL);
    fp_null(dlp2_h1_HL);    fp_new(dlp2_h1_HL);
    fp_null(dlp2_hlp1_HL);  fp_new(dlp2_hlp1_HL);
    fp_null(dlp2_u2_HL);    fp_new(dlp2_u2_HL);
    fp_null(dlp2_h0_HLL);   fp_new(dlp2_h0_HLL);
    fp_null(dlp2_h1_HLL);   fp_new(dlp2_h1_HLL);
    fp_null(dlp2_u2_HLL);   fp_new(dlp2_u2_HLL);

    fp_null(f_A);           fp_new(f_A);
    fp_null(hlp1_A);        fp_new(hlp1_A);
    fp_null(f_A_sq);        fp_new(f_A_sq);

    fp_null(u_X0);          fp_new(u_X0);
    fp_null(h0_X0);         fp_new(h0_X0);

    fp_null(dlp3_h0);       fp_new(dlp3_h0);
    fp_null(dlp3_h0_HH);    fp_new(dlp3_h0_HH);
    fp_null(dlp3_h1_HH);    fp_new(dlp3_h1_HH);
    fp_null(dlp3_u2_HH);    fp_new(dlp3_u2_HH);
    fp_null(dlp3_h1_H);     fp_new(dlp3_h1_H);
    fp_null(dlp3_u2_H);     fp_new(dlp3_u2_H);
    fp_null(dlp3_hlp_HL);   fp_new(dlp3_hlp_HL);
    fp_null(dlp3_h0_HL);    fp_new(dlp3_h0_HL);
    fp_null(dlp3_h1_HL);    fp_new(dlp3_h1_HL);
    fp_null(dlp3_hlp1_HL);  fp_new(dlp3_hlp1_HL);
    fp_null(dlp3_u2_HL);    fp_new(dlp3_u2_HL);
    fp_null(dlp3_h0_HLL);   fp_new(dlp3_h0_HLL);
    fp_null(dlp3_h1_HLL);   fp_new(dlp3_h1_HLL);
    fp_null(dlp3_u2_HLL);   fp_new(dlp3_u2_HLL);

    fp_null(f_X0);          fp_new(f_X0);
    fp_null(f_X0_sq);       fp_new(f_X0_sq);

    fp_null(u_X1);          fp_new(u_X1);
    fp_null(h0_X1);         fp_new(h0_X1);
    fp_null(hlp_X1P);       fp_new(hlp_X1P);
    fp_null(h0_X1P);        fp_new(h0_X1P);
    fp_null(h1_X1P);        fp_new(h1_X1P);
    fp_null(hlp1_X1P);      fp_new(hlp1_X1P);
    fp_null(u2_X1P);        fp_new(u2_X1P);
    fp_null(h0_X1PL);       fp_new(h0_X1PL);
    fp_null(h1_X1PL);       fp_new(h1_X1PL);
    fp_null(u2_X1PL);       fp_new(u2_X1PL);
    fp_null(f_X1);          fp_new(f_X1);
    fp_null(f_X1_sq);       fp_new(f_X1_sq);

    fp_null(u_X2);          fp_new(u_X2);
    fp_null(hlp_X2);        fp_new(hlp_X2);
    fp_null(h0_X2);         fp_new(h0_X2);
    fp_null(f_X2);          fp_new(f_X2);
    fp_null(hlp1_X2);       fp_new(hlp1_X2);
    fp_null(f_X2_sq);       fp_new(f_X2_sq);

    fp_null(u_X3);          fp_new(u_X3);
    fp_null(h0_X3);         fp_new(h0_X3);
    fp_null(f_X3);          fp_new(f_X3);
    fp_null(f_X3_sq);       fp_new(f_X3_sq);

    fp_null(u_X4);          fp_new(u_X4);
    fp_null(d1_X4);         fp_new(d1_X4);

    fp_null(t_X3);          fp_new(t_X3);
    fp_null(t_X2);          fp_new(t_X2);
    fp_null(t_X1);          fp_new(t_X1);
    fp_null(t_X0);          fp_new(t_X0);
    fp_null(t_A);           fp_new(t_A);

    int _ii, _tmp;

    
    fp_copy(h0_A, u_A);
    for (int _j = 0; _j < 19; _j++) {
        if (_j == 10) fp_copy(hlp_A, h0_A);
        fp_sqr(h0_A, h0_A);
    }
    //countS += 19;

    
    fp_copy(h0_PA, h0_A);
    for (int _j = 0; _j < 9; _j++) fp_sqr(h0_PA, h0_PA);
    //countS += 9;

    fp_copy(dlp1_h0, h0_PA);
    /* square 5 */
    for (int _j = 0; _j < 5; _j++) fp_sqr(dlp1_h0, dlp1_h0);
    //countS += 5;
   
    fp_copy(dlp1_h0_HH, dlp1_h0);
    for (int _j = 0; _j < 2; _j++) fp_sqr(dlp1_h0_HH, dlp1_h0_HH);
    //countS += 2;
    
	bn_t tmp_bn;
	bn_null(tmp_bn);bn_new(tmp_bn);
	
    fp_prime_back(tmp_bn, dlp1_h0_HH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    _tmp=rlll[d];

    uint64_t dlp1_c_HH = (uint64_t)_tmp;           
    GPOW_i33_e2(dlp1_h1_HH, (one_k << 2) - dlp1_c_HH, gw);
    SELECT(dlp1_h1_HH, gpp[35], dlp1_c_HH == 0, dlp1_h1_HH);
    //countM++;
    fp_mul(dlp1_u2_HH, dlp1_h0, dlp1_h1_HH);
    
    fp_prime_back(tmp_bn, dlp1_u2_HH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    _tmp=rlll[d];

    uint64_t dlp1_d_HH = (uint64_t)_tmp;        
    uint64_t dlp1_c_H  = dlp1_c_HH + ((dlp1_d_HH - 1) % 4) * 4;
    
    GPOW_i28_e4(dlp1_h1_H, (one_k << 4) - dlp1_c_H, gw);
    SELECT(dlp1_h1_H, gpp[32], dlp1_c_H == 0, dlp1_h1_H);
    //countM++;
    fp_mul(dlp1_u2_H, h0_PA, dlp1_h1_H);
    
    fp_copy(dlp1_h0_HL, dlp1_u2_H);
    for (int _j = 0; _j < 3; _j++) {
        if (_j == 2) fp_copy(dlp1_hlp_HL, dlp1_h0_HL);
        fp_sqr(dlp1_h0_HL, dlp1_h0_HL);
    }
    //countS += 3;
    
    fp_prime_back(tmp_bn, dlp1_h0_HL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    _tmp=rlll[d];

    uint64_t dlp1_c_HL = (uint64_t)_tmp;           /* >> 0 */
    GPOW_i32_e2(dlp1_h1_HL,   (one_k << 2) - dlp1_c_HL, gw);
    SELECT(dlp1_h1_HL,   gpp[34], dlp1_c_HL == 0, dlp1_h1_HL);
    GPOW_i34_e2(dlp1_hlp1_HL, (one_k << 2) - dlp1_c_HL, gw);
    SELECT(dlp1_hlp1_HL, gpp[36], dlp1_c_HL == 0, dlp1_hlp1_HL);
    fp_mul(dlp1_hlp_HL, dlp1_hlp_HL, dlp1_hlp1_HL); //countM++;
    //countM++;
    fp_mul(dlp1_u2_HL, dlp1_u2_H, dlp1_h1_HL);
    
    fp_copy(dlp1_h0_HLL, dlp1_hlp_HL);
    
    fp_prime_back(tmp_bn, dlp1_h0_HLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    _tmp=rlll[d];

    uint64_t dlp1_c_HLL = (uint64_t)_tmp >> 1;
    GPOW_i34_e1(dlp1_h1_HLL, (one_k << 1) - dlp1_c_HLL, gw);
    SELECT(dlp1_h1_HLL, gpp[35], dlp1_c_HLL == 0, dlp1_h1_HLL);
    //countM++;
    fp_mul(dlp1_u2_HLL, dlp1_u2_HL, dlp1_h1_HLL);
    
    fp_prime_back(tmp_bn, dlp1_u2_HLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    _tmp=rlll[d];

    uint64_t dlp1_d_HLL = (uint64_t)_tmp;          
    uint64_t dlp1_d_HL  = dlp1_c_HLL + ((dlp1_d_HLL - 1) % 4) * 2;
    uint64_t dlp1_c_L   = dlp1_c_HL  + ((dlp1_d_HL  - 1) % 8) * 4;
    uint64_t c_PA       = dlp1_c_H   + ((dlp1_c_L   - 1) % 32) * 16;

    
    GPOW_i19_e9(h1_PA, (one_k << 9) - c_PA, gw);
    SELECT(h1_PA, gpp[28], c_PA == 0, h1_PA);
    //countM++;
    fp_mul(u2_PA, h0_A, h1_PA);

    
    fp_copy(dlp2_h0, u2_PA);
    for (int _j = 0; _j < 5; _j++) fp_sqr(dlp2_h0, dlp2_h0);
    //countS += 5;
    fp_copy(dlp2_h0_HH, dlp2_h0);
    for (int _j = 0; _j < 2; _j++) fp_sqr(dlp2_h0_HH, dlp2_h0_HH);
    //countS += 2;
    fp_prime_back(tmp_bn, dlp2_h0_HH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    _tmp=rlll[d];

    uint64_t dlp2_c_HH = (uint64_t)_tmp;
    GPOW_i33_e2(dlp2_h1_HH, (one_k << 2) - dlp2_c_HH, gw);
    SELECT(dlp2_h1_HH, gpp[35], dlp2_c_HH == 0, dlp2_h1_HH);
    //countM++;
    fp_mul(dlp2_u2_HH, dlp2_h0, dlp2_h1_HH);
    fp_prime_back(tmp_bn, dlp2_u2_HH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    _tmp=rlll[d];

    uint64_t dlp2_d_HH = (uint64_t)_tmp;
    uint64_t dlp2_c_H  = dlp2_c_HH + ((dlp2_d_HH - 1) % 4) * 4;
    GPOW_i28_e4(dlp2_h1_H, (one_k << 4) - dlp2_c_H, gw);
    SELECT(dlp2_h1_H, gpp[32], dlp2_c_H == 0, dlp2_h1_H);
    //countM++;
    fp_mul(dlp2_u2_H, u2_PA, dlp2_h1_H);
    fp_copy(dlp2_h0_HL, dlp2_u2_H);
    for (int _j = 0; _j < 3; _j++) {
        if (_j == 2) fp_copy(dlp2_hlp_HL, dlp2_h0_HL);
        fp_sqr(dlp2_h0_HL, dlp2_h0_HL);
    }
    //countS += 3;
    fp_prime_back(tmp_bn, dlp2_h0_HL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    _tmp=rlll[d];

    uint64_t dlp2_c_HL = (uint64_t)_tmp;
    GPOW_i32_e2(dlp2_h1_HL,   (one_k << 2) - dlp2_c_HL, gw);
    SELECT(dlp2_h1_HL,   gpp[34], dlp2_c_HL == 0, dlp2_h1_HL);
    GPOW_i34_e2(dlp2_hlp1_HL, (one_k << 2) - dlp2_c_HL, gw);
    SELECT(dlp2_hlp1_HL, gpp[36], dlp2_c_HL == 0, dlp2_hlp1_HL);
    fp_mul(dlp2_hlp_HL, dlp2_hlp_HL, dlp2_hlp1_HL); //countM++;
    //countM++;
    fp_mul(dlp2_u2_HL, dlp2_u2_H, dlp2_h1_HL);
    fp_copy(dlp2_h0_HLL, dlp2_hlp_HL);
    fp_prime_back(tmp_bn, dlp2_h0_HLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    _tmp=rlll[d];

    uint64_t dlp2_c_HLL = (uint64_t)_tmp >> 1;
    GPOW_i34_e1(dlp2_h1_HLL, (one_k << 1) - dlp2_c_HLL, gw);
    SELECT(dlp2_h1_HLL, gpp[35], dlp2_c_HLL == 0, dlp2_h1_HLL);
    //countM++;
    fp_mul(dlp2_u2_HLL, dlp2_u2_HL, dlp2_h1_HLL);
    fp_prime_back(tmp_bn, dlp2_u2_HLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    _tmp=rlll[d];

    uint64_t dlp2_d_HLL = (uint64_t)_tmp;
    uint64_t dlp2_d_HL  = dlp2_c_HLL + ((dlp2_d_HLL - 1) % 4) * 2;
    uint64_t dlp2_c_L   = dlp2_c_HL  + ((dlp2_d_HL  - 1) % 8) * 4;
    uint64_t d_PA       = dlp2_c_H   + ((dlp2_c_L   - 1) % 32) * 16;

   
    uint64_t c_A = c_PA + ((d_PA - 1) % 512) * 512;

    
    GPOW_i0_e17(f_A,    ((one_k << 18) - c_A + 1) >> 1, gw);
    SELECT(f_A,    gpp[17], c_A == 0, f_A);
    GPOW_i10_e18(hlp1_A, (one_k << 18) - c_A, gw);
    SELECT(hlp1_A, gpp[28], c_A == 0, hlp1_A);
    fp_mul(hlp_A, hlp_A, hlp1_A); //countM++;
    //countM++;
    //countS++;   

    
    fp_sqr(f_A_sq, f_A);
    fp_mul(u_X0, u_A, f_A_sq);
    fp_copy(h0_X0, hlp_A); 
    fp_copy(dlp3_h0, h0_X0);
    for (int _j = 0; _j < 5; _j++) fp_sqr(dlp3_h0, dlp3_h0);
    //countS += 5;
    fp_copy(dlp3_h0_HH, dlp3_h0);
    for (int _j = 0; _j < 2; _j++) fp_sqr(dlp3_h0_HH, dlp3_h0_HH);
    //countS += 2;
    fp_prime_back(tmp_bn, dlp3_h0_HH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    _tmp=rlll[d];
    uint64_t dlp3_c_HH = (uint64_t)_tmp;
    GPOW_i33_e2(dlp3_h1_HH, (one_k << 2) - dlp3_c_HH, gw);
    SELECT(dlp3_h1_HH, gpp[35], dlp3_c_HH == 0, dlp3_h1_HH);
    //countM++;
    fp_mul(dlp3_u2_HH, dlp3_h0, dlp3_h1_HH);
    fp_prime_back(tmp_bn, dlp3_u2_HH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    _tmp=rlll[d];
    uint64_t dlp3_d_HH = (uint64_t)_tmp;
    uint64_t dlp3_c_H  = dlp3_c_HH + ((dlp3_d_HH - 1) % 4) * 4;
    GPOW_i28_e4(dlp3_h1_H, (one_k << 4) - dlp3_c_H, gw);
    SELECT(dlp3_h1_H, gpp[32], dlp3_c_H == 0, dlp3_h1_H);
    //countM++;
    fp_mul(dlp3_u2_H, h0_X0, dlp3_h1_H);
    fp_copy(dlp3_h0_HL, dlp3_u2_H);
    for (int _j = 0; _j < 3; _j++) {
        if (_j == 2) fp_copy(dlp3_hlp_HL, dlp3_h0_HL);
        fp_sqr(dlp3_h0_HL, dlp3_h0_HL);
    }
    fp_prime_back(tmp_bn, dlp3_h0_HL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    _tmp=rlll[d];
    uint64_t dlp3_c_HL = (uint64_t)_tmp;
    GPOW_i32_e2(dlp3_h1_HL,   (one_k << 2) - dlp3_c_HL, gw);
    SELECT(dlp3_h1_HL,   gpp[34], dlp3_c_HL == 0, dlp3_h1_HL);
    GPOW_i34_e2(dlp3_hlp1_HL, (one_k << 2) - dlp3_c_HL, gw);
    SELECT(dlp3_hlp1_HL, gpp[36], dlp3_c_HL == 0, dlp3_hlp1_HL);
    fp_mul(dlp3_hlp_HL, dlp3_hlp_HL, dlp3_hlp1_HL); //countM++;
    //countM++;
    fp_mul(dlp3_u2_HL, dlp3_u2_H, dlp3_h1_HL);
    fp_copy(dlp3_h0_HLL, dlp3_hlp_HL);
    fp_prime_back(tmp_bn, dlp3_h0_HLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    _tmp=rlll[d];

    uint64_t dlp3_c_HLL = (uint64_t)_tmp >> 1;
    GPOW_i34_e1(dlp3_h1_HLL, (one_k << 1) - dlp3_c_HLL, gw);
    SELECT(dlp3_h1_HLL, gpp[35], dlp3_c_HLL == 0, dlp3_h1_HLL);
    //countM++;
    fp_mul(dlp3_u2_HLL, dlp3_u2_HL, dlp3_h1_HLL);
    fp_prime_back(tmp_bn, dlp3_u2_HLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    _tmp=rlll[d];

    uint64_t dlp3_d_HLL = (uint64_t)_tmp;
    uint64_t dlp3_d_HL  = dlp3_c_HLL + ((dlp3_d_HLL - 1) % 4) * 2;
    uint64_t dlp3_c_L   = dlp3_c_HL  + ((dlp3_d_HL  - 1) % 8) * 4;
    uint64_t c_X0       = dlp3_c_H   + ((dlp3_c_L   - 1) % 32) * 16;

    
    GPOW_i17_e9(f_X0, (one_k << 9) - c_X0, gw);
    SELECT(f_X0, gpp[26], c_X0 == 0, f_X0);
    //countM++;
    //countS++;  
    fp_sqr(f_X0_sq, f_X0);
    fp_mul(u_X1, u_X0, f_X0_sq);
    fp_copy(h0_X1, u_X1);
    for (int _j = 0; _j < 5; _j++) fp_sqr(h0_X1, h0_X1);
    //countS += 5;

    fp_copy(h0_X1P, h0_X1);
    for (int _j = 0; _j < 3; _j++) {
        if (_j == 2) fp_copy(hlp_X1P, h0_X1P);
        fp_sqr(h0_X1P, h0_X1P);
    }
    //countS += 3;
    fp_prime_back(tmp_bn, h0_X1P);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    _tmp=rlll[d];
    uint64_t c_X1H = (uint64_t)_tmp;               /* >> 0 */
    GPOW_i32_e2(h1_X1P,   (one_k << 2) - c_X1H, gw);
    SELECT(h1_X1P,   gpp[34], c_X1H == 0, h1_X1P);
    GPOW_i34_e2(hlp1_X1P, (one_k << 2) - c_X1H, gw);
    SELECT(hlp1_X1P, gpp[36], c_X1H == 0, hlp1_X1P);
    fp_mul(hlp_X1P, hlp_X1P, hlp1_X1P); //countM++;
    //countM++;
    fp_mul(u2_X1P, h0_X1, h1_X1P);


    fp_copy(h0_X1PL, hlp_X1P);         
    fp_prime_back(tmp_bn, h0_X1PL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    _tmp=rlll[d];
    uint64_t c_X1L = (uint64_t)_tmp >> 1;
    GPOW_i34_e1(h1_X1PL, (one_k << 1) - c_X1L, gw);
    SELECT(h1_X1PL, gpp[35], c_X1L == 0, h1_X1PL);
    //countM++;
    fp_mul(u2_X1PL, u2_X1P, h1_X1PL);
    fp_prime_back(tmp_bn, u2_X1PL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    _tmp=rlll[d];
    uint64_t d_X1L = (uint64_t)_tmp;               /* >> 0 */
    uint64_t d_X1  = c_X1L + ((d_X1L - 1) % 4) * 2;
    uint64_t c_X1  = c_X1H + ((d_X1  - 1) % 8) * 4;


    GPOW_i26_e5(f_X1, (one_k << 5) - c_X1, gw);
    SELECT(f_X1, gpp[31], c_X1 == 0, f_X1);
    //countM++;
    //countS++;   


    fp_sqr(f_X1_sq, f_X1);
    fp_mul(u_X2, u_X1, f_X1_sq);
    fp_copy(h0_X2, u_X2);
    for (int _j = 0; _j < 3; _j++) {
        if (_j == 2) fp_copy(hlp_X2, h0_X2);
        fp_sqr(h0_X2, h0_X2);
    }
    //countS += 3;
    fp_prime_back(tmp_bn, h0_X2);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    _tmp=rlll[d];

    uint64_t c_X2 = (uint64_t)_tmp;                
    GPOW_i31_e2(f_X2,    (one_k << 2) - c_X2, gw);
    SELECT(f_X2,    gpp[33], c_X2 == 0, f_X2);
    GPOW_i34_e2(hlp1_X2, (one_k << 2) - c_X2, gw);
    SELECT(hlp1_X2, gpp[36], c_X2 == 0, hlp1_X2);
    fp_mul(hlp_X2, hlp_X2, hlp1_X2); //countM++;
    
    fp_sqr(f_X2_sq, f_X2);
    fp_mul(u_X3, u_X2, f_X2_sq);
    fp_copy(h0_X3, hlp_X2);            
    fp_prime_back(tmp_bn, h0_X3);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    _tmp=rlll[d];

    uint64_t c_X3 = (uint64_t)_tmp >> 1;           
    GPOW_i33_e1(f_X3, (one_k << 1) - c_X3, gw);
    SELECT(f_X3, gpp[34], c_X3 == 0, f_X3);
   
    fp_sqr(f_X3_sq, f_X3);
    fp_mul(u_X4, u_X3, f_X3_sq);
    fp_prime_back(tmp_bn, u_X4);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    _tmp=rlll[d];

    uint64_t c1_X4 = (uint64_t)_tmp;             
    fp_copy(d1_X4, fll[c1_X4]);                    


    fp_mul(t_X3, f_X3, d1_X4);

    fp_mul(t_X2, f_X2, t_X3);

    fp_mul(t_X1, f_X1, t_X2);

    fp_mul(t_X0, f_X0, t_X1);

    fp_mul(t_A, f_A, t_X0);

    fp_copy(out_t, t_A);

    /* ── Free all ──────────────────────────────────────────────────────── */
    fp_free(h0_A);          fp_free(hlp_A);
    fp_free(h0_PA);

    fp_free(dlp1_h0);       fp_free(dlp1_h0_HH);  fp_free(dlp1_h1_HH);
    fp_free(dlp1_u2_HH);    fp_free(dlp1_h1_H);   fp_free(dlp1_u2_H);
    fp_free(dlp1_hlp_HL);   fp_free(dlp1_h0_HL);  fp_free(dlp1_h1_HL);
    fp_free(dlp1_hlp1_HL);  fp_free(dlp1_u2_HL);
    fp_free(dlp1_h0_HLL);   fp_free(dlp1_h1_HLL); fp_free(dlp1_u2_HLL);

    fp_free(h1_PA);         fp_free(u2_PA);

    fp_free(dlp2_h0);       fp_free(dlp2_h0_HH);  fp_free(dlp2_h1_HH);
    fp_free(dlp2_u2_HH);    fp_free(dlp2_h1_H);   fp_free(dlp2_u2_H);
    fp_free(dlp2_hlp_HL);   fp_free(dlp2_h0_HL);  fp_free(dlp2_h1_HL);
    fp_free(dlp2_hlp1_HL);  fp_free(dlp2_u2_HL);
    fp_free(dlp2_h0_HLL);   fp_free(dlp2_h1_HLL); fp_free(dlp2_u2_HLL);

    fp_free(f_A);           fp_free(hlp1_A);       fp_free(f_A_sq);

    fp_free(u_X0);          fp_free(h0_X0);

    fp_free(dlp3_h0);       fp_free(dlp3_h0_HH);  fp_free(dlp3_h1_HH);
    fp_free(dlp3_u2_HH);    fp_free(dlp3_h1_H);   fp_free(dlp3_u2_H);
    fp_free(dlp3_hlp_HL);   fp_free(dlp3_h0_HL);  fp_free(dlp3_h1_HL);
    fp_free(dlp3_hlp1_HL);  fp_free(dlp3_u2_HL);
    fp_free(dlp3_h0_HLL);   fp_free(dlp3_h1_HLL); fp_free(dlp3_u2_HLL);

    fp_free(f_X0);          fp_free(f_X0_sq);

    fp_free(u_X1);          fp_free(h0_X1);
    fp_free(hlp_X1P);       fp_free(h0_X1P);
    fp_free(h1_X1P);        fp_free(hlp1_X1P);    fp_free(u2_X1P);
    fp_free(h0_X1PL);       fp_free(h1_X1PL);     fp_free(u2_X1PL);
    fp_free(f_X1);          fp_free(f_X1_sq);

    fp_free(u_X2);          fp_free(hlp_X2);       fp_free(h0_X2);
    fp_free(f_X2);          fp_free(hlp1_X2);      fp_free(f_X2_sq);

    fp_free(u_X3);          fp_free(h0_X3);
    fp_free(f_X3);          fp_free(f_X3_sq);

    fp_free(u_X4);          fp_free(d1_X4);

    fp_free(t_X3);          fp_free(t_X2);
    fp_free(t_X1);          fp_free(t_X0);         fp_free(t_A);
}


void sqrt_ext(fp_t x, fp_t y, bn_t e_exp,
              fp_t gw[nw][we], int rlll[4], fp_t rll[we], fp_t fll[we], fp_t gpp[n])
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

    /* Initialize pairing-friendly curve parameters */
    if (ep_param_set_any_pairf() != RLC_OK) {
        printf("Curve initialization failed\n");
        core_clean();
        return 1;
    }

    int i, j;
    fp_t gw[nw][we], rll[we], fll[we], gpp[n], g, z, b, h, hh, y;
    bn_t tmp, m, e;
	int rlll[4]={0};

	rlll[ 0x1 ]= 0 ;
	rlll[ 0x3 ]= 1 ;
	rlll[ 0x0 ]= 2 ;
	rlll[ 0x2 ]= 3 ;

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
    bn_read_str(tmp, "b", 1, 16);
    fp_prime_conv(z, tmp);
    {
        bn_t p_bn, two37;
        bn_null(p_bn); bn_new(p_bn);
        bn_null(two37); bn_new(two37);
        bn_read_str(p_bn,
            "17452A017CBDD682A502A1E13A9D671D27958EECD2C33C3A36C2ADE221D9B956"
            "BEA4F49B2B5EE7D7D72AD20065DCB2B8E9CA0015B5152C00000811E000000001",
            128, 16);
        bn_set_dig(two37, 1);
        bn_lsh(two37, two37, 37);
        bn_sub_dig(p_bn, p_bn, 1);   /* p-1 */
        bn_div(m, p_bn, two37);      /* m = (p-1)/2^37 */
        bn_free(p_bn); bn_free(two37);
    }
    fp_exp(g, z, m);

    
    {
        bn_t one;
        bn_null(one); bn_new(one);
        bn_set_dig(one, 1);
        bn_sub(e, m, one);
        bn_rsh(e, e, 1);
        bn_free(one);
    }

    
    {
        bn_t exp_h;
        bn_null(exp_h); bn_new(exp_h);
        bn_set_dig(exp_h, 1);
        bn_lsh(exp_h, exp_h, 35);
        fp_exp(h, g, exp_h);
        bn_free(exp_h);
    }

    
    fp_srt(hh, h);
    fp_inv(hh, hh);

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
    
    return 0;
}
