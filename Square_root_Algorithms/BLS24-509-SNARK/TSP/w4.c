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
#define we  16       /* 2^w */
#define n   37
#define nw  10       

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
    bn_free(temp); bn_free(one);
}

/* =========================================================================
 * SELECT
 * ========================================================================= */
static void SELECT(fp_t a0, fp_t a1, bool ctl, fp_t out)
{
    if (ctl) fp_copy(out, a1);
    else     fp_copy(out, a0);
}



static void GPOW_i0_e17(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 17) - 1;
    uint64_t wm = we - 1;
    fp_copy(t, gw[0][e & wm]); e >>= w;
    fp_mul(t, t, gw[1][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[2][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[3][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[4][e & wm]); //countM++;
}


static void GPOW_i10_e18(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 18) - 1; e <<= 2;
    uint64_t wm = we - 1;
    fp_copy(t, gw[2][e & wm]); e >>= w;
    fp_mul(t, t, gw[3][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[4][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[5][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[6][e & wm]); //countM++;
}


static void GPOW_i17_e9(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 9) - 1; e <<= 1;
    uint64_t wm = we - 1;
    fp_copy(t, gw[4][e & wm]); e >>= w;
    fp_mul(t, t, gw[5][e & wm]); //countM++;
 e >>= w;
    fp_mul(t, t, gw[6][e & wm]); //countM++;
}


static void GPOW_i19_e9(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 9) - 1; e <<= 3;
    uint64_t wm = we - 1;
    fp_copy(t, gw[4][e & wm]); e >>= w;
    fp_mul(t, t, gw[5][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[6][e & wm]); //countM++;
}


static void GPOW_i23_e9(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 9) - 1; e <<= 3;
    uint64_t wm = we - 1;
    fp_copy(t, gw[5][e & wm]); e >>= w;
    fp_mul(t, t, gw[6][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[7][e & wm]); //countM++;
}


static void GPOW_i24_e9(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 9) - 1;
    uint64_t wm = we - 1;
    fp_copy(t, gw[6][e & wm]); e >>= w;
    fp_mul(t, t, gw[7][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[8][e & wm]); //countM++;
}


static void GPOW_i26_e5(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 5) - 1; e <<= 2;
    uint64_t wm = we - 1;
    fp_copy(t, gw[6][e & wm]); e >>= w;
    fp_mul(t, t, gw[7][e & wm]); //countM++;
}


static void GPOW_i28_e4(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 4) - 1;
    uint64_t wm = we - 1;
    fp_copy(t, gw[7][e & wm]);
}


static void GPOW_i31_e2(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 2) - 1; e <<= 3;
    uint64_t wm = we - 1;
    fp_copy(t, gw[7][e & wm]); e >>= w;
    fp_mul(t, t, gw[8][e & wm]); //countM++;
}


static void GPOW_i31_e4(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 4) - 1; e <<= 3;
    uint64_t wm = we - 1;
    fp_copy(t, gw[7][e & wm]); e >>= w;
    fp_mul(t, t, gw[8][e & wm]); //countM++;
}


static void GPOW_i32_e2(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 2) - 1;
    uint64_t wm = we - 1;
    fp_copy(t, gw[8][e & wm]);
}


void DLPpow2ext(fp_t u_A, fp_t out_t,
                fp_t gw[nw][we], int rlll[64], fp_t rll[we], fp_t fll[we], fp_t gpp[n])
{
    uint64_t one_k = 1;
    int _ii, _tmp;

    fp_t h0_A,   hlp_A,   hlp1_A;
    fp_t h0_PA,  hlp_PA,  hlp1_PA,  h1_PA,  u2_PA;
    fp_t h0_PAH, hlp_PAH, hlp1_PAH, h1_PAH, u2_PAH;
    fp_t h0_PAHL, h1_PAHL, u2_PAHL;
    fp_t h0_PAL,  h1_PAL,  u2_PAL;
    fp_t h0_PALL, h1_PALL, u2_PALL;
    fp_t f_A, f_A_sq, u_X0;
    fp_t h0_X0,  h0_X0P, hlp_X0P, hlp1_X0P, h1_X0P, u2_X0P;
    fp_t h0_X0PL, h1_X0PL, u2_X0PL;
    fp_t f_X0, hlp_X0, hlp1_X0, f_X0_sq, u_X1;
    fp_t h0_X1,  h0_X1P, h1_X1P, u2_X1P;
    fp_t f_X1, f_X1_sq, u_X2;
    fp_t h0_X2;
    fp_t f_X2, f_X2_sq, u_X3;
    fp_t t_X2, t_X1, t_X0, t_A;

    fp_null(h0_A);     fp_new(h0_A);
    fp_null(hlp_A);    fp_new(hlp_A);
    fp_null(hlp1_A);   fp_new(hlp1_A);
    fp_null(h0_PA);    fp_new(h0_PA);
    fp_null(hlp_PA);   fp_new(hlp_PA);
    fp_null(hlp1_PA);  fp_new(hlp1_PA);
    fp_null(h1_PA);    fp_new(h1_PA);
    fp_null(u2_PA);    fp_new(u2_PA);
    fp_null(h0_PAH);   fp_new(h0_PAH);
    fp_null(hlp_PAH);  fp_new(hlp_PAH);
    fp_null(hlp1_PAH); fp_new(hlp1_PAH);
    fp_null(h1_PAH);   fp_new(h1_PAH);
    fp_null(u2_PAH);   fp_new(u2_PAH);
    fp_null(h0_PAHL);  fp_new(h0_PAHL);
    fp_null(h1_PAHL);  fp_new(h1_PAHL);
    fp_null(u2_PAHL);  fp_new(u2_PAHL);
    fp_null(h0_PAL);   fp_new(h0_PAL);
    fp_null(h1_PAL);   fp_new(h1_PAL);
    fp_null(u2_PAL);   fp_new(u2_PAL);
    fp_null(h0_PALL);  fp_new(h0_PALL);
    fp_null(h1_PALL);  fp_new(h1_PALL);
    fp_null(u2_PALL);  fp_new(u2_PALL);
    fp_null(f_A);      fp_new(f_A);
    fp_null(f_A_sq);   fp_new(f_A_sq);
    fp_null(u_X0);     fp_new(u_X0);
    fp_null(h0_X0);    fp_new(h0_X0);
    fp_null(h0_X0P);   fp_new(h0_X0P);
    fp_null(hlp_X0P);  fp_new(hlp_X0P);
    fp_null(hlp1_X0P); fp_new(hlp1_X0P);
    fp_null(h1_X0P);   fp_new(h1_X0P);
    fp_null(u2_X0P);   fp_new(u2_X0P);
    fp_null(h0_X0PL);  fp_new(h0_X0PL);
    fp_null(h1_X0PL);  fp_new(h1_X0PL);
    fp_null(u2_X0PL);  fp_new(u2_X0PL);
    fp_null(f_X0);     fp_new(f_X0);
    fp_null(hlp_X0);   fp_new(hlp_X0);
    fp_null(hlp1_X0);  fp_new(hlp1_X0);
    fp_null(f_X0_sq);  fp_new(f_X0_sq);
    fp_null(u_X1);     fp_new(u_X1);
    fp_null(h0_X1);    fp_new(h0_X1);
    fp_null(h0_X1P);   fp_new(h0_X1P);
    fp_null(h1_X1P);   fp_new(h1_X1P);
    fp_null(u2_X1P);   fp_new(u2_X1P);
    fp_null(f_X1);     fp_new(f_X1);
    fp_null(f_X1_sq);  fp_new(f_X1_sq);
    fp_null(u_X2);     fp_new(u_X2);
    fp_null(h0_X2);    fp_new(h0_X2);
    fp_null(f_X2);     fp_new(f_X2);
    fp_null(f_X2_sq);  fp_new(f_X2_sq);
    fp_null(u_X3);     fp_new(u_X3);
    fp_null(t_X2);     fp_new(t_X2);
    fp_null(t_X1);     fp_new(t_X1);
    fp_null(t_X0);     fp_new(t_X0);
    fp_null(t_A);      fp_new(t_A);

    /* =========================================================================
     * TOP: lb=37, a=18, b=19, nlb1=10, do_hlp_A=True
     * Square 19; capture hlp_A at j=10.
     * ========================================================================= */
    fp_copy(h0_A, u_A);
    for (int _j = 0; _j < 19; _j++) {
        if (_j == 10) fp_copy(hlp_A, h0_A);
        fp_sqr(h0_A, h0_A);
    }
    //countS += 19;


    
    fp_copy(h0_PA, h0_A);
    for (int _j = 0; _j < 9; _j++) {
        if (_j == 5) fp_copy(hlp_PA, h0_PA);
        fp_sqr(h0_PA, h0_PA);
    }
    //countS += 9;

    
    fp_copy(h0_PAH, h0_PA);
    for (int _j = 0; _j < 5; _j++) {
        if (_j == 3) fp_copy(hlp_PAH, h0_PAH);
        fp_sqr(h0_PAH, h0_PAH);
    }
    //countS += 5;

	bn_t tmp_bn;
	bn_null(tmp_bn);bn_new(tmp_bn);
	dig_t d;
    fp_prime_back(tmp_bn, h0_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    _tmp=rlll[d];
    uint64_t c_PAH = (uint64_t)_tmp >> 0;

    GPOW_i28_e4(h1_PAH,   (one_k << 4) - c_PAH, gw);
    SELECT(h1_PAH,   gpp[32], c_PAH == 0, h1_PAH);

    GPOW_i31_e4(hlp1_PAH, (one_k << 4) - c_PAH, gw);
    SELECT(hlp1_PAH, gpp[35], c_PAH == 0, hlp1_PAH);

    fp_mul(hlp_PAH, hlp_PAH, hlp1_PAH); //countM++;

    //countM++;
    fp_mul(u2_PAH, h0_PA, h1_PAH);   /* u_PAH = h0_PA */


    
    fp_copy(h0_PAHL, hlp_PAH);   /* updated hlp_PAH consumed */

    
	fp_prime_back(tmp_bn, h0_PAHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    _tmp=rlll[d];

    uint64_t c_PAHL = (uint64_t)_tmp >> 2;

    GPOW_i32_e2(h1_PAHL, (one_k << 2) - c_PAHL, gw);
    SELECT(h1_PAHL, gpp[34], c_PAHL == 0, h1_PAHL);
    //countM++;
    fp_mul(u2_PAHL, u2_PAH, h1_PAHL);   

    
	fp_prime_back(tmp_bn, u2_PAHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    _tmp=rlll[d];
    uint64_t d_PAHL = (uint64_t)_tmp >> 1;

    
    uint64_t c_PAHL_comb = c_PAHL + ((d_PAHL - 1) % 8) * 4;

    
    c_PAH = c_PAH + ((c_PAHL_comb - 1) % 32) * 16;

    
    GPOW_i19_e9(h1_PA,   (one_k << 9) - c_PAH, gw);
    SELECT(h1_PA,   gpp[28], c_PAH == 0, h1_PA);
    GPOW_i24_e9(hlp1_PA, (one_k << 9) - c_PAH, gw);
    SELECT(hlp1_PA, gpp[33], c_PAH == 0, hlp1_PA);
    fp_mul(hlp_PA, hlp_PA, hlp1_PA); //countM++;
    //countM++;
    fp_mul(u2_PA, h0_A, h1_PA);   /* u_PA = h0_A */

    
    fp_copy(h0_PAL, hlp_PA);   /* updated hlp_PA consumed */

    
	fp_prime_back(tmp_bn, h0_PAL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    _tmp=rlll[d];

    uint64_t c_PAL = (uint64_t)_tmp >> 0;

    
    GPOW_i28_e4(h1_PAL, (one_k << 4) - c_PAL, gw);
    SELECT(h1_PAL, gpp[32], c_PAL == 0, h1_PAL);
    //countM++;
    fp_mul(u2_PAL, u2_PA, h1_PAL);   

    
    fp_copy(h0_PALL, u2_PAL);
    for (int _j = 0; _j < 3; _j++) fp_sqr(h0_PALL, h0_PALL);
    //countS += 3;

    
	fp_prime_back(tmp_bn, h0_PALL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    _tmp=rlll[d];

    uint64_t c_PALL = (uint64_t)_tmp >> 2;

    
    GPOW_i32_e2(h1_PALL, (one_k << 2) - c_PALL, gw);
    SELECT(h1_PALL, gpp[34], c_PALL == 0, h1_PALL);
    //countM++;
    fp_mul(u2_PALL, u2_PAL, h1_PALL);   /* u_PALL = u2_PAL */

    /* leaf-L of PALL: i=34, lb=3 → >> 1 */
	fp_prime_back(tmp_bn, u2_PALL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    _tmp=rlll[d];

    uint64_t d_PALL = (uint64_t)_tmp >> 1;

    
    uint64_t d_PAL = c_PALL + ((d_PALL - 1) % 8) * 4;

    
    uint64_t c_PA = c_PAL + ((d_PAL - 1) % 32) * 16;

    
    uint64_t c_A = c_PAH + ((c_PA - 1) % 512) * 512;

    GPOW_i0_e17(f_A,    ((one_k << 18) - c_A + 1) >> 1, gw);
    SELECT(f_A,    gpp[17], c_A == 0, f_A);
    GPOW_i10_e18(hlp1_A, (one_k << 18) - c_A, gw);
    SELECT(hlp1_A, gpp[28], c_A == 0, hlp1_A);
    fp_mul(hlp_A, hlp_A, hlp1_A); //countM++;
    //countM++;
    //countS++;
    fp_sqr(f_A_sq, f_A);
    fp_mul(u_X0, u_A, f_A_sq); //countM++;

    
    fp_copy(h0_X0, hlp_A);   

    
    fp_copy(h0_X0P, h0_X0);
    for (int _j = 0; _j < 5; _j++) {
        if (_j == 3) fp_copy(hlp_X0P, h0_X0P);
        fp_sqr(h0_X0P, h0_X0P);
    }
    //countS += 5;

    
	fp_prime_back(tmp_bn, h0_X0P);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    _tmp=rlll[d];

    uint64_t c_X0P = (uint64_t)_tmp >> 0;

    GPOW_i28_e4(h1_X0P,   (one_k << 4) - c_X0P, gw);
    SELECT(h1_X0P,   gpp[32], c_X0P == 0, h1_X0P);
    GPOW_i31_e4(hlp1_X0P, (one_k << 4) - c_X0P, gw);
    SELECT(hlp1_X0P, gpp[35], c_X0P == 0, hlp1_X0P);
    fp_mul(hlp_X0P, hlp_X0P, hlp1_X0P); //countM++;
    //countM++;
    fp_mul(u2_X0P, h0_X0, h1_X0P);   /* u_X0P = h0_X0 */

    fp_copy(h0_X0PL, hlp_X0P);   /* updated hlp_X0P consumed */

	fp_prime_back(tmp_bn, h0_X0PL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    _tmp=rlll[d];

    uint64_t c_X0PL = (uint64_t)_tmp >> 2;

    GPOW_i32_e2(h1_X0PL, (one_k << 2) - c_X0PL, gw);
    SELECT(h1_X0PL, gpp[34], c_X0PL == 0, h1_X0PL);
    //countM++;
    fp_mul(u2_X0PL, u2_X0P, h1_X0PL);   /* u_X0PL = u2_X0P */

    
	fp_prime_back(tmp_bn, u2_X0PL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    _tmp=rlll[d];

    uint64_t d_X0PL = (uint64_t)_tmp >> 1;

    
    uint64_t c_X0PL_comb = c_X0PL + ((d_X0PL - 1) % 8) * 4;

    
    uint64_t c_X0 = c_X0P + ((c_X0PL_comb - 1) % 32) * 16;

    GPOW_i17_e9(f_X0,    (one_k << 9) - c_X0, gw);
    SELECT(f_X0,    gpp[26], c_X0 == 0, f_X0);

    fp_sqr(f_X0_sq, f_X0);

    fp_mul(u_X1, u_X0, f_X0_sq); //countM++;

    fp_copy(h0_X1, u_X1);   /* updated hlp_X0 consumed */
fp_t hlp_X1;
fp_null(hlp_X1);fp_null(hlp_X1);
    for (int _j = 0; _j < 5; _j++) {
        if (_j == 3) fp_copy(hlp_X1, h0_X1);
        fp_sqr(h0_X1, h0_X1);
    }

    fp_copy(h0_X1P, h0_X1);
    for (int _j = 0; _j < 3; _j++) fp_sqr(h0_X1P, h0_X1P);
    //countS += 3;
 
    
	fp_prime_back(tmp_bn, h0_X1P);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    _tmp=rlll[d];

    uint64_t c_X1H = (uint64_t)_tmp >> 2;


    GPOW_i32_e2(h1_X1P, (one_k << 2) - c_X1H, gw);
    SELECT(h1_X1P, gpp[34], c_X1H == 0, h1_X1P);
    //countM++;
    fp_mul(u2_X1P, h0_X1, h1_X1P);   /* u_X1P = h0_X1 */
 
    
	fp_prime_back(tmp_bn, u2_X1P);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    _tmp=rlll[d];

    uint64_t d_X1L = (uint64_t)_tmp >> 1;
 
    /* combine X1P: b=3, a=2 → gives c_X1 */
    uint64_t c_X1 = c_X1H + ((d_X1L - 1) % 8) * 4;

    GPOW_i26_e5(f_X1, (one_k << 5) - c_X1, gw);
    SELECT(f_X1, gpp[31], c_X1 == 0, f_X1);
    //countM++;
    //countS++;
    fp_sqr(f_X1_sq, f_X1);
    fp_mul(u_X2, u_X1, f_X1_sq); //countM++;

    
    fp_copy(h0_X2, u_X2);
    for (int _j = 0; _j < 3; _j++) fp_sqr(h0_X2, h0_X2);
    //countS += 3;

    
	fp_prime_back(tmp_bn, h0_X2);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    _tmp=rlll[d];

    uint64_t c_X2 = (uint64_t)_tmp >> 2;

    GPOW_i31_e2(f_X2, (one_k << 2) - c_X2, gw);
    SELECT(f_X2, gpp[33], c_X2 == 0, f_X2);
    //countM++;
    //countS++;
    fp_sqr(f_X2_sq, f_X2);
    fp_mul(u_X3, u_X2, f_X2_sq); //countM++;
	//fp_print(u_X3);

    
	fp_prime_back(tmp_bn, u_X3);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3f;
    _tmp=rlll[d];


    uint64_t c1_X3 = (uint64_t)_tmp >> 1;


    fp_mul(t_X2,f_X2, fll[c1_X3 << 1]);   



    fp_mul(t_X1, f_X1, t_X2); //countM++;

    fp_mul(t_X0, f_X0, t_X1); //countM++;


    fp_mul(t_A, f_A, t_X0); //countM++;


    fp_copy(out_t, t_A);

    /* ---- free all ---- */
    fp_free(h0_A);     fp_free(hlp_A);    fp_free(hlp1_A);
    fp_free(h0_PA);    fp_free(hlp_PA);   fp_free(hlp1_PA);
    fp_free(h1_PA);    fp_free(u2_PA);
    fp_free(h0_PAH);   fp_free(hlp_PAH);  fp_free(hlp1_PAH);
    fp_free(h1_PAH);   fp_free(u2_PAH);
    fp_free(h0_PAHL);  fp_free(h1_PAHL);  fp_free(u2_PAHL);
    fp_free(h0_PAL);   fp_free(h1_PAL);   fp_free(u2_PAL);
    fp_free(h0_PALL);  fp_free(h1_PALL);  fp_free(u2_PALL);
    fp_free(f_A);      fp_free(f_A_sq);   fp_free(u_X0);
    fp_free(h0_X0);    fp_free(h0_X0P);   fp_free(hlp_X0P);
    fp_free(hlp1_X0P); fp_free(h1_X0P);   fp_free(u2_X0P);
    fp_free(h0_X0PL);  fp_free(h1_X0PL);  fp_free(u2_X0PL);
    fp_free(f_X0);     fp_free(hlp_X0);   fp_free(hlp1_X0);
    fp_free(f_X0_sq);  fp_free(u_X1);
    fp_free(h0_X1);    fp_free(h0_X1P);   fp_free(h1_X1P);   fp_free(u2_X1P);
    fp_free(f_X1);     fp_free(f_X1_sq);  fp_free(u_X2);
    fp_free(h0_X2);    fp_free(f_X2);     fp_free(f_X2_sq);  fp_free(u_X3);
    fp_free(t_X2);     fp_free(t_X1);     fp_free(t_X0);     fp_free(t_A);
	bn_free(tmp_bn);
}

/* =========================================================================
 * sqrt_ext
 * ========================================================================= */
void sqrt_ext(fp_t x, fp_t y, bn_t e_exp,
              fp_t gw[nw][we],int rlll[64], fp_t rll[we], fp_t fll[we], fp_t gpp[n])
{
    fp_t u, v, w_, t;
    fp_null(u); fp_null(v); fp_null(w_); fp_null(t);
    RLC_TRY {
        fp_new(u); fp_new(v); fp_new(w_); fp_new(t);
        fp_exp(v,  x, e_exp);
        fp_mul(w_, x, v);  //countM++;
        fp_mul(u,  w_, v); //countM++;
//fp_print(u);
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

int rlll[64]={0};

rlll[ 0x1 ]= 0 ;
rlll[ 0x5 ]= 1 ;
rlll[ 0x1d ]= 2 ;
rlll[ 0x22 ]= 3 ;
rlll[ 0x23 ]= 4 ;
rlll[ 0x3 ]= 5 ;
rlll[ 0x2c ]= 6 ;
rlll[ 0x36 ]= 7 ;
rlll[ 0x0 ]= 8 ;
rlll[ 0x3c ]= 9 ;
rlll[ 0x24 ]= 10 ;
rlll[ 0x1f ]= 11 ;
rlll[ 0x1e ]= 12 ;
rlll[ 0x3e ]= 13 ;
rlll[ 0x15 ]= 14 ;
rlll[ 0xb ]= 15 ;

    for (i = 0; i < nw; i++)
        for (j = 0; j < we; j++) { fp_null(gw[i][j]); fp_new(gw[i][j]); }
    for (i = 0; i < we; i++) { fp_null(rll[i]); fp_new(rll[i]); }
    for (i = 0; i < we; i++) { fp_null(fll[i]); fp_new(fll[i]); }
    for (i = 0; i < n;  i++) { fp_null(gpp[i]); fp_new(gpp[i]); }
    fp_null(b);   fp_new(b);   fp_null(y);  fp_new(y);
    fp_null(g);   fp_new(g);   fp_null(z);  fp_new(z);
    fp_null(h);   fp_new(h);   fp_null(hh); fp_new(hh);
    bn_null(tmp); bn_new(tmp);
    bn_null(m);   bn_new(m);
    bn_null(e);   bn_new(e);

    fp_rand(b);
    while (fp_is_sqr(b) != 1) fp_rand(b);
//    fp_set_dig(b,4);
    
    
    bn_read_str(tmp,"b",96,16);
    fp_prime_conv(z,tmp);
    
    bn_read_str(m,"ba29500be5eeb41528150f09d4eb38e93cac77669619e1d1b6156f110ecdcab5f527a4d95af73ebeb95690032ee595c74e5000ada8a9600000408f",118,16);
    fp_exp(g,z,m);
    
    bn_read_str(e,"5d14a805f2f75a0a940a8784ea759c749e563bb34b0cf0e8db0ab7888766e55afa93d26cad7b9f5f5cab48019772cae3a7280056d454b000002047",118,16);
    
    bn_t a1;
    bn_new(a1);
    bn_set_dig(a1,1);
    bn_lsh(a1,a1,n-w);
    fp_exp(h,g,a1);

    
//    fp_print(h);
    fp_srt(hh, h);
    fp_inv(hh, hh);
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
    
    return 0;
}
