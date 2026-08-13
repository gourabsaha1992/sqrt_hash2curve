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
#define we  16       /* 2^w = 16 */
#define n   96
#define nw  24       /* ceil(96/4) */

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
                    fp_t gw[nw][we], fp_t fll[we], fp_t gpp[n])
{
    bn_t temp, one;
    bn_null(temp); bn_new(temp);
    bn_null(one);  bn_new(one);
    bn_set_dig(one, 1);
    for (int v = 0; v < we; v++) {
        bn_set_dig(temp, v);
//        fp_exp(rll[v], h,  temp);
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

/* =========================================================================
 * SELECT
 * ========================================================================= */
static void SELECT(fp_t a0, fp_t a1, bool ctl, fp_t out)
{
    if (ctl) fp_copy(out, a1);
    else     fp_copy(out, a0);
}

/* =========================================================================
 * GPOW functions for w=4
 *
 * GPOW(i=0,   elen=47): ri=0, row=0,  adj=47, rows=12, 11 muls
 * GPOW(i=24,  elen=48): ri=0, row=6,  adj=48, rows=12, 11 muls
 * GPOW(i=47,  elen=24): ri=3, row=11, adj=27, rows=7,   6 muls
 * GPOW(i=48,  elen=24): ri=0, row=12, adj=24, rows=6,   5 muls
 * GPOW(i=60,  elen=24): ri=0, row=15, adj=24, rows=6,   5 muls
 * GPOW(i=71,  elen=12): ri=3, row=17, adj=15, rows=4,   3 muls
 * GPOW(i=72,  elen=12): ri=0, row=18, adj=12, rows=3,   2 muls
 * GPOW(i=78,  elen=12): ri=2, row=19, adj=14, rows=4,   3 muls
 * GPOW(i=83,  elen=6):  ri=3, row=20, adj=9,  rows=3,   2 muls
 * GPOW(i=84,  elen=6):  ri=0, row=21, adj=6,  rows=2,   1 mul
 * GPOW(i=89,  elen=3):  ri=1, row=22, adj=4,  rows=1,   0 muls
 * GPOW(i=90,  elen=3):  ri=2, row=22, adj=5,  rows=2,   1 mul
 * ========================================================================= */

/* GPOW(i=0, elen=47): row=0, 11 muls */
static void GPOW_i0_e47(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 47) - 1;
    uint64_t wm = (1 << w) - 1;
    fp_copy(t, gw[0][e & wm]); e >>= w;
    fp_mul(t, t, gw[1][e & wm]);  //countM++; 
e >>= w;
    fp_mul(t, t, gw[2][e & wm]);  //countM++; 
e >>= w;
    fp_mul(t, t, gw[3][e & wm]);  //countM++; 
e >>= w;
    fp_mul(t, t, gw[4][e & wm]);  //countM++; 
e >>= w;
    fp_mul(t, t, gw[5][e & wm]);  //countM++; 
e >>= w;
    fp_mul(t, t, gw[6][e & wm]);  //countM++; 
e >>= w;
    fp_mul(t, t, gw[7][e & wm]);  //countM++; 
e >>= w;
    fp_mul(t, t, gw[8][e & wm]);  //countM++; 
e >>= w;
    fp_mul(t, t, gw[9][e & wm]);  //countM++; 
e >>= w;
    fp_mul(t, t, gw[10][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[11][e & wm]); //countM++;
}

/* GPOW(i=24, elen=48): row=6, 11 muls */
static void GPOW_i24_e48(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 48) - 1;
    uint64_t wm = (1 << w) - 1;
    fp_copy(t, gw[6][e & wm]);  e >>= w;
    fp_mul(t, t, gw[7][e & wm]);  //countM++; 
e >>= w;
    fp_mul(t, t, gw[8][e & wm]);  //countM++; 
e >>= w;
    fp_mul(t, t, gw[9][e & wm]);  //countM++; 
e >>= w;
    fp_mul(t, t, gw[10][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[11][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[12][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[13][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[14][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[15][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[16][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[17][e & wm]); //countM++;
}

/* GPOW(i=47, elen=24): ri=3, row=11, e<<=3, adj=27, rows=7, 6 muls */
static void GPOW_i47_e24(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 24) - 1;
    e <<= 3;
    uint64_t wm = (1 << w) - 1;
    fp_copy(t, gw[11][e & wm]); e >>= w;
    fp_mul(t, t, gw[12][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[13][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[14][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[15][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[16][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[17][e & wm]); //countM++;
}

/* GPOW(i=48, elen=24): row=12, adj=24, rows=6, 5 muls */
static void GPOW_i48_e24(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 24) - 1;
    uint64_t wm = (1 << w) - 1;
    fp_copy(t, gw[12][e & wm]); e >>= w;
    fp_mul(t, t, gw[13][e & wm]); countM++; e >>= w;
    fp_mul(t, t, gw[14][e & wm]); countM++; e >>= w;
    fp_mul(t, t, gw[15][e & wm]); countM++; e >>= w;
    fp_mul(t, t, gw[16][e & wm]); countM++; e >>= w;
    fp_mul(t, t, gw[17][e & wm]); countM++;
}

/* GPOW(i=60, elen=24): row=15, adj=24, rows=6, 5 muls */
static void GPOW_i60_e24(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 24) - 1;
    uint64_t wm = (1 << w) - 1;
    fp_copy(t, gw[15][e & wm]); e >>= w;
    fp_mul(t, t, gw[16][e & wm]); countM++; e >>= w;
    fp_mul(t, t, gw[17][e & wm]); countM++; e >>= w;
    fp_mul(t, t, gw[18][e & wm]); countM++; e >>= w;
    fp_mul(t, t, gw[19][e & wm]); countM++; e >>= w;
    fp_mul(t, t, gw[20][e & wm]); countM++;
}

/* GPOW(i=71, elen=12): ri=3, row=17, e<<=3, adj=15, rows=4, 3 muls */
static void GPOW_i71_e12(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 12) - 1;
    e <<= 3;
    uint64_t wm = (1 << w) - 1;
    fp_copy(t, gw[17][e & wm]); e >>= w;
    fp_mul(t, t, gw[18][e & wm]); countM++; e >>= w;
    fp_mul(t, t, gw[19][e & wm]); countM++; e >>= w;
    fp_mul(t, t, gw[20][e & wm]); countM++;
}

/* GPOW(i=72, elen=12): row=18, adj=12, rows=3, 2 muls */
static void GPOW_i72_e12(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 12) - 1;
    uint64_t wm = (1 << w) - 1;
    fp_copy(t, gw[18][e & wm]); e >>= w;
    fp_mul(t, t, gw[19][e & wm]); countM++; e >>= w;
    fp_mul(t, t, gw[20][e & wm]); countM++;
}

/* GPOW(i=78, elen=12): ri=2, row=19, e<<=2, adj=14, rows=4, 3 muls */
static void GPOW_i78_e12(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 12) - 1;
    e <<= 2;
    uint64_t wm = (1 << w) - 1;
    fp_copy(t, gw[19][e & wm]); e >>= w;
    fp_mul(t, t, gw[20][e & wm]); countM++; e >>= w;
    fp_mul(t, t, gw[21][e & wm]); countM++; e >>= w;
    fp_mul(t, t, gw[22][e & wm]); countM++;
}

/* GPOW(i=83, elen=6): ri=3, row=20, e<<=3, adj=9, rows=3, 2 muls */
static void GPOW_i83_e6(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 6) - 1;
    e <<= 3;
    uint64_t wm = (1 << w) - 1;
    fp_copy(t, gw[20][e & wm]); e >>= w;
    fp_mul(t, t, gw[21][e & wm]); countM++; e >>= w;
    fp_mul(t, t, gw[22][e & wm]); countM++;
}

/* GPOW(i=84, elen=6): row=21, adj=6, rows=2, 1 mul */
static void GPOW_i84_e6(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 6) - 1;
    uint64_t wm = (1 << w) - 1;
    fp_copy(t, gw[21][e & wm]); e >>= w;
    fp_mul(t, t, gw[22][e & wm]); countM++;
}

/* GPOW(i=89, elen=3): ri=1, row=22, e<<=1, adj=4, rows=1, 0 muls */
static void GPOW_i89_e3(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 3) - 1;
    e <<= 1;
    uint64_t wm = (1 << w) - 1;
    fp_copy(t, gw[22][e & wm]);
}

/* GPOW(i=90, elen=3): ri=2, row=22, e<<=2, adj=5, rows=2, 1 mul */
static void GPOW_i90_e3(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 3) - 1;
    e <<= 2;
    uint64_t wm = (1 << w) - 1;
    fp_copy(t, gw[22][e & wm]); e >>= w;
    fp_mul(t, t, gw[23][e & wm]); countM++;
}


/* =========================================================================
 * DLPpow2ext for w=4 — recursion-free, stack-free.
 * Follows corrected sqrt_ext_w4.sage exactly.
 *
 * Key:
 *   BASE: i=93, lb=3 -> >> (w-lb)=(4-3)=1
 *   Expected: M=124, S=191
 * ========================================================================= */
void DLPpow2ext(fp_t u_A, fp_t out_t,
                fp_t gw[nw][we], int rll[128], fp_t fll[we], fp_t gpp[n])
{
    uint64_t one_k = 1;
    int _ii, _tmp;

    /* ---- temporaries ---- */
    fp_t h0_A, hlp_A, f_A, f_A_sq;
    fp_t h0_PA, hlp_PA, h1_PA, hlp1_PA, u2_PA;
    fp_t h0_PAH, hlp_PAH, h1_PAH, hlp1_PAH, u2_PAH;
    fp_t h0_PAHH, h1_PAHH, u2_PAHH;
    fp_t h0_PAHL, h1_PAHL, u2_PAHL;
    fp_t h0_PAL, h1_PAL, u2_PAL;
    fp_t h0_PALH, h1_PALH, u2_PALH;
    fp_t h0_PALL, h1_PALL, u2_PALL;
    fp_t hlp1_A;
    /* EXT2 */
    fp_t u_X0, h0_X0;
    fp_t h0_X0P, hlp_X0P, h1_X0P, hlp1_X0P, u2_X0P;
    fp_t h0_X0PH, h1_X0PH, u2_X0PH;
    fp_t h0_X0PL, h1_X0PL, u2_X0PL;
    fp_t f_X0, f_X0_sq;
    fp_t u_X1, hlp_X1, h0_X1;
    fp_t h0_X1P, h1_X1P, u2_X1P;
    fp_t f_X1, hlp1_X1, f_X1_sq;
    fp_t u_X2, h0_X2;
    fp_t h0_X2P, h1_X2P, u2_X2P;
    fp_t f_X2, f_X2_sq;
    fp_t u_X3, h0_X3;
    fp_t f_X3, f_X3_sq;
    fp_t u_X4, z_X4;
    fp_t t_X3, t_X2, t_X1, t_X0, t_A;
    bn_t tmp_bn;
    dig_t d;

    fp_null(h0_A);     fp_new(h0_A);
    fp_null(hlp_A);    fp_new(hlp_A);
    fp_null(f_A);      fp_new(f_A);
    fp_null(f_A_sq);   fp_new(f_A_sq);
    fp_null(h0_PA);    fp_new(h0_PA);
    fp_null(hlp_PA);   fp_new(hlp_PA);
    fp_null(h1_PA);    fp_new(h1_PA);
    fp_null(hlp1_PA);  fp_new(hlp1_PA);
    fp_null(u2_PA);    fp_new(u2_PA);
    fp_null(h0_PAH);   fp_new(h0_PAH);
    fp_null(hlp_PAH);  fp_new(hlp_PAH);
    fp_null(h1_PAH);   fp_new(h1_PAH);
    fp_null(hlp1_PAH); fp_new(hlp1_PAH);
    fp_null(u2_PAH);   fp_new(u2_PAH);
    fp_null(h0_PAHH);  fp_new(h0_PAHH);
    fp_null(h1_PAHH);  fp_new(h1_PAHH);
    fp_null(u2_PAHH);  fp_new(u2_PAHH);
    fp_null(h0_PAHL);  fp_new(h0_PAHL);
    fp_null(h1_PAHL);  fp_new(h1_PAHL);
    fp_null(u2_PAHL);  fp_new(u2_PAHL);
    fp_null(h0_PAL);   fp_new(h0_PAL);
    fp_null(h1_PAL);   fp_new(h1_PAL);
    fp_null(u2_PAL);   fp_new(u2_PAL);
    fp_null(h0_PALH);  fp_new(h0_PALH);
    fp_null(h1_PALH);  fp_new(h1_PALH);
    fp_null(u2_PALH);  fp_new(u2_PALH);
    fp_null(h0_PALL);  fp_new(h0_PALL);
    fp_null(h1_PALL);  fp_new(h1_PALL);
    fp_null(u2_PALL);  fp_new(u2_PALL);
    fp_null(hlp1_A);   fp_new(hlp1_A);
    fp_null(u_X0);     fp_new(u_X0);
    fp_null(h0_X0);    fp_new(h0_X0);
    fp_null(h0_X0P);   fp_new(h0_X0P);
    fp_null(hlp_X0P);  fp_new(hlp_X0P);
    fp_null(h1_X0P);   fp_new(h1_X0P);
    fp_null(hlp1_X0P); fp_new(hlp1_X0P);
    fp_null(u2_X0P);   fp_new(u2_X0P);
    fp_null(h0_X0PH);  fp_new(h0_X0PH);
    fp_null(h1_X0PH);  fp_new(h1_X0PH);
    fp_null(u2_X0PH);  fp_new(u2_X0PH);
    fp_null(h0_X0PL);  fp_new(h0_X0PL);
    fp_null(h1_X0PL);  fp_new(h1_X0PL);
    fp_null(u2_X0PL);  fp_new(u2_X0PL);
    fp_null(f_X0);     fp_new(f_X0);
    fp_null(f_X0_sq);  fp_new(f_X0_sq);
    fp_null(u_X1);     fp_new(u_X1);
    fp_null(hlp_X1);   fp_new(hlp_X1);
    fp_null(h0_X1);    fp_new(h0_X1);
    fp_null(h0_X1P);   fp_new(h0_X1P);
    fp_null(h1_X1P);   fp_new(h1_X1P);
    fp_null(u2_X1P);   fp_new(u2_X1P);
    fp_null(f_X1);     fp_new(f_X1);
    fp_null(hlp1_X1);  fp_new(hlp1_X1);
    fp_null(f_X1_sq);  fp_new(f_X1_sq);
    fp_null(u_X2);     fp_new(u_X2);
    fp_null(h0_X2);    fp_new(h0_X2);
    fp_null(h0_X2P);   fp_new(h0_X2P);
    fp_null(h1_X2P);   fp_new(h1_X2P);
    fp_null(u2_X2P);   fp_new(u2_X2P);
    fp_null(f_X2);     fp_new(f_X2);
    fp_null(f_X2_sq);  fp_new(f_X2_sq);
    fp_null(u_X3);     fp_new(u_X3);
    fp_null(h0_X3);    fp_new(h0_X3);
    fp_null(f_X3);     fp_new(f_X3);
    fp_null(f_X3_sq);  fp_new(f_X3_sq);
    fp_null(u_X4);     fp_new(u_X4);
    fp_null(z_X4);     fp_new(z_X4);
    fp_null(t_X3);     fp_new(t_X3);
    fp_null(t_X2);     fp_new(t_X2);
    fp_null(t_X1);     fp_new(t_X1);
    fp_null(t_X0);     fp_new(t_X0);
    fp_null(t_A);      fp_new(t_A);

    /* =========================================================================
     * TOP LEVEL: lb=96, a=48, b=48, nlb1=24, do_hlp_A=True
     * sq 48, capture hlp_A at j=24
     * ========================================================================= */
    fp_copy(h0_A, u_A);
    for (int _j = 0; _j < 48; _j++) {
        if (_j == 24) fp_copy(hlp_A, h0_A);
        fp_sqr(h0_A, h0_A);
    }
    countS += 48;
//    fp_print(h0_A);

    /* =========================================================================
     * PA (i=48): sq 24, hlp_PA at j=12
     * ========================================================================= */
    fp_copy(h0_PA, h0_A);
    for (int _j = 0; _j < 24; _j++) {
        if (_j == 12) fp_copy(hlp_PA, h0_PA);
        fp_sqr(h0_PA, h0_PA);
    }
    countS += 24;
//    fp_print(h0_PA);

    /* =========================================================================
     * PAH (i=72): sq 12, hlp_PAH at j=6
     * ========================================================================= */
    fp_copy(h0_PAH, h0_PA);
    for (int _j = 0; _j < 12; _j++) {
        if (_j == 6) fp_copy(hlp_PAH, h0_PAH);
        fp_sqr(h0_PAH, h0_PAH);
    }
    countS += 12;
//    fp_print(h0_PAH);

    /* =========================================================================
     * PAHH (i=84): sq 6, do_hlp=False (cost=3>=nlb1=3)
     * c_PAHH = DLPpow2(h0_PAHH, i=90) via DLP90
     * ========================================================================= */
    fp_copy(h0_PAHH, h0_PAH);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_PAHH, h0_PAHH);
    countS += 6;
//    fp_print(h0_PAHH);
    /* c_PAHH = DLPpow2(h0_PAHH, i=90)  [HIGH of PAHH, Sage: u_PAHHH=h0_PAHH] */
    uint64_t c_PAHH;
    /* DLP(i=90) [PAHHH]: sq3, BASE(93)>>1, GPOW_i90_e3, BASE(93)>>1  [HIGH of PAHH] */
    {
    fp_t h0_PAHHH, h1_PAHHH, u2_PAHHH;
    fp_null(h0_PAHHH); fp_new(h0_PAHHH);
    fp_null(h1_PAHHH); fp_new(h1_PAHHH);
    fp_null(u2_PAHHH); fp_new(u2_PAHHH);
    fp_copy(h0_PAHHH, h0_PAHH);
    fp_sqr(h0_PAHHH, h0_PAHHH); fp_sqr(h0_PAHHH, h0_PAHHH); fp_sqr(h0_PAHHH, h0_PAHHH);
    countS += 3;
    //_tmp = 0;
    fp_prime_back(tmp_bn, h0_PAHHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_PAHHH) == RLC_EQ) _tmp = _ii;
    (c_PAHH) = (uint64_t)_tmp >> 1;
    
    GPOW_i90_e3(h1_PAHHH, (one_k << 3) - (c_PAHH), gw);
    SELECT(h1_PAHHH, gpp[93], (c_PAHH) == 0, h1_PAHHH);
    countM++;
    fp_mul(u2_PAHHH, h0_PAHH, h1_PAHHH);
    //_tmp = 0;
    fp_prime_back(tmp_bn, u2_PAHHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_PAHHH) == RLC_EQ) _tmp = _ii;
    { uint64_t _d_PAHHH = (uint64_t)_tmp >> 1;
      (c_PAHH) = (c_PAHH) + ((_d_PAHHH - 1) % 8) * 8; }
    fp_free(h0_PAHHH); fp_free(h1_PAHHH); fp_free(u2_PAHHH);
    }
    
    //printf("\n%lx\n",c_PAHH);

    /* correct PAHH at i=84, no hlp */
    GPOW_i84_e6(h1_PAHH, (one_k << 6) - c_PAHH, gw);
    SELECT(h1_PAHH, gpp[84 + 6], c_PAHH == 0, h1_PAHH);
    countM++;
    fp_mul(u2_PAHH, h0_PAH, h1_PAHH);   /* u_PAHH = h0_PAH */
//    fp_print(u2_PAHH);
    
    
    //ok
    /* d_PAH = DLPpow2(u2_PAHH, i=90)  [LOW of PAHH, Sage: u_PAHHL=u2_PAHH] */
    uint64_t d_PAH;
    /* DLP(i=90) [PAHHL]: sq3, BASE(93)>>1, GPOW_i90_e3, BASE(93)>>1  [LOW  of PAHH] */
    {
    fp_t h0_PAHHL, h1_PAHHL, u2_PAHHL;
    fp_null(h0_PAHHL); fp_new(h0_PAHHL);
    fp_null(h1_PAHHL); fp_new(h1_PAHHL);
    fp_null(u2_PAHHL); fp_new(u2_PAHHL);
    fp_copy(h0_PAHHL, u2_PAHH);
    fp_sqr(h0_PAHHL, h0_PAHHL); fp_sqr(h0_PAHHL, h0_PAHHL); fp_sqr(h0_PAHHL, h0_PAHHL);
    countS += 3;
    //_tmp = 0;
    fp_prime_back(tmp_bn, h0_PAHHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_PAHHL) == RLC_EQ) _tmp = _ii;
    (d_PAH) = (uint64_t)_tmp >> 1;
    GPOW_i90_e3(h1_PAHHL, (one_k << 3) - (d_PAH), gw);
    SELECT(h1_PAHHL, gpp[93], (d_PAH) == 0, h1_PAHHL);
    countM++;
    fp_mul(u2_PAHHL, u2_PAHH, h1_PAHHL);
//    fp_print(u2_PAHHL);
    //_tmp = 0;
    fp_prime_back(tmp_bn, u2_PAHHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_PAHHL) == RLC_EQ) _tmp = _ii;
    { uint64_t _d_PAHHL = (uint64_t)_tmp >> 1;
      (d_PAH) = (d_PAH) + ((_d_PAHHL - 1) % 8) * 8; }
    fp_free(h0_PAHHL); fp_free(h1_PAHHL); fp_free(u2_PAHHL);
    }

    uint64_t c_PAH = c_PAHH + ((d_PAH - 1) % 64) * 64;   /* combine PAHH, b=a=6 */
//    printf("\n%lx\n%lx\n",c_PAH,d_PAH);
    /* =========================================================================
     * correct PAH at i=72, a=12 (hlp_PAH present)
     * ========================================================================= */
    GPOW_i72_e12(h1_PAH,   (one_k << 12) - c_PAH, gw);
    SELECT(h1_PAH,   gpp[72 + 12], c_PAH == 0, h1_PAH);
    GPOW_i78_e12(hlp1_PAH, (one_k << 12) - c_PAH, gw);
    SELECT(hlp1_PAH, gpp[78 + 12], c_PAH == 0, hlp1_PAH);
    fp_mul(hlp_PAH, hlp_PAH, hlp1_PAH); countM++;
    countM++;
    fp_mul(u2_PAH, h0_PA, h1_PAH);   /* u_PAH = h0_PA */

    /* =========================================================================
     * PAHL (i=84): helper=hlp_PAH -> else branch: h0_PAHL=hlp_PAH
     *
     * c_PAHL = DLPpow2(h0_PAHL, i=90)  [HIGH of PAHL, Sage: u_PAHLH=h0_PAHL]
     * correct PAHL at i=84
     * d_PA   = DLPpow2(u2_PAHL, i=90)  [LOW  of PAHL, Sage: u_PAHLL=u2_PAHL]
     * c_PA (from PAHL) = c_PAHL + ((d_PA-1)%64)*64
     * c_PA (from PAH)  = c_PAH  + ((c_PA_pahl-1)%(2^12))*(2^12)
     * ========================================================================= */
    fp_copy(h0_PAHL, hlp_PAH);   /* else branch */

    uint64_t c_PAHL;
    /* DLP(i=90) [PAHLH]: sq3, BASE(93)>>1, GPOW_i90_e3, BASE(93)>>1  [HIGH of PAHL] */
    {
    fp_t h0_PAHLH, h1_PAHLH, u2_PAHLH;
    fp_null(h0_PAHLH); fp_new(h0_PAHLH);
    fp_null(h1_PAHLH); fp_new(h1_PAHLH);
    fp_null(u2_PAHLH); fp_new(u2_PAHLH);
    fp_copy(h0_PAHLH, h0_PAHL);
    fp_sqr(h0_PAHLH, h0_PAHLH); fp_sqr(h0_PAHLH, h0_PAHLH); fp_sqr(h0_PAHLH, h0_PAHLH);
    countS += 3;
    //_tmp = 0;
    fp_prime_back(tmp_bn, h0_PAHLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_PAHLH) == RLC_EQ) _tmp = _ii;
    (c_PAHL) = (uint64_t)_tmp >> 1;
    GPOW_i90_e3(h1_PAHLH, (one_k << 3) - (c_PAHL), gw);
    SELECT(h1_PAHLH, gpp[93], (c_PAHL) == 0, h1_PAHLH);
    countM++;
    fp_mul(u2_PAHLH, h0_PAHL, h1_PAHLH);
    //_tmp = 0;
    fp_prime_back(tmp_bn, u2_PAHLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_PAHLH) == RLC_EQ) _tmp = _ii;
    { uint64_t _d_PAHLH = (uint64_t)_tmp >> 1;
      (c_PAHL) = (c_PAHL) + ((_d_PAHLH - 1) % 8) * 8; }
    fp_free(h0_PAHLH); fp_free(h1_PAHLH); fp_free(u2_PAHLH);
    }

    GPOW_i84_e6(h1_PAHL, (one_k << 6) - c_PAHL, gw);
    SELECT(h1_PAHL, gpp[84 + 6], c_PAHL == 0, h1_PAHL);
    countM++;
    fp_mul(u2_PAHL, u2_PAH, h1_PAHL);   /* u_PAHL = u2_PAH */

    uint64_t d_PA;
    /* DLP(i=90) [PAHLL]: sq3, BASE(93)>>1, GPOW_i90_e3, BASE(93)>>1  [LOW  of PAHL] */
    {
    fp_t h0_PAHLL, h1_PAHLL, u2_PAHLL;
    fp_null(h0_PAHLL); fp_new(h0_PAHLL);
    fp_null(h1_PAHLL); fp_new(h1_PAHLL);
    fp_null(u2_PAHLL); fp_new(u2_PAHLL);
    fp_copy(h0_PAHLL, u2_PAHL);
    fp_sqr(h0_PAHLL, h0_PAHLL); fp_sqr(h0_PAHLL, h0_PAHLL); fp_sqr(h0_PAHLL, h0_PAHLL);
    countS += 3;
    //_tmp = 0;
    fp_prime_back(tmp_bn, h0_PAHLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_PAHLL) == RLC_EQ) _tmp = _ii;
    (d_PA) = (uint64_t)_tmp >> 1;
    GPOW_i90_e3(h1_PAHLL, (one_k << 3) - (d_PA), gw);
    SELECT(h1_PAHLL, gpp[93], (d_PA) == 0, h1_PAHLL);
    countM++;
    fp_mul(u2_PAHLL, u2_PAHL, h1_PAHLL);
    //_tmp = 0;
    fp_prime_back(tmp_bn, u2_PAHLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_PAHLL) == RLC_EQ) _tmp = _ii;
    { uint64_t _d_PAHLL = (uint64_t)_tmp >> 1;
      (d_PA) = (d_PA) + ((_d_PAHLL - 1) % 8) * 8; }
    fp_free(h0_PAHLL); fp_free(h1_PAHLL); fp_free(u2_PAHLL);
    }

    uint64_t c_PA_pahl = c_PAHL + ((d_PA   - 1) % 64) * 64;   /* combine PAHL, b=a=6 */
    uint64_t c_PA      = c_PAH  + ((c_PA_pahl - 1) % ((one_k << 12))) * (one_k << 12);
    /* c_PA = full result of DLPpow2(h0_PA, i=72) */

    /* =========================================================================
     * correct PA at i=48, a=24 (hlp_PA present)
     * ========================================================================= */
    GPOW_i48_e24(h1_PA,   (one_k << 24) - c_PA, gw);
    SELECT(h1_PA,   gpp[48 + 24], c_PA == 0, h1_PA);
    GPOW_i60_e24(hlp1_PA, (one_k << 24) - c_PA, gw);
    SELECT(hlp1_PA, gpp[60 + 24], c_PA == 0, hlp1_PA);
    fp_mul(hlp_PA, hlp_PA, hlp1_PA); countM++;
    countM++;
    fp_mul(u2_PA, h0_A, h1_PA);   /* u_PA = h0_A */

    /* =========================================================================
     * PAL (i=72): helper=hlp_PA -> else branch: h0_PAL=hlp_PA
     *
     * PALH (i=84): sq 6, do_hlp=False
     *   c_PALH = DLPpow2(h0_PALH, i=90)  [HIGH of PALH, u_PALHH=h0_PALH]
     *   correct PALH at i=84
     *   d_PAL  = DLPpow2(u2_PALH, i=90)  [LOW  of PALH, u_PALHL=u2_PALH]
     *   c_PAL  = c_PALH + ((d_PAL-1)%64)*64
     * correct PAL at i=72, no hlp
     * PALL (i=84): sq 6, do_hlp=False
     *   c_PALL = DLPpow2(h0_PALL, i=90)  [HIGH of PALL, u_PALLH=h0_PALL]
     *   correct PALL at i=84
     *   d_PA2  = DLPpow2(u2_PALL, i=90)  [LOW  of PALL, u_PALLL=u2_PALL]
     *   d_PA3  = c_PALL + ((d_PA2-1)%64)*64
     * d_PA = c_PAL + ((d_PA3-1)%(2^12))*(2^12)
     * c_A  = c_PA  + ((d_PA -1)%(2^24))*(2^24)
     * ========================================================================= */
    fp_copy(h0_PAL, hlp_PA);   /* else branch */

    /* PALH: sq 6 */
    fp_copy(h0_PALH, h0_PAL);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_PALH, h0_PALH);
    countS += 6;

    uint64_t c_PALH;
    /* DLP(i=90) [PALHH]: sq3, BASE(93)>>1, GPOW_i90_e3, BASE(93)>>1  [HIGH of PALH] */
    {
    fp_t h0_PALHH, h1_PALHH, u2_PALHH;
    fp_null(h0_PALHH); fp_new(h0_PALHH);
    fp_null(h1_PALHH); fp_new(h1_PALHH);
    fp_null(u2_PALHH); fp_new(u2_PALHH);
    fp_copy(h0_PALHH, h0_PALH);
    fp_sqr(h0_PALHH, h0_PALHH); fp_sqr(h0_PALHH, h0_PALHH); fp_sqr(h0_PALHH, h0_PALHH);
    countS += 3;
    //_tmp = 0;
    fp_prime_back(tmp_bn, h0_PALHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_PALHH) == RLC_EQ) _tmp = _ii;
    (c_PALH) = (uint64_t)_tmp >> 1;
    GPOW_i90_e3(h1_PALHH, (one_k << 3) - (c_PALH), gw);
    SELECT(h1_PALHH, gpp[93], (c_PALH) == 0, h1_PALHH);
    countM++;
    fp_mul(u2_PALHH, h0_PALH, h1_PALHH);
    //_tmp = 0;
    fp_prime_back(tmp_bn, u2_PALHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_PALHH) == RLC_EQ) _tmp = _ii;
    { uint64_t _d_PALHH = (uint64_t)_tmp >> 1;
      (c_PALH) = (c_PALH) + ((_d_PALHH - 1) % 8) * 8; }
    fp_free(h0_PALHH); fp_free(h1_PALHH); fp_free(u2_PALHH);
    }

    GPOW_i84_e6(h1_PALH, (one_k << 6) - c_PALH, gw);
    SELECT(h1_PALH, gpp[84 + 6], c_PALH == 0, h1_PALH);
    countM++;
    fp_mul(u2_PALH, h0_PAL, h1_PALH);   /* u_PALH = h0_PAL */

    uint64_t d_PAL;
    /* DLP(i=90) [PALHL]: sq3, BASE(93)>>1, GPOW_i90_e3, BASE(93)>>1  [LOW  of PALH] */
    {
    fp_t h0_PALHL, h1_PALHL, u2_PALHL;
    fp_null(h0_PALHL); fp_new(h0_PALHL);
    fp_null(h1_PALHL); fp_new(h1_PALHL);
    fp_null(u2_PALHL); fp_new(u2_PALHL);
    fp_copy(h0_PALHL, u2_PALH);
    fp_sqr(h0_PALHL, h0_PALHL); fp_sqr(h0_PALHL, h0_PALHL); fp_sqr(h0_PALHL, h0_PALHL);
    countS += 3;
    //_tmp = 0;
    fp_prime_back(tmp_bn, h0_PALHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//       if (fp_cmp(rll[_ii], h0_PALHL) == RLC_EQ) _tmp = _ii;
    (d_PAL) = (uint64_t)_tmp >> 1;
    GPOW_i90_e3(h1_PALHL, (one_k << 3) - (d_PAL), gw);
    SELECT(h1_PALHL, gpp[93], (d_PAL) == 0, h1_PALHL);
    countM++;
    fp_mul(u2_PALHL, u2_PALH, h1_PALHL);
    //_tmp = 0;
    fp_prime_back(tmp_bn, u2_PALHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_PALHL) == RLC_EQ) _tmp = _ii;
    { uint64_t _d_PALHL = (uint64_t)_tmp >> 1;
      (d_PAL) = (d_PAL) + ((_d_PALHL - 1) % 8) * 8; }
    fp_free(h0_PALHL); fp_free(h1_PALHL); fp_free(u2_PALHL);
    }

    uint64_t c_PAL = c_PALH + ((d_PAL - 1) % 64) * 64;   /* combine PALH, b=a=6 */

    /* correct PAL at i=72, no hlp (hlp_PAL=None) */
    GPOW_i72_e12(h1_PAL, (one_k << 12) - c_PAL, gw);
    SELECT(h1_PAL, gpp[72 + 12], c_PAL == 0, h1_PAL);
    countM++;
    fp_mul(u2_PAL, u2_PA, h1_PAL);   /* u_PAL = u2_PA */

    /* PALL: sq 6 */
    fp_copy(h0_PALL, u2_PAL);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_PALL, h0_PALL);
    countS += 6;

    uint64_t c_PALL;
    /* DLP(i=90) [PALLH]: sq3, BASE(93)>>1, GPOW_i90_e3, BASE(93)>>1  [HIGH of PALL] */
    {
    fp_t h0_PALLH, h1_PALLH, u2_PALLH;
    fp_null(h0_PALLH); fp_new(h0_PALLH);
    fp_null(h1_PALLH); fp_new(h1_PALLH);
    fp_null(u2_PALLH); fp_new(u2_PALLH);
    fp_copy(h0_PALLH, h0_PALL);
    fp_sqr(h0_PALLH, h0_PALLH); fp_sqr(h0_PALLH, h0_PALLH); fp_sqr(h0_PALLH, h0_PALLH);
    countS += 3;
    //_tmp = 0;
    fp_prime_back(tmp_bn, h0_PALLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\naaaaa",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_PALLH) == RLC_EQ) _tmp = _ii;
    (c_PALL) = (uint64_t)_tmp >> 1;
    GPOW_i90_e3(h1_PALLH, (one_k << 3) - (c_PALL), gw);
    SELECT(h1_PALLH, gpp[93], (c_PALL) == 0, h1_PALLH);
    countM++;
    fp_mul(u2_PALLH, h0_PALL, h1_PALLH);
    //_tmp = 0;
    fp_prime_back(tmp_bn, u2_PALLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\nbbbb",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_PALLH) == RLC_EQ) _tmp = _ii;
    { uint64_t _d_PALLH = (uint64_t)_tmp >> 1;
      (c_PALL) = (c_PALL) + ((_d_PALLH - 1) % 8) * 8; }
    fp_free(h0_PALLH); fp_free(h1_PALLH); fp_free(u2_PALLH);
    }

    GPOW_i84_e6(h1_PALL, (one_k << 6) - c_PALL, gw);
    SELECT(h1_PALL, gpp[84 + 6], c_PALL == 0, h1_PALL);
    countM++;
    fp_mul(u2_PALL, u2_PAL, h1_PALL);   /* u_PALL = u2_PAL */

    uint64_t d_PA2;
    /* DLP(i=90) [PALLL]: sq3, BASE(93)>>1, GPOW_i90_e3, BASE(93)>>1  [LOW  of PALL] */
    {
    fp_t h0_PALLL, h1_PALLL, u2_PALLL;
    fp_null(h0_PALLL); fp_new(h0_PALLL);
    fp_null(h1_PALLL); fp_new(h1_PALLL);
    fp_null(u2_PALLL); fp_new(u2_PALLL);
    fp_copy(h0_PALLL, u2_PALL);
    fp_sqr(h0_PALLL, h0_PALLL); fp_sqr(h0_PALLL, h0_PALLL); fp_sqr(h0_PALLL, h0_PALLL);
    countS += 3;
    //_tmp = 0;
    fp_prime_back(tmp_bn, h0_PALLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
 //   printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_PALLL) == RLC_EQ) _tmp = _ii;
    (d_PA2) = (uint64_t)_tmp >> 1;
    GPOW_i90_e3(h1_PALLL, (one_k << 3) - (d_PA2), gw);
    SELECT(h1_PALLL, gpp[93], (d_PA2) == 0, h1_PALLL);
    countM++;
    fp_mul(u2_PALLL, u2_PALL, h1_PALLL);
    //_tmp = 0;
    fp_prime_back(tmp_bn, u2_PALLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_PALLL) == RLC_EQ) _tmp = _ii;
    { uint64_t _d_PALLL = (uint64_t)_tmp >> 1;
      (d_PA2) = (d_PA2) + ((_d_PALLL - 1) % 8) * 8; }
    fp_free(h0_PALLL); fp_free(h1_PALLL); fp_free(u2_PALLL);
    }

    uint64_t d_PA3 = c_PALL + ((d_PA2 - 1) % 64) * 64;            /* combine PALL, b=a=6 */
    uint64_t d_PA4 = c_PAL  + ((d_PA3 - 1) % ((one_k << 12))) * (one_k << 12); /* combine PAL */
    uint64_t c_A   = c_PA   + ((d_PA4 - 1) % ((one_k << 24))) * (one_k << 24); /* combine PA  */

    /* =========================================================================
     * f_A top-level correction
     * f_A   = GPOW_i0_e47 (11 muls)
     * hlp1_A = GPOW_i24_e48 (11 muls)
     * ========================================================================= */
    GPOW_i0_e47(f_A,    ((one_k << 48) - c_A + 1) >> 1, gw);
    SELECT(f_A,    gpp[47],      c_A == 0, f_A);
    GPOW_i24_e48(hlp1_A, (one_k << 48) - c_A, gw);
    SELECT(hlp1_A, gpp[24 + 48], c_A == 0, hlp1_A);
    fp_mul(hlp_A, hlp_A, hlp1_A); countM++;
    countM++;
    countS++;

    /* =========================================================================
     * EXT2 depth 0: i_X0=48, helper=hlp_A -> else branch
     * c_X0 = DLPpow2(h0_X0, i_X0P=72): same structure as PAH
     * f_X0 = GPOW_i47_e24 (6 muls), no hlp
     * ========================================================================= */
    fp_sqr(f_A_sq, f_A);
    fp_mul(u_X0, u_A, f_A_sq);
    fp_copy(h0_X0, hlp_A);   /* else branch: h0_X0=hlp_A, hlp_X0=None */

    /* X0P (i=72): sq 12, hlp_X0P at j=6 */
    fp_copy(h0_X0P, h0_X0);
    for (int _j = 0; _j < 12; _j++) {
        if (_j == 6) fp_copy(hlp_X0P, h0_X0P);
        fp_sqr(h0_X0P, h0_X0P);
    }
    countS += 12;

    /* X0PH (i=84): sq 6, do_hlp=False */
    fp_copy(h0_X0PH, h0_X0P);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_X0PH, h0_X0PH);
    countS += 6;

    uint64_t c_X0PH;
    /* DLP(i=90) [X0PHH]: sq3, BASE(93)>>1, GPOW_i90_e3, BASE(93)>>1  [HIGH of X0PH] */
    {
    fp_t h0_X0PHH, h1_X0PHH, u2_X0PHH;
    fp_null(h0_X0PHH); fp_new(h0_X0PHH);
    fp_null(h1_X0PHH); fp_new(h1_X0PHH);
    fp_null(u2_X0PHH); fp_new(u2_X0PHH);
    fp_copy(h0_X0PHH, h0_X0PH);
    fp_sqr(h0_X0PHH, h0_X0PHH); fp_sqr(h0_X0PHH, h0_X0PHH); fp_sqr(h0_X0PHH, h0_X0PHH);
    countS += 3;
    //_tmp = 0;
    fp_prime_back(tmp_bn, h0_X0PHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_X0PHH) == RLC_EQ) _tmp = _ii;
    (c_X0PH) = (uint64_t)_tmp >> 1;
    GPOW_i90_e3(h1_X0PHH, (one_k << 3) - (c_X0PH), gw);
    SELECT(h1_X0PHH, gpp[93], (c_X0PH) == 0, h1_X0PHH);
    countM++;
    fp_mul(u2_X0PHH, h0_X0PH, h1_X0PHH);
    //_tmp = 0;
    fp_prime_back(tmp_bn, u2_X0PHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_X0PHH) == RLC_EQ) _tmp = _ii;
    { uint64_t _d_X0PHH = (uint64_t)_tmp >> 1;
      (c_X0PH) = (c_X0PH) + ((_d_X0PHH - 1) % 8) * 8; }
    fp_free(h0_X0PHH); fp_free(h1_X0PHH); fp_free(u2_X0PHH);
    }

    GPOW_i84_e6(h1_X0PH, (one_k << 6) - c_X0PH, gw);
    SELECT(h1_X0PH, gpp[84 + 6], c_X0PH == 0, h1_X0PH);
    countM++;
    fp_mul(u2_X0PH, h0_X0P, h1_X0PH);   /* u_X0PH = h0_X0P */

    uint64_t d_X0PH;
    /* DLP(i=90) [X0PHL]: sq3, BASE(93)>>1, GPOW_i90_e3, BASE(93)>>1  [LOW  of X0PH] */
    {
    fp_t h0_X0PHL, h1_X0PHL, u2_X0PHL;
    fp_null(h0_X0PHL); fp_new(h0_X0PHL);
    fp_null(h1_X0PHL); fp_new(h1_X0PHL);
    fp_null(u2_X0PHL); fp_new(u2_X0PHL);
    fp_copy(h0_X0PHL, u2_X0PH);
    fp_sqr(h0_X0PHL, h0_X0PHL); fp_sqr(h0_X0PHL, h0_X0PHL); fp_sqr(h0_X0PHL, h0_X0PHL);
    countS += 3;
    //_tmp = 0;
    fp_prime_back(tmp_bn, h0_X0PHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
 //   printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_X0PHL) == RLC_EQ) _tmp = _ii;
    (d_X0PH) = (uint64_t)_tmp >> 1;
    GPOW_i90_e3(h1_X0PHL, (one_k << 3) - (d_X0PH), gw);
    SELECT(h1_X0PHL, gpp[93], (d_X0PH) == 0, h1_X0PHL);
    countM++;
    fp_mul(u2_X0PHL, u2_X0PH, h1_X0PHL);
    //_tmp = 0;
    fp_prime_back(tmp_bn, u2_X0PHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_X0PHL) == RLC_EQ) _tmp = _ii;
    { uint64_t _d_X0PHL = (uint64_t)_tmp >> 1;
      (d_X0PH) = (d_X0PH) + ((_d_X0PHL - 1) % 8) * 8; }
    fp_free(h0_X0PHL); fp_free(h1_X0PHL); fp_free(u2_X0PHL);
    }

    uint64_t c_X0P = c_X0PH + ((d_X0PH - 1) % 64) * 64;   /* combine X0PH */

    /* correct X0P at i=72 (hlp_X0P present) */
    GPOW_i72_e12(h1_X0P,   (one_k << 12) - c_X0P, gw);
    SELECT(h1_X0P,   gpp[72 + 12], c_X0P == 0, h1_X0P);
    GPOW_i78_e12(hlp1_X0P, (one_k << 12) - c_X0P, gw);
    SELECT(hlp1_X0P, gpp[78 + 12], c_X0P == 0, hlp1_X0P);
    fp_mul(hlp_X0P, hlp_X0P, hlp1_X0P); countM++;
    countM++;
    fp_mul(u2_X0P, h0_X0, h1_X0P);   /* u_X0P = h0_X0 */

    /* X0PL (i=84): helper=hlp_X0P -> else branch */
    fp_copy(h0_X0PL, hlp_X0P);   /* else branch */

    uint64_t c_X0PL;
    /* DLP(i=90) [X0PLH]: sq3, BASE(93)>>1, GPOW_i90_e3, BASE(93)>>1  [HIGH of X0PL] */
    {
    fp_t h0_X0PLH, h1_X0PLH, u2_X0PLH;
    fp_null(h0_X0PLH); fp_new(h0_X0PLH);
    fp_null(h1_X0PLH); fp_new(h1_X0PLH);
    fp_null(u2_X0PLH); fp_new(u2_X0PLH);
    fp_copy(h0_X0PLH, h0_X0PL);
    fp_sqr(h0_X0PLH, h0_X0PLH); fp_sqr(h0_X0PLH, h0_X0PLH); fp_sqr(h0_X0PLH, h0_X0PLH);
    countS += 3;
    //_tmp = 0;
    fp_prime_back(tmp_bn, h0_X0PLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_X0PLH) == RLC_EQ) _tmp = _ii;
    (c_X0PL) = (uint64_t)_tmp >> 1;
    GPOW_i90_e3(h1_X0PLH, (one_k << 3) - (c_X0PL), gw);
    SELECT(h1_X0PLH, gpp[93], (c_X0PL) == 0, h1_X0PLH);
    countM++;
    fp_mul(u2_X0PLH, h0_X0PL, h1_X0PLH);
    //_tmp = 0;
    fp_prime_back(tmp_bn, u2_X0PLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_X0PLH) == RLC_EQ) _tmp = _ii;
    { uint64_t _d_X0PLH = (uint64_t)_tmp >> 1;
      (c_X0PL) = (c_X0PL) + ((_d_X0PLH - 1) % 8) * 8; }
    fp_free(h0_X0PLH); fp_free(h1_X0PLH); fp_free(u2_X0PLH);
    }

    GPOW_i84_e6(h1_X0PL, (one_k << 6) - c_X0PL, gw);
    SELECT(h1_X0PL, gpp[84 + 6], c_X0PL == 0, h1_X0PL);
    countM++;
    fp_mul(u2_X0PL, u2_X0P, h1_X0PL);   /* u_X0PL = u2_X0P */

    uint64_t d_X0PL;
    /* DLP(i=90) [X0PLL]: sq3, BASE(93)>>1, GPOW_i90_e3, BASE(93)>>1  [LOW  of X0PL] */
    {
    fp_t h0_X0PLL, h1_X0PLL, u2_X0PLL;
    fp_null(h0_X0PLL); fp_new(h0_X0PLL);
    fp_null(h1_X0PLL); fp_new(h1_X0PLL);
    fp_null(u2_X0PLL); fp_new(u2_X0PLL);
    fp_copy(h0_X0PLL, u2_X0PL);
    fp_sqr(h0_X0PLL, h0_X0PLL); fp_sqr(h0_X0PLL, h0_X0PLL); fp_sqr(h0_X0PLL, h0_X0PLL);
    countS += 3;
    //_tmp = 0;
    fp_prime_back(tmp_bn, h0_X0PLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_X0PLL) == RLC_EQ) _tmp = _ii;
    (d_X0PL) = (uint64_t)_tmp >> 1;
    GPOW_i90_e3(h1_X0PLL, (one_k << 3) - (d_X0PL), gw);
    SELECT(h1_X0PLL, gpp[93], (d_X0PL) == 0, h1_X0PLL);
    countM++;
    fp_mul(u2_X0PLL, u2_X0PL, h1_X0PLL);
    //_tmp = 0;
    fp_prime_back(tmp_bn, u2_X0PLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_X0PLL) == RLC_EQ) _tmp = _ii;
    { uint64_t _d_X0PLL = (uint64_t)_tmp >> 1;
      (d_X0PL) = (d_X0PL) + ((_d_X0PLL - 1) % 8) * 8; }
    fp_free(h0_X0PLL); fp_free(h1_X0PLL); fp_free(u2_X0PLL);
    }

    uint64_t d_X0P = c_X0PL + ((d_X0PL  - 1) % 64) * 64;
    uint64_t c_X0  = c_X0P  + ((d_X0P   - 1) % ((one_k << 12))) * (one_k << 12);

    /* ext2 depth 0 correction: GPOW_i47_e24, no hlp */
    GPOW_i47_e24(f_X0, (one_k << 24) - c_X0, gw);
    SELECT(f_X0, gpp[47 + 24], c_X0 == 0, f_X0);
    countM++;
    countS++;

    /* =========================================================================
     * EXT2 depth 1: i_X1=72, helper=None -> do_hlp=True, sq12, hlp_X1 at j=6
     * c_X1 = DLPpow2(h0_X1, i_X1P=84): sq6 + DLP90x2
     * f_X1 = GPOW_i71_e12 + hlp1_X1(GPOW_i78_e12)
     * ========================================================================= */
    fp_sqr(f_X0_sq, f_X0);
    fp_mul(u_X1, u_X0, f_X0_sq);
    fp_copy(h0_X1, u_X1);
    for (int _j = 0; _j < 12; _j++) {
        if (_j == 6) fp_copy(hlp_X1, h0_X1);
        fp_sqr(h0_X1, h0_X1);
    }
    countS += 12;

    /* X1P (i=84): sq 6, do_hlp=False */
    fp_copy(h0_X1P, h0_X1);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_X1P, h0_X1P);
    countS += 6;

    uint64_t c_X1P;
    /* DLP(i=90) [X1PH]: sq3, BASE(93)>>1, GPOW_i90_e3, BASE(93)>>1  [HIGH of X1P] */
    {
    fp_t h0_X1PH, h1_X1PH, u2_X1PH;
    fp_null(h0_X1PH); fp_new(h0_X1PH);
    fp_null(h1_X1PH); fp_new(h1_X1PH);
    fp_null(u2_X1PH); fp_new(u2_X1PH);
    fp_copy(h0_X1PH, h0_X1P);
    fp_sqr(h0_X1PH, h0_X1PH); fp_sqr(h0_X1PH, h0_X1PH); fp_sqr(h0_X1PH, h0_X1PH);
    countS += 3;
//    //_tmp = 0;
    fp_prime_back(tmp_bn, h0_X1PH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_X1PH) == RLC_EQ) _tmp = _ii;
    (c_X1P) = (uint64_t)_tmp >> 1;
    GPOW_i90_e3(h1_X1PH, (one_k << 3) - (c_X1P), gw);
    SELECT(h1_X1PH, gpp[93], (c_X1P) == 0, h1_X1PH);
    countM++;
    fp_mul(u2_X1PH, h0_X1P, h1_X1PH);
//    //_tmp = 0;
    fp_prime_back(tmp_bn, u2_X1PH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_X1PH) == RLC_EQ) _tmp = _ii;
    { uint64_t _d_X1PH = (uint64_t)_tmp >> 1;
      (c_X1P) = (c_X1P) + ((_d_X1PH - 1) % 8) * 8; }
    fp_free(h0_X1PH); fp_free(h1_X1PH); fp_free(u2_X1PH);
    }

    GPOW_i84_e6(h1_X1P, (one_k << 6) - c_X1P, gw);
    SELECT(h1_X1P, gpp[84 + 6], c_X1P == 0, h1_X1P);
    countM++;
    fp_mul(u2_X1P, h0_X1, h1_X1P);   /* u_X1P = h0_X1 */

    uint64_t d_X1P;
    /* DLP(i=90) [X1PL]: sq3, BASE(93)>>1, GPOW_i90_e3, BASE(93)>>1  [LOW  of X1P] */
    {
    fp_t h0_X1PL, h1_X1PL, u2_X1PL;
    fp_null(h0_X1PL); fp_new(h0_X1PL);
    fp_null(h1_X1PL); fp_new(h1_X1PL);
    fp_null(u2_X1PL); fp_new(u2_X1PL);
    fp_copy(h0_X1PL, u2_X1P);
    fp_sqr(h0_X1PL, h0_X1PL); fp_sqr(h0_X1PL, h0_X1PL); fp_sqr(h0_X1PL, h0_X1PL);
    countS += 3;
//    //_tmp = 0;
    fp_prime_back(tmp_bn, h0_X1PL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_X1PL) == RLC_EQ) _tmp = _ii;
    (d_X1P) = (uint64_t)_tmp >> 1;
    GPOW_i90_e3(h1_X1PL, (one_k << 3) - (d_X1P), gw);
    SELECT(h1_X1PL, gpp[93], (d_X1P) == 0, h1_X1PL);
    countM++;
    fp_mul(u2_X1PL, u2_X1P, h1_X1PL);
//    //_tmp = 0;
    fp_prime_back(tmp_bn, u2_X1PL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_X1PL) == RLC_EQ) _tmp = _ii;
    { uint64_t _d_X1PL = (uint64_t)_tmp >> 1;
      (d_X1P) = (d_X1P) + ((_d_X1PL - 1) % 8) * 8; }
    fp_free(h0_X1PL); fp_free(h1_X1PL); fp_free(u2_X1PL);
    }

    uint64_t c_X1 = c_X1P + ((d_X1P - 1) % 64) * 64;

    /* ext2 depth 1 correction + hlp_X1 */
    GPOW_i71_e12(f_X1,    (one_k << 12) - c_X1, gw);
    SELECT(f_X1,    gpp[71 + 12], c_X1 == 0, f_X1);
    GPOW_i78_e12(hlp1_X1, (one_k << 12) - c_X1, gw);
    SELECT(hlp1_X1, gpp[78 + 12], c_X1 == 0, hlp1_X1);
    fp_mul(hlp_X1, hlp_X1, hlp1_X1); countM++;
    countM++;
    countS++;

    /* =========================================================================
     * EXT2 depth 2: i_X2=84, helper=hlp_X1 -> else branch
     * c_X2 = DLPpow2(h0_X2, i_X2P=90) = DLP90
     * f_X2 = GPOW_i83_e6 (2 muls), no hlp
     * ========================================================================= */
    fp_sqr(f_X1_sq, f_X1);
    fp_mul(u_X2, u_X1, f_X1_sq);
    fp_copy(h0_X2, hlp_X1);   /* else branch */

    uint64_t c_X2;
    /* DLP(i=90) [X2P]: sq3, BASE(93)>>1, GPOW_i90_e3, BASE(93)>>1  [X2 DLP90] */
    {
    fp_t h0_X2P, h1_X2P, u2_X2P;
    fp_null(h0_X2P); fp_new(h0_X2P);
    fp_null(h1_X2P); fp_new(h1_X2P);
    fp_null(u2_X2P); fp_new(u2_X2P);
    fp_copy(h0_X2P, h0_X2);
    fp_sqr(h0_X2P, h0_X2P); fp_sqr(h0_X2P, h0_X2P); fp_sqr(h0_X2P, h0_X2P);
    countS += 3;
//    //_tmp = 0;
    fp_prime_back(tmp_bn, h0_X2P);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n22222",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_X2P) == RLC_EQ) _tmp = _ii;
    (c_X2) = (uint64_t)_tmp >> 1;
    GPOW_i90_e3(h1_X2P, (one_k << 3) - (c_X2), gw);
    SELECT(h1_X2P, gpp[93], (c_X2) == 0, h1_X2P);
    countM++;
    fp_mul(u2_X2P, h0_X2, h1_X2P);
//    fp_print(u2_X2P);
//    //_tmp = 0;
    fp_prime_back(tmp_bn, u2_X2P);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
    
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_X2P) == RLC_EQ) _tmp = _ii;
    { uint64_t _d_X2P = (uint64_t)_tmp >> 1;
      (c_X2) = (c_X2) + ((_d_X2P - 1) % 8) * 8; }
    fp_free(h0_X2P); fp_free(h1_X2P); fp_free(u2_X2P);
    }

    GPOW_i83_e6(f_X2, (one_k << 6) - c_X2, gw);
    SELECT(f_X2, gpp[83 + 6], c_X2 == 0, f_X2);
    countM++;
    countS++;

    /* =========================================================================
     * EXT2 depth 3: i_X3=90, helper=None, do_hlp=False (b=3<=w=4), sq 3
     * c_X3 = DLPpow2(h0_X3, i_X3P=93): lb=3<=w -> BASE >> 1
     * f_X3 = GPOW_i89_e3 (0 muls), no hlp
     * ========================================================================= */
    fp_sqr(f_X2_sq, f_X2);
    fp_mul(u_X3, u_X2, f_X2_sq);
    fp_copy(h0_X3, u_X3);
    fp_sqr(h0_X3, h0_X3); fp_sqr(h0_X3, h0_X3); fp_sqr(h0_X3, h0_X3);
    countS += 3;

    /* BASE at i=93, lb=3 -> >> (w-lb)=(4-3)=1 */
//    //_tmp = 0;
    fp_prime_back(tmp_bn, h0_X3);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_X3) == RLC_EQ) _tmp = _ii;
    uint64_t c_X3 = (uint64_t)_tmp >> 1;

    GPOW_i89_e3(f_X3, (one_k << 3) - c_X3, gw);
    SELECT(f_X3, gpp[89 + 3], c_X3 == 0, f_X3);
    countM++;
    countS++;

    /* =========================================================================
     * EXT2 base: i_X4=93, lb=3<=w -> lookup in fll
     * >> (w-lb)=(4-3)=1 ; fll[c1<<1]
     * ========================================================================= */
    fp_sqr(f_X3_sq, f_X3);
    fp_mul(u_X4, u_X3, f_X3_sq);
//    //_tmp = 0;
    fp_prime_back(tmp_bn, u_X4);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7f;
    _tmp=rll[d];
//    printf("\n%d\n",_tmp);
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u_X4) == RLC_EQ) _tmp = _ii;
    uint64_t c1_X4 = (uint64_t)_tmp >> 1;
    fp_copy(z_X4, fll[c1_X4 << 1]);
//    uint64_t e_X4 = c1_X4;

    /* =========================================================================
     * Unwind EXT2 chain
     * ========================================================================= */
    fp_mul(t_X3, f_X3, z_X4); //countM++;
//    uint64_t e_X3 = c_X3 + ((e_X4 - 1) % 8) * 8;           /* b=a=3 */

    fp_mul(t_X2, f_X2, t_X3); //countM++;
//    uint64_t e_X2 = c_X2 + ((e_X3 - 1) % 64) * 64;          /* b=a=6 */

    fp_mul(t_X1, f_X1, t_X2); //countM++;
//    uint64_t e_X1 = c_X1 + ((e_X2 - 1) % ((one_k << 6))) * (one_k << 6);  /* b=a=6 */

    fp_mul(t_X0, f_X0, t_X1); //countM++;
//    uint64_t e_X0 = c_X0 + ((e_X1 - 1) % ((one_k << 12))) * (one_k << 12); /* b=a=12 */

    fp_mul(t_A, f_A, t_X0); //countM++;
//    (void)(c_A + ((e_X0 - 1) % ((one_k << 24))) * (one_k << 24));   /* e_final unused */

    fp_copy(out_t, t_A);

    /* ---- free all ---- */
    fp_free(h0_A);    fp_free(hlp_A);   fp_free(f_A);    fp_free(f_A_sq);
    fp_free(h0_PA);   fp_free(hlp_PA);  fp_free(h1_PA);
    fp_free(hlp1_PA); fp_free(u2_PA);
    fp_free(h0_PAH);  fp_free(hlp_PAH); fp_free(h1_PAH);
    fp_free(hlp1_PAH);fp_free(u2_PAH);
    fp_free(h0_PAHH); fp_free(h1_PAHH); fp_free(u2_PAHH);
    fp_free(h0_PAHL); fp_free(h1_PAHL); fp_free(u2_PAHL);
    fp_free(h0_PAL);  fp_free(h1_PAL);  fp_free(u2_PAL);
    fp_free(h0_PALH); fp_free(h1_PALH); fp_free(u2_PALH);
    fp_free(h0_PALL); fp_free(h1_PALL); fp_free(u2_PALL);
    fp_free(hlp1_A);
    fp_free(u_X0);    fp_free(h0_X0);
    fp_free(h0_X0P);  fp_free(hlp_X0P); fp_free(h1_X0P);
    fp_free(hlp1_X0P);fp_free(u2_X0P);
    fp_free(h0_X0PH); fp_free(h1_X0PH); fp_free(u2_X0PH);
    fp_free(h0_X0PL); fp_free(h1_X0PL); fp_free(u2_X0PL);
    fp_free(f_X0);    fp_free(f_X0_sq);
    fp_free(u_X1);    fp_free(hlp_X1);  fp_free(h0_X1);
    fp_free(h0_X1P);  fp_free(h1_X1P);  fp_free(u2_X1P);
    fp_free(f_X1);    fp_free(hlp1_X1); fp_free(f_X1_sq);
    fp_free(u_X2);    fp_free(h0_X2);
    fp_free(h0_X2P);  fp_free(h1_X2P);  fp_free(u2_X2P);
    fp_free(f_X2);    fp_free(f_X2_sq);
    fp_free(u_X3);    fp_free(h0_X3);
    fp_free(f_X3);    fp_free(f_X3_sq);
    fp_free(u_X4);    fp_free(z_X4);
    fp_free(t_X3);    fp_free(t_X2);    fp_free(t_X1);
    fp_free(t_X0);    fp_free(t_A);
}

/* =========================================================================
 * sqrt_ext
 * ========================================================================= */
void sqrt_ext(fp_t x, fp_t y, bn_t e_exp,
              fp_t gw[nw][we], int rll[128], fp_t fll[we], fp_t gpp[n])
{
    fp_t u, v, w_, t;
    fp_null(u); fp_null(v); fp_null(w_); fp_null(t);
    RLC_TRY {
        fp_new(u); fp_new(v); fp_new(w_); fp_new(t);
        fp_exp(v,  x, e_exp);
        fp_mul(w_, x, v);  //countM++;
        fp_mul(u,  w_, v); //countM++;
        DLPpow2ext(u, t, gw, rll, fll, gpp);
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
    fp_t gw[nw][we], fll[we], gpp[n], g, z, b, h, hh, y;
    bn_t tmp, m, e;
    int rll[128]={0};
    
    rll[ 0x1 ]= 0 ;
    rll[ 0x2f ]= 1 ;
    rll[ 0x3d ]= 2 ;
    rll[ 0x30 ]= 3 ;
    rll[ 0x19 ]= 4 ;
    rll[ 0x38 ]= 5 ;
    rll[ 0x9 ]= 6 ;
    rll[ 0x2c ]= 7 ;
    rll[ 0x0 ]= 8 ;
    rll[ 0x52 ]= 9 ;
    rll[ 0x44 ]= 10 ;
    rll[ 0x51 ]= 11 ;
    rll[ 0x68 ]= 12 ;
    rll[ 0x49 ]= 13 ;
    rll[ 0x78 ]= 14 ;
    rll[ 0x55 ]= 15 ;
    

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

    bn_read_str(tmp, "100000000000000000000000", 24, 16);
    fp_exp(h, g, tmp);

    fp_srt(hh, h);
    fp_inv(hh, hh);

    precomputation(g, h, hh, gw, fll, gpp);

    MEASURE(sqrt_ext(b, y, e, gw, rll, fll, gpp);)

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
