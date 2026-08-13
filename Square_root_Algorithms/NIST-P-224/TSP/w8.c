#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include <immintrin.h>
#include <relic/relic_core.h>
#include "relic/relic_md.h"

#define w   8
#define we  256
#define n   96
#define nw  12

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

/* =========================================================================
 * SELECT(a0, a1, ctl) — same as Sage: returns a1 if ctl else a0
 * In C we write result into 'out'.
 * ========================================================================= */
static void SELECT(fp_t a0, fp_t a1, bool ctl, fp_t out)
{
    if (ctl) fp_copy(out, a1);
    else     fp_copy(out, a0);
}

/* =========================================================================
 * GPOW_i*_e* — same as Sage functions.
 * Sage uses local 't' and returns it; here 't' is the output parameter.
 * 'e' passed by value so in-place shifts don't affect caller.
 * ========================================================================= */

/* Sage: def GPOW_i0_e47(e): */
static void GPOW_i0_e47(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 47) - 1;
    fp_copy(t, gw[0][e & 0xFF]); e >>= 8;
    fp_mul(t, t, gw[1][e & 0xFF]); //countM++; 
e >>= 8;
    fp_mul(t, t, gw[2][e & 0xFF]); //countM++; 
e >>= 8;
    fp_mul(t, t, gw[3][e & 0xFF]); //countM++; 
e >>= 8;
    fp_mul(t, t, gw[4][e & 0xFF]); //countM++; 
e >>= 8;
    fp_mul(t, t, gw[5][e & 0xFF]); //countM++;
}

/* Sage: def GPOW_i24_e48(e): */
static void GPOW_i24_e48(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 48) - 1;
    fp_copy(t, gw[3][e & 0xFF]); e >>= 8;
    fp_mul(t, t, gw[4][e & 0xFF]); //countM++; 
e >>= 8;
    fp_mul(t, t, gw[5][e & 0xFF]); //countM++; 
e >>= 8;
    fp_mul(t, t, gw[6][e & 0xFF]); //countM++; 
e >>= 8;
    fp_mul(t, t, gw[7][e & 0xFF]); //countM++; 
e >>= 8;
    fp_mul(t, t, gw[8][e & 0xFF]); //countM++;
}

/* Sage: def GPOW_i47_e24(e): */
static void GPOW_i47_e24(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 24) - 1;
    e <<= 7;
    fp_copy(t, gw[5][e & 0xFF]); e >>= 8;
    fp_mul(t, t, gw[6][e & 0xFF]); //countM++; 
e >>= 8;
    fp_mul(t, t, gw[7][e & 0xFF]); //countM++; 
e >>= 8;
    fp_mul(t, t, gw[8][e & 0xFF]); //countM++;
}

/* Sage: def GPOW_i48_e24(e): */
static void GPOW_i48_e24(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 24) - 1;
    fp_copy(t, gw[6][e & 0xFF]); e >>= 8;
    fp_mul(t, t, gw[7][e & 0xFF]); //countM++; 
e >>= 8;
    fp_mul(t, t, gw[8][e & 0xFF]); //countM++;
}

/* Sage: def GPOW_i60_e24(e): */
static void GPOW_i60_e24(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 24) - 1;
    e <<= 4;
    fp_copy(t, gw[7][e & 0xFF]);  e >>= 8;
    fp_mul(t, t, gw[8][e & 0xFF]);  //countM++; 
e >>= 8;
    fp_mul(t, t, gw[9][e & 0xFF]);  //countM++; 
e >>= 8;
    fp_mul(t, t, gw[10][e & 0xFF]); //countM++;
}

/* Sage: def GPOW_i71_e12(e): */
static void GPOW_i71_e12(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 12) - 1;
    e <<= 7;
    fp_copy(t, gw[8][e & 0xFF]);  e >>= 8;
    fp_mul(t, t, gw[9][e & 0xFF]);  //countM++; 
e >>= 8;
    fp_mul(t, t, gw[10][e & 0xFF]); //countM++;
}

/* Sage: def GPOW_i72_e12(e): */
static void GPOW_i72_e12(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 12) - 1;
    fp_copy(t, gw[9][e & 0xFF]);  e >>= 8;
    fp_mul(t, t, gw[10][e & 0xFF]); //countM++;
}

/* Sage: def GPOW_i78_e12(e): */
static void GPOW_i78_e12(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 12) - 1;
    e <<= 6;
    fp_copy(t, gw[9][e & 0xFF]);   e >>= 8;
    fp_mul(t, t, gw[10][e & 0xFF]); //countM++; 
e >>= 8;
    fp_mul(t, t, gw[11][e & 0xFF]); //countM++;
}

/* Sage: def GPOW_i83_e6(e): */
static void GPOW_i83_e6(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 6) - 1;
    e <<= 3;
    fp_copy(t, gw[10][e & 0xFF]); e >>= 8;
    fp_mul(t, t, gw[11][e & 0xFF]); //countM++;
}

/* Sage: def GPOW_i84_e6(e): */
static void GPOW_i84_e6(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 6) - 1;
    e <<= 4;
    fp_copy(t, gw[10][e & 0xFF]); e >>= 8;
    fp_mul(t, t, gw[11][e & 0xFF]); //countM++;
}


/* =========================================================================
 * DLPpow2ext — direct line-by-line translation of Sage.
 * Same variable names, same logic, same order.
 * Sage returns (e_final, t_A); here t_A is written into out_t.
 * ========================================================================= */
void DLPpow2ext(fp_t u_A, fp_t out_t,
                fp_t gw[nw][we], int rlll[8192], fp_t rll[we], fp_t fll[we], fp_t gpp[n])
{
    uint64_t one_k = 1;

    /* All fp_t temporaries — declared and allocated here */
    fp_t h0_A, hlp_A, f_A;
    fp_t u_PA, hlp_PA, h0_PA;
    fp_t u_PAH, hlp_PAH, h0_PAH;
    fp_t u_PAHH, h0_PAHH;
    fp_t h1_PAHH, hlp_PAHH, hlp1_PAHH, u2_PAHH;
    fp_t h1_PAH, hlp1_PA, u2_PAH;
    fp_t h0_PAHL, h1_PAHL, u2_PAHL;
    fp_t h1_PA, hlp1_A, u2_PA;
    fp_t h0_PAL, hlp_PAL, u_PALH, h0_PALH, h1_PAL, u2_PAL;
    fp_t u_PALL, h0_PALL, h1_PALL, u2_PALL;
    fp_t hlp1, f_A_sq;
    fp_t u_X0, h0_X0;
    fp_t u_X0P, hlp_X0P, h0_X0P;
    fp_t u_X0PH, h0_X0PH;
    fp_t h1_X0P, hlp1_X0, u2_X0P;
    fp_t h0_X0PL, h1_X0PL, u2_X0PL;
    fp_t f_X0, f_X0_sq;
    fp_t u_X1, hlp_X1, h0_X1;
    fp_t u_X1P, hlp_X1P, h0_X1P;
    fp_t h1_X1P, u2_X1P;
    fp_t h0_X1L, h1_X1L, u2_X1L;
    fp_t f_X1, hlp1_1, f_X1_sq;
    fp_t u_X2, h0_X2;
    fp_t f_X2, f_X2_sq;
    fp_t u_X3;
    fp_t z_X3;
    fp_t t_X2, t_X1, t_X0, t_A;
    fp_t h1_PALH,u2_PALH,h1_X0PH,u2_X0PH,u_X1PL;

    fp_null(h0_A);    fp_new(h0_A);
    fp_null(hlp_A);   fp_new(hlp_A);
    fp_null(f_A);     fp_new(f_A);
    fp_null(u_PA);    fp_new(u_PA);
    fp_null(hlp_PA);  fp_new(hlp_PA);
    fp_null(h0_PA);   fp_new(h0_PA);
    fp_null(u_PAH);   fp_new(u_PAH);
    fp_null(hlp_PAH); fp_new(hlp_PAH);
    fp_null(h0_PAH);  fp_new(h0_PAH);
    fp_null(u_PAHH);  fp_new(u_PAHH);
    fp_null(h0_PAHH); fp_new(h0_PAHH);
    fp_null(h1_PAHH); fp_new(h1_PAHH);
    fp_null(hlp_PAHH);  fp_new(hlp_PAHH);
    fp_null(hlp1_PAHH); fp_new(hlp1_PAHH);
    fp_null(u2_PAHH); fp_new(u2_PAHH);
    fp_null(h1_PAH);  fp_new(h1_PAH);
    fp_null(hlp1_PA); fp_new(hlp1_PA);
    fp_null(u2_PAH);  fp_new(u2_PAH);
    fp_null(h0_PAHL); fp_new(h0_PAHL);
    fp_null(h1_PAHL); fp_new(h1_PAHL);
    fp_null(u2_PAHL); fp_new(u2_PAHL);
    fp_null(h1_PA);   fp_new(h1_PA);
    fp_null(hlp1_A);  fp_new(hlp1_A);
    fp_null(u2_PA);   fp_new(u2_PA);
    fp_null(h0_PAL);  fp_new(h0_PAL);
    fp_null(hlp_PAL); fp_new(hlp_PAL);
    fp_null(u_PALH);  fp_new(u_PALH);
    fp_null(h0_PALH); fp_new(h0_PALH);
    fp_null(h1_PAL);  fp_new(h1_PAL);
    fp_null(u2_PAL);  fp_new(u2_PAL);
    fp_null(u_PALL);  fp_new(u_PALL);
    fp_null(h0_PALL); fp_new(h0_PALL);
    fp_null(h1_PALL); fp_new(h1_PALL);
    fp_null(u2_PALL); fp_new(u2_PALL);
    fp_null(hlp1);    fp_new(hlp1);
    fp_null(f_A_sq);  fp_new(f_A_sq);
    fp_null(u_X0);    fp_new(u_X0);
    fp_null(h0_X0);   fp_new(h0_X0);
    fp_null(u_X0P);   fp_new(u_X0P);
    fp_null(hlp_X0P); fp_new(hlp_X0P);
    fp_null(h0_X0P);  fp_new(h0_X0P);
    fp_null(u_X0PH);  fp_new(u_X0PH);
    fp_null(h0_X0PH); fp_new(h0_X0PH);
    fp_null(h1_X0P);  fp_new(h1_X0P);
    fp_null(hlp1_X0); fp_new(hlp1_X0);
    fp_null(u2_X0P);  fp_new(u2_X0P);
    fp_null(h0_X0PL); fp_new(h0_X0PL);
    fp_null(h1_X0PL); fp_new(h1_X0PL);
    fp_null(u2_X0PL); fp_new(u2_X0PL);
    fp_null(f_X0);    fp_new(f_X0);
    fp_null(f_X0_sq); fp_new(f_X0_sq);
    fp_null(u_X1);    fp_new(u_X1);
    fp_null(hlp_X1);  fp_new(hlp_X1);
    fp_null(h0_X1);   fp_new(h0_X1);
    fp_null(u_X1P);   fp_new(u_X1P);
    fp_null(hlp_X1P); fp_new(hlp_X1P);
    fp_null(h0_X1P);  fp_new(h0_X1P);
    fp_null(h1_X1P);  fp_new(h1_X1P);
    fp_null(u2_X1P);  fp_new(u2_X1P);
    fp_null(h0_X1L);  fp_new(h0_X1L);
    fp_null(h1_X1L);  fp_new(h1_X1L);
    fp_null(u2_X1L);  fp_new(u2_X1L);
    fp_null(f_X1);    fp_new(f_X1);
    fp_null(hlp1_1);  fp_new(hlp1_1);
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

    int _ii, _tmp;

    /* =========================================================================
     * Sage line 205-216:
     * lb_A = 96 - i_A  (= 96)
     * a_A = 48, b_A = 48, nlb1_A = 24
     * hlp_A = None
     * h0_A = u_A
     * for _j in range(48):
     *     if _j == 24: hlp_A = h0_A
     *     h0_A *= h0_A
     * cost["S"] += 48
     * ========================================================================= */
    fp_copy(h0_A, u_A);
    for (int _j = 0; _j < 48; _j++) {
        if (_j == 24) fp_copy(hlp_A, h0_A);
        fp_sqr(h0_A, h0_A);
    }
    //countS += 48;
    /* =========================================================================
     * Sage line 225-232:
     * u_PA = h0_A
     * hlp_PA = None
     * h0_PA = u_PA
     * for _j in range(24):
     *     if _j == 12: hlp_PA = h0_PA
     *     h0_PA *= h0_PA
     * cost["S"] += 24
     * ========================================================================= */
    fp_copy(u_PA, h0_A);
    fp_copy(h0_PA, u_PA);
    for (int _j = 0; _j < 24; _j++) {
        if (_j == 12) fp_copy(hlp_PA, h0_PA);
        fp_sqr(h0_PA, h0_PA);
    }
    //countS += 24;

    /* =========================================================================
     * Sage line 237-244:
     * u_PAH = h0_PA
     * hlp_PAH = None
     * h0_PAH = u_PAH
     * for _j in range(12):
     *     if _j == 6: hlp_PAH = h0_PAH
     *     h0_PAH *= h0_PAH
     * cost["S"] += 12
     * ========================================================================= */
    fp_copy(u_PAH, h0_PA);
    fp_copy(h0_PAH, u_PAH);
    for (int _j = 0; _j < 12; _j++) {
        if (_j == 6) fp_copy(hlp_PAH, h0_PAH);
        fp_sqr(h0_PAH, h0_PAH);
    }
    //countS += 12;

    /* =========================================================================
     * leaf-H of PAH: DLPpow2(h0_PAH, i_PAHH = i_PAH + b_PAH, helper=None)
     *
     * Sage:
     *   u_PAHH  = h0_PAH
     *   i_PAHH  = i_PAH + b_PAH
     *   lb_PAHH = 96 - i_PAHH
     *   if lb_PAHH <= w:
     *       for _ii in range(256): if rll[_ii]==u_PAHH: _tmp=_ii
     *       c_PAH = _tmp >> (w - lb_PAHH)
     *   else:
     *       a_PAHH=lb_PAHH//2; b_PAHH=lb_PAHH-a_PAHH; nlb1_PAHH=b_PAHH-(b_PAHH>>1)
     *       cost_hlp_PAHH=1+GPOW_cost(i_PAHH+nlb1_PAHH,a_PAHH)
     *       do_hlp_PAHH=True
     *       if b_PAHH<=w or cost_hlp_PAHH>=nlb1_PAHH: do_hlp_PAHH=False
     *       hlp_PAHH=None
     *       h0_PAHH=u_PAHH
     *       for _j in range(b_PAHH):
     *           if do_hlp_PAHH and _j==nlb1_PAHH: hlp_PAHH=h0_PAHH
     *           h0_PAHH*=h0_PAHH
     *       cost["S"]+=b_PAHH
     *       [base lookup] c_PAH=_tmp>>(w-(96-(i_PAHH+b_PAHH)))
     *       h1_PAHH=GPOW(i_PAHH,(1<<a_PAHH)-c_PAH,a_PAHH)
     *       h1_PAHH=SELECT(h1_PAHH,gpp[i_PAHH+a_PAHH],c_PAH==0)
     *       if hlp_PAHH is not None:
     *           hlp1_PAHH=GPOW(i_PAHH+nlb1_PAHH,(1<<a_PAHH)-c_PAH,a_PAHH)
     *           hlp1_PAHH=SELECT(hlp1_PAHH,gpp[i_PAHH+nlb1_PAHH+a_PAHH],c_PAH==0)
     *           hlp_PAHH*=hlp1_PAHH; cost["M"]+=1
     *       cost["M"]+=1
     *       u2_PAHH=u_PAHH*h1_PAHH
     *       [base lookup] d_PAH=_tmp>>(w-(96-(i_PAHH+a_PAHH)))
     *       c_PAH=c_PAH+((d_PAH-1)%2^b_PAHH)*(2^a_PAHH)
     * ========================================================================= */
    fp_copy(u_PAHH, h0_PAH);
    fp_copy(h0_PAHH, u_PAHH);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_PAHH, h0_PAHH);
    //countS += 6;
	bn_t tmp_bn;
	dig_t d;
	bn_null(tmp_bn);bn_new(tmp_bn);
    fp_prime_back(tmp_bn, h0_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1fff;
    _tmp=rlll[d];
//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_PAHH) == RLC_EQ) _tmp = _ii;
    uint64_t c_PAH = (uint64_t)_tmp >> 2;   /* >> (w - lb) = >> (8-6) = >> 2 */
 
    GPOW_i84_e6(h1_PAHH, (one_k << 6) - c_PAH, gw);
    SELECT(h1_PAHH, gpp[84 + 6], c_PAH == 0, h1_PAHH);
    //countM++;
    fp_mul(u2_PAHH, u_PAHH, h1_PAHH);
    fp_prime_back(tmp_bn, u2_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1fff;
    _tmp=rlll[d];
//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_PAHH) == RLC_EQ) _tmp = _ii;
    uint64_t d_PAH = (uint64_t)_tmp >> 2;   /* >> (w - lb) = >> 2            */
    c_PAH = c_PAH + ((d_PAH - 1) % 64) * 64;
    /* =========================================================================
     * Sage line 261-268:
     * h1_PAH  = GPOW_i72_e12((1<<12) - c_PAH)
     * h1_PAH  = SELECT(h1_PAH, gpp[72+12], c_PAH==0)
     * hlp1_PA = GPOW_i78_e12((1<<12) - c_PAH)
     * hlp1_PA = SELECT(hlp1_PA, gpp[78+12], c_PAH==0)
     * hlp_PAH *= hlp1_PA
     * cost["M"] += 1
     * cost["M"] += 1
     * u2_PAH = u_PAH * h1_PAH
     * ========================================================================= */
    //printf("\n%lx\n",c_PAH);
    GPOW_i72_e12(h1_PAH,  (one_k << 12) - c_PAH, gw);
    SELECT(h1_PAH,  gpp[72 + 12], c_PAH == 0, h1_PAH);
    GPOW_i78_e12(hlp1_PA, (one_k << 12) - c_PAH, gw);
    SELECT(hlp1_PA, gpp[78 + 12], c_PAH == 0, hlp1_PA);
    fp_mul(hlp_PAH, hlp_PAH, hlp1_PA); //countM++;
    //countM++;
    fp_mul(u2_PAH, u_PAH, h1_PAH);
    //fp_print(u2_PAH);
    /* =========================================================================
     * d_PAH = DLPpow2(u2_PAH, i=84, helper=hlp_PAH)
     * helper IS provided -> else branch: h0 = helper directly, NO squaring
     *
     * Sage:
     * h0_PAHL = hlp_PAH          <- else branch: h0 = helper
     * (no squaring loop)
     * for _ii in range(256): if rll[_ii]==h0_PAHL: _tmp=_ii
     * c_PAHL = _tmp >> 2
     * h1_PAHL = GPOW_i84_e6((1<<6) - c_PAHL)
     * h1_PAHL = SELECT(h1_PAHL, gpp[84+6], c_PAHL==0)
     * cost["M"] += 1
     * u2_PAHL = u2_PAH * h1_PAHL
     * for _ii in range(256): if rll[_ii]==u2_PAHL: _tmp=_ii
     * d_PAHL = _tmp >> 2
     * d_PAH = c_PAHL + ((d_PAHL-1) % 64) * 64
     * ========================================================================= */
    fp_copy(h0_PAHL, hlp_PAH);   /* else branch: h0 = helper, no squaring */
    fp_prime_back(tmp_bn, h0_PAHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1fff;
    _tmp=rlll[d];
//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_PAHL) == RLC_EQ) _tmp = _ii;
    uint64_t c_PAHL = (uint64_t)_tmp >> 2;

    GPOW_i84_e6(h1_PAHL, (one_k << 6) - c_PAHL, gw);
    SELECT(h1_PAHL, gpp[84 + 6], c_PAHL == 0, h1_PAHL);
    //countM++;
    fp_mul(u2_PAHL, u2_PAH, h1_PAHL);
    fp_prime_back(tmp_bn, u2_PAHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1fff;
    _tmp=rlll[d];
//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_PAHL) == RLC_EQ) _tmp = _ii;
    uint64_t d_PAHL = (uint64_t)_tmp >> 2;
    d_PAH  = c_PAHL + ((d_PAHL - 1) % 64) * 64;

    /* =========================================================================
     * Sage line 292:
     * c_PA = c_PAH + ((d_PAH-1) % 4096) * 4096
     * ========================================================================= */
    uint64_t c_PA = c_PAH + ((d_PAH - 1) % 4096) * 4096;
    //printf("\n%lx\n\n%lx\n%lx\n",d_PAHL,d_PAH,c_PA);
    /* =========================================================================
     * Sage line 295-302:
     * h1_PA  = GPOW_i48_e24((1<<24) - c_PA)
     * h1_PA  = SELECT(h1_PA, gpp[48+24], c_PA==0)
     * hlp1_A = GPOW_i60_e24((1<<24) - c_PA)
     * hlp1_A = SELECT(hlp1_A, gpp[60+24], c_PA==0)
     * hlp_PA *= hlp1_A
     * cost["M"] += 1
     * cost["M"] += 1
     * u2_PA = u_PA * h1_PA
     * ========================================================================= */
    GPOW_i48_e24(h1_PA,  (one_k << 24) - c_PA, gw);
    SELECT(h1_PA,  gpp[48 + 24], c_PA == 0, h1_PA);
    GPOW_i60_e24(hlp1_A, (one_k << 24) - c_PA, gw);
    SELECT(hlp1_A, gpp[60 + 24], c_PA == 0, hlp1_A);
    fp_mul(hlp_PA, hlp_PA, hlp1_A); //countM++;
    //countM++;
    fp_mul(u2_PA, u_PA, h1_PA);
    //fp_print(u2_PA);
    /* =========================================================================
     * d_PA = DLPpow2(u2_PA, i=72, helper=hlp_PA)
     * helper IS provided -> else branch: h0 = helper directly, NO squaring
     *   h0_PAL = hlp_PA     <- else branch
     *   hlp_PAL = None      <- no hlp captured (helper branch skips squaring)
     *
     * Then HIGH sub-call: DLPpow2(h0_PAL, i=84, helper=None)
     *   helper=None -> if branch -> squares b=6 times
     *
     * Sage:
     * h0_PAL = hlp_PA           <- else: h0 = helper
     * hlp_PAL = None
     * u_PALH = h0_PAL           <- HIGH sub-call input
     * h0_PALH = u_PALH
     * for _j in range(6): h0_PALH *= h0_PALH   <- if branch squares
     * cost["S"] += 6
     * for _ii in range(256): if rll[_ii]==h0_PALH: _tmp=_ii
     * c_PAL = _tmp >> 2
     * h1_PAL = GPOW_i72_e12((1<<12) - c_PAL)
     * h1_PAL = SELECT(h1_PAL, gpp[72+12], c_PAL==0)
     * cost["M"] += 1
     * u2_PAL = u_PALH * h1_PAL
     * ========================================================================= */
    fp_copy(h0_PAL, hlp_PA);     /* else branch: h0 = helper, no squaring here */
    /* hlp_PAL = None */
    //fp_print(h0_PAL);
    fp_copy(u_PALH, h0_PAL);/* HIGH sub-call: helper=None -> if branch -> square */
    //fp_print(u_PALH);
    fp_copy(h0_PALH, u_PALH);
    //fp_print(h0_PALH);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_PALH, h0_PALH);
    //fp_print(h0_PALH);
    //countS += 6;
    fp_prime_back(tmp_bn, h0_PALH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1fff;
    _tmp=rlll[d];
//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_PALH) == RLC_EQ) _tmp = _ii;
    uint64_t c_PAL = (uint64_t)_tmp >> 2;

    GPOW_i84_e6(h1_PALH, (one_k << 6) - c_PAL, gw);
    //fp_print(h1_PALH);
    SELECT(h1_PALH, gpp[84 + 6], c_PAL == 0, h1_PALH);
    //printf("\n%d\n",)
    //fp_print(h1_PALH);
    //countM++;
    fp_mul(u2_PALH, u_PALH, h1_PALH);
    //fp_print(u2_PALH);
    fp_prime_back(tmp_bn, u2_PALH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1fff;
    _tmp=rlll[d];
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_PALH) == RLC_EQ) _tmp = _ii;
    uint64_t d_PAHL_leaf= (uint64_t)_tmp >> 2;
    c_PAL=c_PAL+((d_PAHL_leaf-1)&0x3f)*64;
    //printf("\n%lx\n",c_PAL);
    
    GPOW_i72_e12(h1_PAL, (one_k << 12) - c_PAL, gw);
    SELECT(h1_PAL, gpp[72 + 12], c_PAL == 0, h1_PAL);
    //countM++;
    fp_mul(u2_PAL,u2_PA,h1_PAL);
    //fp_print(u2_PAL);
    
    /* =========================================================================
     * LOW sub-call of d_PA: DLPpow2(u2_PAL, i=84, helper=hlp_PAL=None)
     * helper=None -> if branch -> squares b=6 times
     * b=6 <= w=8 -> do_hlp=False -> no hlp captured
     *
     * Sage:
     * u_PALL = u2_PAL
     * h0_PALL = u_PALL
     * for _j in range(6): h0_PALL *= h0_PALL    <- if branch squares
     * cost["S"] += 6
     * for _ii in range(256): if rll[_ii]==h0_PALL: _tmp=_ii
     * c_PALL = _tmp >> 2
     * h1_PALL = GPOW_i84_e6((1<<6) - c_PALL)
     * h1_PALL = SELECT(h1_PALL, gpp[84+6], c_PALL==0)
     * cost["M"] += 1
     * u2_PALL = u_PALL * h1_PALL
     * for _ii in range(256): if rll[_ii]==u2_PALL: _tmp=_ii
     * d_PALL = _tmp >> 2
     * d_PAL = c_PALL + ((d_PALL-1) % 64) * 64
     * d_PA = c_PAL + ((d_PAL-1) % 4096) * 4096
     * c_A = c_PA + ((d_PA-1) % (2^24)) * (2^24)
     * ========================================================================= */
    fp_copy(u_PALL, u2_PAL);
    fp_copy(h0_PALL, u_PALL);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_PALL, h0_PALL);
    //countS += 6;
    fp_prime_back(tmp_bn, h0_PALL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1fff;
    _tmp=rlll[d];
//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_PALL) == RLC_EQ) _tmp = _ii;
    uint64_t c_PALL = (uint64_t)_tmp >> 2;

    GPOW_i84_e6(h1_PALL, (one_k << 6) - c_PALL, gw);
    SELECT(h1_PALL, gpp[84 + 6], c_PALL == 0, h1_PALL);
    //countM++;
    fp_mul(u2_PALL, u_PALL, h1_PALL);
    fp_prime_back(tmp_bn, u2_PALL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1fff;
    _tmp=rlll[d];
//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_PALL) == RLC_EQ) _tmp = _ii;
    uint64_t d_PALL = (uint64_t)_tmp >> 2;
    uint64_t d_PAL  = c_PALL + ((d_PALL - 1) % 64)   * 64;
    uint64_t d_PA   = c_PAL  + ((d_PAL  - 1) % 4096) * 4096;
    uint64_t c_A    = c_PA   + ((d_PA   - 1) % ((uint64_t)1 << 24)) * ((uint64_t)1 << 24);
   //printf("\n%lx\n%lx\n%lx\n%lx\n",d_PALL,d_PAL,d_PA,c_A);
    /* =========================================================================
     * Sage line 357-364:
     * f_A  = GPOW_i0_e47(((1<<48)-c_A+1)>>1)
     * f_A  = SELECT(f_A, gpp[47], c_A==0)
     * hlp1 = GPOW_i24_e48((1<<48)-c_A)
     * hlp1 = SELECT(hlp1, gpp[24+48], c_A==0)
     * hlp_A *= hlp1
     * cost["M"] += 1
     * cost["M"] += 1
     * cost["S"] += 1
     * ========================================================================= */
    GPOW_i0_e47(f_A,  ((one_k << 48) - c_A + 1) >> 1, gw);
    SELECT(f_A,  gpp[47], c_A == 0, f_A);
    GPOW_i24_e48(hlp1, (one_k << 48) - c_A, gw);
    SELECT(hlp1, gpp[24 + 48], c_A == 0, hlp1);
    fp_mul(hlp_A, hlp_A, hlp1); //countM++;
    //countM++;
    //countS++;
    //fp_print(f_A);
    //fp_print(hlp_A);

    /* =========================================================================
     * Sage line 374-376:
     * u_X0  = u_A * f_A^2
     * h0_X0 = hlp_A
     * hlp_X0 = None
     * ========================================================================= */
    fp_sqr(f_A_sq, f_A);
    fp_mul(u_X0, u_A, f_A_sq);
    fp_copy(h0_X0, hlp_A);
    /* hlp_X0 = None */

    /* =========================================================================
     * Sage line 381-388:
     * u_X0P = h0_X0
     * hlp_X0P = None
     * h0_X0P = u_X0P
     * for _j in range(12):
     *     if _j == 6: hlp_X0P = h0_X0P
     *     h0_X0P *= h0_X0P
     * cost["S"] += 12
     * ========================================================================= */
    fp_copy(u_X0P, h0_X0);
    fp_copy(h0_X0P, u_X0P);
    for (int _j = 0; _j < 12; _j++) {
        if (_j == 6) fp_copy(hlp_X0P, h0_X0P);
        fp_sqr(h0_X0P, h0_X0P);
    }
    //countS += 12;
    //fp_print(h0_X0P);

    /* =========================================================================
     * Sage line 391-399:
     * u_X0PH = h0_X0P
     * h0_X0PH = u_X0PH
     * for _j in range(6): h0_X0PH *= h0_X0PH
     * cost["S"] += 6
     * for _ii in range(256): if rll[_ii]==h0_X0PH: _tmp=_ii
     * c_X0P = _tmp >> 2
     * ========================================================================= */
    fp_copy(u_X0PH, h0_X0P);
    fp_copy(h0_X0PH, u_X0PH);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_X0PH, h0_X0PH);
    //countS += 6;
    fp_prime_back(tmp_bn, h0_X0PH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1fff;
    _tmp=rlll[d];
//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_X0PH) == RLC_EQ) _tmp = _ii;
    uint64_t c_X0P = (uint64_t)_tmp >> 2;
    GPOW_i84_e6(h1_X0PH,  (one_k << 6) - c_X0P, gw);
    SELECT(h1_X0PH,  gpp[84 + 6], c_X0P == 0, h1_X0PH);
    //countM++;
    fp_mul(u2_X0PH,u_X0PH,h1_X0PH);
    fp_prime_back(tmp_bn, u2_X0PH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1fff;
    _tmp=rlll[d];
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_X0PH) == RLC_EQ) _tmp = _ii;
    uint64_t d_X0PH=_tmp>>(w-(96-(84+6)));
    c_X0P=c_X0P+((d_X0PH-1)&0x3f)*64;
    //printf("\n%lx\n",c_X0P);
    /* =========================================================================
     * Sage line 401-408:
     * h1_X0P  = GPOW_i72_e12((1<<12) - c_X0P)
     * h1_X0P  = SELECT(h1_X0P, gpp[72+12], c_X0P==0)
     * hlp1_X0 = GPOW_i78_e12((1<<12) - c_X0P)
     * hlp1_X0 = SELECT(hlp1_X0, gpp[78+12], c_X0P==0)
     * hlp_X0P *= hlp1_X0
     * cost["M"] += 1
     * cost["M"] += 1
     * u2_X0P = u_X0P * h1_X0P
     * ========================================================================= */
    GPOW_i72_e12(h1_X0P,  (one_k << 12) - c_X0P, gw);
    SELECT(h1_X0P,  gpp[72 + 12], c_X0P == 0, h1_X0P);
    GPOW_i78_e12(hlp1_X0, (one_k << 12) - c_X0P, gw);
    SELECT(hlp1_X0, gpp[78 + 12], c_X0P == 0, hlp1_X0);
    fp_mul(hlp_X0P, hlp_X0P, hlp1_X0); //countM++;
    //countM++;
    fp_mul(u2_X0P, u_X0P, h1_X0P);

    /* =========================================================================
     * low: DLPpow2(u2_X0P, i=84, helper=hlp_X0P)
     * helper IS provided -> else branch: h0 = helper directly, NO squaring
     *
     * Sage:
     * h0_X0PL = hlp_X0P          <- else branch: h0 = helper
     * (no squaring)
     * for _ii in range(256): if rll[_ii]==h0_X0PL: _tmp=_ii
     * c_X0PL = _tmp >> 2
     * h1_X0PL = GPOW_i84_e6((1<<6) - c_X0PL)
     * h1_X0PL = SELECT(h1_X0PL, gpp[84+6], c_X0PL==0)
     * cost["M"] += 1
     * u2_X0PL = u2_X0P * h1_X0PL
     * for _ii in range(256): if rll[_ii]==u2_X0PL: _tmp=_ii
     * d_X0PL = _tmp >> 2
     * d_X0P  = c_X0PL + ((d_X0PL-1) % 64) * 64
     * c_X0   = c_X0P  + ((d_X0P-1) % 4096) * 4096
     * ========================================================================= */
    fp_copy(h0_X0PL, hlp_X0P);   /* else branch: h0 = helper, no squaring */
    fp_prime_back(tmp_bn, h0_X0PL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1fff;
    _tmp=rlll[d];
//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_X0PL) == RLC_EQ) _tmp = _ii;
    uint64_t c_X0PL = (uint64_t)_tmp >> 2;

    GPOW_i84_e6(h1_X0PL, (one_k << 6) - c_X0PL, gw);
    SELECT(h1_X0PL, gpp[84 + 6], c_X0PL == 0, h1_X0PL);
    //countM++;
    fp_mul(u2_X0PL, u2_X0P, h1_X0PL);
    fp_prime_back(tmp_bn, u2_X0PL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1fff;
    _tmp=rlll[d];
//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_X0PL) == RLC_EQ) _tmp = _ii;
    uint64_t d_X0PL = (uint64_t)_tmp >> 2;
    uint64_t d_X0P  = c_X0PL + ((d_X0PL - 1) % 64)   * 64;
    uint64_t c_X0   = c_X0P  + ((d_X0P  - 1) % 4096) * 4096;
    //printf("\n%lx\n%lx\n%lx\n",d_X0PL,d_X0P,c_X0);

    /* =========================================================================
     * Sage line 430-434:
     * f_X0 = GPOW_i47_e24((1<<24) - c_X0)
     * f_X0 = SELECT(f_X0, gpp[47+24], c_X0==0)
     * cost["M"] += 1
     * cost["S"] += 1
     * ========================================================================= */
    GPOW_i47_e24(f_X0, (one_k << 24) - c_X0, gw);
    SELECT(f_X0, gpp[47 + 24], c_X0 == 0, f_X0);
    //countM++;
    //countS++;

    /* =========================================================================
     * Sage line 438-446:
     * u_X1  = u_X0 * f_X0^2
     * hlp_X1 = None
     * h0_X1 = u_X1
     * for _j in range(12):
     *     if _j == 6: hlp_X1 = h0_X1
     *     h0_X1 *= h0_X1
     * cost["S"] += 12
     * ========================================================================= */
    fp_sqr(f_X0_sq, f_X0);
    fp_mul(u_X1, u_X0, f_X0_sq);
    fp_copy(h0_X1, u_X1);
    for (int _j = 0; _j < 12; _j++) {
        if (_j == 6) fp_copy(hlp_X1, h0_X1);
        fp_sqr(h0_X1, h0_X1);
    }
    //countS += 12;
    //fp_print(h0_X1);
    /* =========================================================================
     * Sage line 450-460:
     * u_X1P  = h0_X1
     * hlp_X1P = None
     * h0_X1P = u_X1P
     * for _j in range(6): h0_X1P *= h0_X1P
     * cost["S"] += 6
     * for _ii in range(256): if rll[_ii]==h0_X1P: _tmp=_ii
     * c_X1P = _tmp >> 2
     * ========================================================================= */
    fp_copy(u_X1P, h0_X1);
    fp_copy(h0_X1P, u_X1P);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_X1P, h0_X1P);
    //countS += 6;
    fp_prime_back(tmp_bn, h0_X1P);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1fff;
    _tmp=rlll[d];
//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_X1P) == RLC_EQ) _tmp = _ii;
    uint64_t c_X1P = (uint64_t)_tmp >> 2;
    //printf("\n%lx\n",c_X1P);

    /* =========================================================================
     * Sage line 462-468:
     * h1_X1P = GPOW_i84_e6((1<<6) - c_X1P)
     * h1_X1P = SELECT(h1_X1P, gpp[84+6], c_X1P==0)
     * cost["M"] += 1
     * u2_X1P = u_X1P * h1_X1P
     * ========================================================================= */
    GPOW_i84_e6(h1_X1P, (one_k << 6) - c_X1P, gw);
    SELECT(h1_X1P, gpp[84 + 6], c_X1P == 0, h1_X1P);
    //countM++;
    fp_mul(u2_X1P, u_X1P, h1_X1P);
    //fp_print(u2_X1P);
    /* =========================================================================
     * Sage line 471-488:
     * h0_X1L = u2_X1P
     * for _j in range(6): h0_X1L *= h0_X1L
     * cost["S"] += 6
     * for _ii in range(256): if rll[_ii]==h0_X1L: _tmp=_ii
     * c_X1L = _tmp >> 2
     * h1_X1L = GPOW_i84_e6((1<<6) - c_X1L)
     * h1_X1L = SELECT(h1_X1L, gpp[84+6], c_X1L==0)
     * cost["M"] += 1
     * u2_X1L = u2_X1P * h1_X1L
     * for _ii in range(256): if rll[_ii]==u2_X1L: _tmp=_ii
     * d_X1L = _tmp >> 2
     * d_X1P = c_X1L + ((d_X1L-1) % 64) * 64
     * c_X1  = c_X1P + ((d_X1P-1) % 4096) * 4096
     * ========================================================================= */
    fp_copy(u_X1PL, u2_X1P);
    fp_prime_back(tmp_bn, u_X1PL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1fff;
    _tmp=rlll[d];
//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u_X1PL) == RLC_EQ) _tmp = _ii;
    uint64_t d_X1P = (uint64_t)_tmp >> 2;
  
//    GPOW_i84_e6(h1_X1L, (one_k << 6) - c_X1L, gw);
//    SELECT(h1_X1L, gpp[84 + 6], c_X1L == 0, h1_X1L);
//    countM++;
//    fp_mul(u2_X1L, u2_X1P, h1_X1L);
//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_X1L) == RLC_EQ) _tmp = _ii;
//    uint64_t d_X1L = (uint64_t)_tmp >> 2;
//    uint64_t d_X1P = c_X1L + ((d_X1L - 1) % 64)   * 64;
    uint64_t c_X1  = c_X1P + ((d_X1P - 1) & 0x3f) * 64;
    //printf("\n%lx\n%lx\n",d_X1P,c_X1);

    /* =========================================================================
     * Sage line 493-500:
     * f_X1   = GPOW_i71_e12((1<<12) - c_X1)
     * f_X1   = SELECT(f_X1, gpp[71+12], c_X1==0)
     * hlp1_1 = GPOW_i78_e12((1<<12) - c_X1)
     * hlp1_1 = SELECT(hlp1_1, gpp[78+12], c_X1==0)
     * hlp_X1 *= hlp1_1
     * cost["M"] += 1
     * cost["M"] += 1
     * cost["S"] += 1
     * ========================================================================= */
    GPOW_i71_e12(f_X1,   (one_k << 12) - c_X1, gw);
    SELECT(f_X1,   gpp[71 + 12], c_X1 == 0, f_X1);
    GPOW_i78_e12(hlp1_1, (one_k << 12) - c_X1, gw);
    SELECT(hlp1_1, gpp[78 + 12], c_X1 == 0, hlp1_1);
    fp_mul(hlp_X1, hlp_X1, hlp1_1); //countM++;
    //countM++;
    //countS++;

    /* =========================================================================
     * Sage line 504-512:
     * u_X2  = u_X1 * f_X1^2
     * h0_X2 = hlp_X1    <- else branch: helper provided, h0 = helper directly
     * (no squaring — helper branch skips the squaring loop)
     * for _ii in range(256): if rll[_ii]==h0_X2: _tmp=_ii
     * c_X2 = _tmp >> 2
     * ========================================================================= */
    fp_sqr(f_X1_sq, f_X1);
    fp_mul(u_X2, u_X1, f_X1_sq);
    fp_copy(h0_X2, hlp_X1);
    fp_prime_back(tmp_bn, h0_X2);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1fff;
    _tmp=rlll[d];
//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_X2) == RLC_EQ) _tmp = _ii;
    uint64_t c_X2 = (uint64_t)_tmp >> 2;

    /* =========================================================================
     * Sage line 515-518:
     * f_X2 = GPOW_i83_e6((1<<6) - c_X2)
     * f_X2 = SELECT(f_X2, gpp[83+6], c_X2==0)
     * cost["M"] += 1
     * cost["S"] += 1
     * ========================================================================= */
    GPOW_i83_e6(f_X2, (one_k << 6) - c_X2, gw);
    SELECT(f_X2, gpp[83 + 6], c_X2 == 0, f_X2);
    //countM++;
    //countS++;

    /* =========================================================================
     * Sage line 521-527:
     * u_X3  = u_X2 * f_X2^2
     * for _ii in range(256): if rll[_ii]==u_X3: _tmp=_ii
     * c1_X3 = _tmp >> 2
     * d1_X3 = fll[c1_X3 << 2]
     * e_X3, z_X3 = c1_X3, d1_X3
     * ========================================================================= */
    fp_sqr(f_X2_sq, f_X2);
    fp_mul(u_X3, u_X2, f_X2_sq);
    fp_prime_back(tmp_bn, u_X3);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1fff;
    _tmp=rlll[d];
//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u_X3) == RLC_EQ) _tmp = _ii;
    uint64_t c1_X3 = (uint64_t)_tmp >> 2;
    fp_copy(z_X3, fll[c1_X3 << 2]);   /* d1_X3 = fll[c1_X3<<2]; z_X3 = d1_X3 */
    //uint64_t e_X3  = c1_X3;           /* e_X3 = c1_X3 */

    /* =========================================================================
     * Sage line 533-548:
     * t_X2 = f_X2 * z_X3;  cost["M"] += 1
     * e_X2 = c_X2 + ((e_X3-1) % 64) * 64
     * t_X1 = f_X1 * t_X2;  cost["M"] += 1
     * e_X1 = c_X1 + ((e_X2-1) % 4096) * 4096
     * t_X0 = f_X0 * t_X1;  cost["M"] += 1
     * e_X0 = c_X0 + ((e_X1-1) % (2^24)) * (2^24)
     * t_A  = f_A * t_X0;   cost["M"] += 1
     * e_final = c_A + ((e_X0-1) % (2^48)) * (2^48)
     * return (e_final, t_A)
     * ========================================================================= */
    fp_mul(t_X2, f_X2, z_X3); //countM++;
    //uint64_t e_X2 = c_X2 + ((e_X3 - 1) % 64)   * 64;

    fp_mul(t_X1, f_X1, t_X2); //countM++;
    //uint64_t e_X1 = c_X1 + ((e_X2 - 1) % 4096) * 4096;

    fp_mul(t_X0, f_X0, t_X1); //countM++;
    //uint64_t e_X0 = c_X0 + ((e_X1 - 1) % ((uint64_t)1 << 24)) * ((uint64_t)1 << 24);

    fp_mul(out_t, f_A, t_X0);   //countM++;
    //uint64_t e_final = c_A + ((e_X0 - 1) % ((uint64_t)1 << 48)) * ((uint64_t)1 << 48);
    //(void)e_final; /* not needed by sqrt_ext caller */

    //fp_copy(out_t, t_A);

    /* free all */
    fp_free(h0_A);    fp_free(hlp_A);   fp_free(f_A);
    fp_free(u_PA);    fp_free(hlp_PA);  fp_free(h0_PA);
    fp_free(u_PAH);   fp_free(hlp_PAH); fp_free(h0_PAH);
    fp_free(u_PAHH);  fp_free(h0_PAHH);
    fp_free(h1_PAHH); fp_free(hlp_PAHH); fp_free(hlp1_PAHH); fp_free(u2_PAHH);
    fp_free(h1_PAH);  fp_free(hlp1_PA); fp_free(u2_PAH);
    fp_free(h0_PAHL); fp_free(h1_PAHL); fp_free(u2_PAHL);
    fp_free(h1_PA);   fp_free(hlp1_A);  fp_free(u2_PA);
    fp_free(h0_PAL);  fp_free(hlp_PAL); fp_free(u_PALH);
    fp_free(h0_PALH); fp_free(h1_PAL);  fp_free(u2_PAL);
    fp_free(u_PALL);  fp_free(h0_PALL); fp_free(h1_PALL); fp_free(u2_PALL);
    fp_free(hlp1);    fp_free(f_A_sq);
    fp_free(u_X0);    fp_free(h0_X0);
    fp_free(u_X0P);   fp_free(hlp_X0P); fp_free(h0_X0P);
    fp_free(u_X0PH);  fp_free(h0_X0PH);
    fp_free(h1_X0P);  fp_free(hlp1_X0); fp_free(u2_X0P);
    fp_free(h0_X0PL); fp_free(h1_X0PL); fp_free(u2_X0PL);
    fp_free(f_X0);    fp_free(f_X0_sq);
    fp_free(u_X1);    fp_free(hlp_X1);  fp_free(h0_X1);
    fp_free(u_X1P);   fp_free(hlp_X1P); fp_free(h0_X1P);
    fp_free(h1_X1P);  fp_free(u2_X1P);
    fp_free(h0_X1L);  fp_free(h1_X1L);  fp_free(u2_X1L);
    fp_free(f_X1);    fp_free(hlp1_1);  fp_free(f_X1_sq);
    fp_free(u_X2);    fp_free(h0_X2);
    fp_free(f_X2);    fp_free(f_X2_sq);
    fp_free(u_X3);    fp_free(z_X3);
    fp_free(t_X2);    fp_free(t_X1);    fp_free(t_X0);    fp_free(t_A);
}

/* =========================================================================
 * sqrt_ext — same as Sage sqrt_ext
 * Sage:
 *   v  = x^((m-1)//2)
 *   w_ = x * v
 *   u  = w_ * v
 *   cost["M"] += 2
 *   e, t = DLPpow2ext(u, 0)
 *   cost["M"] += 1
 *   return w_ * t
 * ========================================================================= */
void sqrt_ext(fp_t x, fp_t y, bn_t e_exp,
              fp_t gw[nw][we], int rlll[8192], fp_t rll[we], fp_t fll[we], fp_t gpp[n])
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

int rlll[8192]={0};


rlll[ 0x1 ]= 0 ;
rlll[ 0xfbd ]= 1 ;
rlll[ 0xa11 ]= 2 ;
rlll[ 0x7a6 ]= 3 ;
rlll[ 0xb4b ]= 4 ;
rlll[ 0x162f ]= 5 ;
rlll[ 0x72f ]= 6 ;
rlll[ 0xf3f ]= 7 ;
rlll[ 0x1ed3 ]= 8 ;
rlll[ 0x1b45 ]= 9 ;
rlll[ 0xc10 ]= 10 ;
rlll[ 0x1aa2 ]= 11 ;
rlll[ 0xbe0 ]= 12 ;
rlll[ 0x1e8b ]= 13 ;
rlll[ 0x1a8d ]= 14 ;
rlll[ 0x1c92 ]= 15 ;
rlll[ 0xf2f ]= 16 ;
rlll[ 0xeaf ]= 17 ;
rlll[ 0x621 ]= 18 ;
rlll[ 0x1354 ]= 19 ;
rlll[ 0x1ea5 ]= 20 ;
rlll[ 0x1d70 ]= 21 ;
rlll[ 0x191f ]= 22 ;
rlll[ 0x1e72 ]= 23 ;
rlll[ 0x1ab3 ]= 24 ;
rlll[ 0xdd0 ]= 25 ;
rlll[ 0x156e ]= 26 ;
rlll[ 0x181b ]= 27 ;
rlll[ 0x6c5 ]= 28 ;
rlll[ 0x1ddf ]= 29 ;
rlll[ 0x16e8 ]= 30 ;
rlll[ 0x1a63 ]= 31 ;
rlll[ 0xdbd ]= 32 ;
rlll[ 0x1e1c ]= 33 ;
rlll[ 0x115b ]= 34 ;
rlll[ 0x1cb6 ]= 35 ;
rlll[ 0x405 ]= 36 ;
rlll[ 0x1989 ]= 37 ;
rlll[ 0xfe7 ]= 38 ;
rlll[ 0xc36 ]= 39 ;
rlll[ 0x1762 ]= 40 ;
rlll[ 0x1da9 ]= 41 ;
rlll[ 0x142c ]= 42 ;
rlll[ 0x189 ]= 43 ;
rlll[ 0x1222 ]= 44 ;
rlll[ 0x1c73 ]= 45 ;
rlll[ 0x1b85 ]= 46 ;
rlll[ 0x13c8 ]= 47 ;
rlll[ 0x19b0 ]= 48 ;
rlll[ 0x16b1 ]= 49 ;
rlll[ 0x257 ]= 50 ;
rlll[ 0x1c0c ]= 51 ;
rlll[ 0x1f37 ]= 52 ;
rlll[ 0x3e9 ]= 53 ;
rlll[ 0x16a4 ]= 54 ;
rlll[ 0x1417 ]= 55 ;
rlll[ 0x37c ]= 56 ;
rlll[ 0x15d1 ]= 57 ;
rlll[ 0x60 ]= 58 ;
rlll[ 0x1139 ]= 59 ;
rlll[ 0x1002 ]= 60 ;
rlll[ 0x1b63 ]= 61 ;
rlll[ 0x1fca ]= 62 ;
rlll[ 0xe1c ]= 63 ;
rlll[ 0x1e19 ]= 64 ;
rlll[ 0x73e ]= 65 ;
rlll[ 0x4f8 ]= 66 ;
rlll[ 0x1f75 ]= 67 ;
rlll[ 0xf8b ]= 68 ;
rlll[ 0x1abd ]= 69 ;
rlll[ 0x1c5e ]= 70 ;
rlll[ 0xf6f ]= 71 ;
rlll[ 0x145e ]= 72 ;
rlll[ 0x51 ]= 73 ;
rlll[ 0xc71 ]= 74 ;
rlll[ 0x1fd ]= 75 ;
rlll[ 0x687 ]= 76 ;
rlll[ 0x1eb9 ]= 77 ;
rlll[ 0x131b ]= 78 ;
rlll[ 0x40 ]= 79 ;
rlll[ 0x8b8 ]= 80 ;
rlll[ 0xd74 ]= 81 ;
rlll[ 0x15fb ]= 82 ;
rlll[ 0x49f ]= 83 ;
rlll[ 0x663 ]= 84 ;
rlll[ 0x10cc ]= 85 ;
rlll[ 0x1f1d ]= 86 ;
rlll[ 0x1d6d ]= 87 ;
rlll[ 0x10a ]= 88 ;
rlll[ 0xa0e ]= 89 ;
rlll[ 0x1ae ]= 90 ;
rlll[ 0x127d ]= 91 ;
rlll[ 0x383 ]= 92 ;
rlll[ 0x989 ]= 93 ;
rlll[ 0x1a29 ]= 94 ;
rlll[ 0x17c1 ]= 95 ;
rlll[ 0x1109 ]= 96 ;
rlll[ 0x456 ]= 97 ;
rlll[ 0x1ffa ]= 98 ;
rlll[ 0x7b0 ]= 99 ;
rlll[ 0x1e69 ]= 100 ;
rlll[ 0xcaf ]= 101 ;
rlll[ 0x1fb8 ]= 102 ;
rlll[ 0x19 ]= 103 ;
rlll[ 0xe96 ]= 104 ;
rlll[ 0xd8f ]= 105 ;
rlll[ 0xa74 ]= 106 ;
rlll[ 0x18a2 ]= 107 ;
rlll[ 0x1765 ]= 108 ;
rlll[ 0x1b86 ]= 109 ;
rlll[ 0x1559 ]= 110 ;
rlll[ 0x12b5 ]= 111 ;
rlll[ 0xcac ]= 112 ;
rlll[ 0x298 ]= 113 ;
rlll[ 0xec6 ]= 114 ;
rlll[ 0x17c2 ]= 115 ;
rlll[ 0x1a72 ]= 116 ;
rlll[ 0x1db2 ]= 117 ;
rlll[ 0x114f ]= 118 ;
rlll[ 0x122d ]= 119 ;
rlll[ 0x301 ]= 120 ;
rlll[ 0x873 ]= 121 ;
rlll[ 0x776 ]= 122 ;
rlll[ 0x1c81 ]= 123 ;
rlll[ 0x8d3 ]= 124 ;
rlll[ 0x1b07 ]= 125 ;
rlll[ 0x65b ]= 126 ;
rlll[ 0xea4 ]= 127 ;
rlll[ 0x0 ]= 128 ;
rlll[ 0x1044 ]= 129 ;
rlll[ 0x15f0 ]= 130 ;
rlll[ 0x185b ]= 131 ;
rlll[ 0x14b6 ]= 132 ;
rlll[ 0x9d2 ]= 133 ;
rlll[ 0x18d2 ]= 134 ;
rlll[ 0x10c2 ]= 135 ;
rlll[ 0x12e ]= 136 ;
rlll[ 0x4bc ]= 137 ;
rlll[ 0x13f1 ]= 138 ;
rlll[ 0x55f ]= 139 ;
rlll[ 0x1421 ]= 140 ;
rlll[ 0x176 ]= 141 ;
rlll[ 0x574 ]= 142 ;
rlll[ 0x36f ]= 143 ;
rlll[ 0x10d2 ]= 144 ;
rlll[ 0x1152 ]= 145 ;
rlll[ 0x19e0 ]= 146 ;
rlll[ 0xcad ]= 147 ;
rlll[ 0x15c ]= 148 ;
rlll[ 0x291 ]= 149 ;
rlll[ 0x6e2 ]= 150 ;
rlll[ 0x18f ]= 151 ;
rlll[ 0x54e ]= 152 ;
rlll[ 0x1231 ]= 153 ;
rlll[ 0xa93 ]= 154 ;
rlll[ 0x7e6 ]= 155 ;
rlll[ 0x193c ]= 156 ;
rlll[ 0x222 ]= 157 ;
rlll[ 0x919 ]= 158 ;
rlll[ 0x59e ]= 159 ;
rlll[ 0x1244 ]= 160 ;
rlll[ 0x1e5 ]= 161 ;
rlll[ 0xea6 ]= 162 ;
rlll[ 0x34b ]= 163 ;
rlll[ 0x1bfc ]= 164 ;
rlll[ 0x678 ]= 165 ;
rlll[ 0x101a ]= 166 ;
rlll[ 0x13cb ]= 167 ;
rlll[ 0x89f ]= 168 ;
rlll[ 0x258 ]= 169 ;
rlll[ 0xbd5 ]= 170 ;
rlll[ 0x1e78 ]= 171 ;
rlll[ 0xddf ]= 172 ;
rlll[ 0x38e ]= 173 ;
rlll[ 0x47c ]= 174 ;
rlll[ 0xc39 ]= 175 ;
rlll[ 0x651 ]= 176 ;
rlll[ 0x950 ]= 177 ;
rlll[ 0x1daa ]= 178 ;
rlll[ 0x3f5 ]= 179 ;
rlll[ 0xca ]= 180 ;
rlll[ 0x1c18 ]= 181 ;
rlll[ 0x95d ]= 182 ;
rlll[ 0xbea ]= 183 ;
rlll[ 0x1c85 ]= 184 ;
rlll[ 0xa30 ]= 185 ;
rlll[ 0x1fa1 ]= 186 ;
rlll[ 0xec8 ]= 187 ;
rlll[ 0xfff ]= 188 ;
rlll[ 0x49e ]= 189 ;
rlll[ 0x37 ]= 190 ;
rlll[ 0x11e5 ]= 191 ;
rlll[ 0x1e8 ]= 192 ;
rlll[ 0x18c3 ]= 193 ;
rlll[ 0x1b09 ]= 194 ;
rlll[ 0x8c ]= 195 ;
rlll[ 0x1076 ]= 196 ;
rlll[ 0x544 ]= 197 ;
rlll[ 0x3a3 ]= 198 ;
rlll[ 0x1092 ]= 199 ;
rlll[ 0xba3 ]= 200 ;
rlll[ 0x1fb0 ]= 201 ;
rlll[ 0x1390 ]= 202 ;
rlll[ 0x1e04 ]= 203 ;
rlll[ 0x197a ]= 204 ;
rlll[ 0x148 ]= 205 ;
rlll[ 0xce6 ]= 206 ;
rlll[ 0x1fc1 ]= 207 ;
rlll[ 0x1749 ]= 208 ;
rlll[ 0x128d ]= 209 ;
rlll[ 0xa06 ]= 210 ;
rlll[ 0x1b62 ]= 211 ;
rlll[ 0x199e ]= 212 ;
rlll[ 0xf35 ]= 213 ;
rlll[ 0xe4 ]= 214 ;
rlll[ 0x294 ]= 215 ;
rlll[ 0x1ef7 ]= 216 ;
rlll[ 0x15f3 ]= 217 ;
rlll[ 0x1e53 ]= 218 ;
rlll[ 0xd84 ]= 219 ;
rlll[ 0x1c7e ]= 220 ;
rlll[ 0x1678 ]= 221 ;
rlll[ 0x5d8 ]= 222 ;
rlll[ 0x840 ]= 223 ;
rlll[ 0xef8 ]= 224 ;
rlll[ 0x1bab ]= 225 ;
rlll[ 0x7 ]= 226 ;
rlll[ 0x1851 ]= 227 ;
rlll[ 0x198 ]= 228 ;
rlll[ 0x1352 ]= 229 ;
rlll[ 0x49 ]= 230 ;
rlll[ 0x1fe8 ]= 231 ;
rlll[ 0x116b ]= 232 ;
rlll[ 0x1272 ]= 233 ;
rlll[ 0x158d ]= 234 ;
rlll[ 0x75f ]= 235 ;
rlll[ 0x89c ]= 236 ;
rlll[ 0x47b ]= 237 ;
rlll[ 0xaa8 ]= 238 ;
rlll[ 0xd4c ]= 239 ;
rlll[ 0x1355 ]= 240 ;
rlll[ 0x1d69 ]= 241 ;
rlll[ 0x113b ]= 242 ;
rlll[ 0x83f ]= 243 ;
rlll[ 0x58f ]= 244 ;
rlll[ 0x24f ]= 245 ;
rlll[ 0xeb2 ]= 246 ;
rlll[ 0xdd4 ]= 247 ;
rlll[ 0x1d00 ]= 248 ;
rlll[ 0x178e ]= 249 ;
rlll[ 0x188b ]= 250 ;
rlll[ 0x380 ]= 251 ;
rlll[ 0x172e ]= 252 ;
rlll[ 0x4fa ]= 253 ;
rlll[ 0x19a6 ]= 254 ;
rlll[ 0x115d ]= 255 ;

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

    bn_read_str(tmp, "10000000000000000000000", 24, 16);
    fp_exp(h, g, tmp);

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
}
