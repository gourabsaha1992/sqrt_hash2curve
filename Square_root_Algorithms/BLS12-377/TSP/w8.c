#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include <immintrin.h>
#include <relic/relic_core.h>
#include "relic/relic_md.h"


#define w       8
#define we      256      /* 2^w */
#define n       46
#define nw      6        /* ceil(46/8) */
#define leaf_w  8

//int countM = 0;
//int countS = 0;

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
 * Note: rll and fhl are built differently from the standard rll/fll tables.
 * rll is a lookup-by-value map (here implemented as a scanned array).
 * fhl[v] = h1^(-(v>>1)) built per the original Sage construction.
 * ========================================================================= */
void precomputation(fp_t g, fp_t h, fp_t h1,
                    fp_t gw[nw][we], fp_t rll[we], fp_t fhl[we], fp_t gpp[n])
{
    bn_t temp, one;
    bn_null(temp); bn_new(temp);
    bn_null(one);  bn_new(one);
    bn_set_dig(one, 1);

    /* gw[i][j] = g^(j * 2^(i*w)) */
    for (int i = 0; i < nw; i++) {
        for (int j = 0; j < we; j++) {
            bn_lsh(temp, one, i * w);
            bn_mul_dig(temp, temp, j);
            fp_exp(gw[i][j], g, temp);
        }
    }

    /* gpp[i] = g^(2^i) */
    fp_copy(gpp[0], g);
    for (int i = 1; i < n; i++)
        fp_sqr(gpp[i], gpp[i - 1]);

    /* rll[v] = h^v  (rll[0] = 1) */
    fp_set_dig(rll[0], 1);
    for (int v = 1; v < we; v++)
        fp_mul(rll[v], rll[v - 1], h);


    fp_t u_tmp, v_tmp;
    fp_null(u_tmp); fp_new(u_tmp);
    fp_null(v_tmp); fp_new(v_tmp);


    bn_lsh(temp, one, n);         
    bn_t exp2; bn_null(exp2); bn_new(exp2);
    bn_lsh(exp2, one, n - leaf_w - 1); 
    bn_sub(temp, temp, exp2);  
    fp_exp(u_tmp, g, temp);

    fp_set_dig(v_tmp, 1);
    fp_copy(fhl[0], v_tmp);
    for (int i = 1; i < we; i++) {
        fp_mul(v_tmp, v_tmp, u_tmp);
        fp_copy(fhl[i], v_tmp);
    }

    fp_free(u_tmp); fp_free(v_tmp);
    bn_free(exp2);
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


static void GPOW_i0_e22(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 22) - 1;
    uint64_t wm = we - 1;
    fp_copy(t, gw[0][e & wm]); e >>= w;
    fp_mul(t, t, gw[1][e & wm]); //countM++; 
    e >>= w;
    fp_mul(t, t, gw[2][e & wm]); //countM++;
}

/* GPOW_i12_e23: ri=4, row=1, e<<=4, adj=27, rows=4, 3 muls */
static void GPOW_i12_e23(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 23) - 1; e <<= 4;
    uint64_t wm = we - 1;
    fp_copy(t, gw[1][e & wm]); e >>= w;
    fp_mul(t, t, gw[2][e & wm]); //countM++; 
    e >>= w;
    fp_mul(t, t, gw[3][e & wm]); //countM++; 
    e >>= w;
    fp_mul(t, t, gw[4][e & wm]); //countM++;
}

/* GPOW_i22_e11: ri=6, row=2, e<<=6, adj=17, rows=3, 2 muls */
static void GPOW_i22_e11(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 11) - 1; e <<= 6;
    uint64_t wm = we - 1;
    fp_copy(t, gw[2][e & wm]); e >>= w;
    fp_mul(t, t, gw[3][e & wm]); //countM++; 
    e >>= w;
    fp_mul(t, t, gw[4][e & wm]); //countM++;
}

/* GPOW_i23_e11: ri=7, row=2, e<<=7, adj=18, rows=3, 2 muls */
static void GPOW_i23_e11(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 11) - 1; e <<= 7;
    uint64_t wm = we - 1;
    fp_copy(t, gw[2][e & wm]); e >>= w;
    fp_mul(t, t, gw[3][e & wm]); //countM++; 
    e >>= w;
    fp_mul(t, t, gw[4][e & wm]); //countM++;
}

/* GPOW_i29_e11: ri=5, row=3, e<<=5, adj=16, rows=2, 1 mul */
static void GPOW_i29_e11(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 11) - 1; e <<= 5;
    uint64_t wm = we - 1;
    fp_copy(t, gw[3][e & wm]); e >>= w;
    fp_mul(t, t, gw[4][e & wm]); //countM++;
}

/* GPOW_i33_e6: ri=1, row=4, e<<=1, adj=7, rows=1, 0 muls */
static void GPOW_i33_e6(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 6) - 1; e <<= 1;
    uint64_t wm = we - 1;
    fp_copy(t, gw[4][e & wm]);
}

/* GPOW_i35_e5: ri=3, row=4, e<<=3, adj=8, rows=1, 0 muls */
static void GPOW_i35_e5(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 5) - 1; e <<= 3;
    uint64_t wm = we - 1;
    fp_copy(t, gw[4][e & wm]);
}

/* =========================================================================
 * rll_lookup: find index v such that rll[v] == x; return v.
 * ========================================================================= */
static int rll_lookup(fp_t x, fp_t rll[we])
{
    for (int _ii = 0; _ii < we; _ii++)
        if (fp_cmp(rll[_ii], x) == RLC_EQ) return _ii;
    return 0;
}

/* =========================================================================
 * solve_dlp_pow2_flat + SQRT — fully inlined, recursion-free, stack-free.
 *
 * Follows solve_dlp_pow2_flat(u) in the Sage exactly.
 * out_d receives d_final (the sqrt correction factor).
 * Expected: M=27, S=51
 * ========================================================================= */
void solve_dlp_pow2_flat(fp_t u, fp_t out_d,
                         fp_t gw[nw][we],int rlll[65536], fp_t fhl[we],
                         fp_t gpp[n])
{
    uint64_t one_k = 1;

    fp_t h0_A,  hlp_A,  hlp1_A;
    fp_t h0_PA, hlp_PA, hlp1_PA, h1_PA, u2_PA;
    fp_t h0_PAH, h1_PAH, u2_PAH;
    fp_t f_PAL, f_PAL_sq, u2_PAL;
    fp_t h0_PAL;
    fp_t f_A, f_A_sq, h1_A;
    fp_t h0_X, h0_XH, h1_XH, u2_XH;
    fp_t f_X, f_X_sq, h1_X;
    fp_t h0_XL, f_XL, f_XL_sq, u2_XL;
    fp_t d_PA, d_XL, d_X;

    fp_null(h0_A);    fp_new(h0_A);
    fp_null(hlp_A);   fp_new(hlp_A);
    fp_null(hlp1_A);  fp_new(hlp1_A);
    fp_null(h0_PA);   fp_new(h0_PA);
    fp_null(hlp_PA);  fp_new(hlp_PA);
    fp_null(hlp1_PA); fp_new(hlp1_PA);
    fp_null(h1_PA);   fp_new(h1_PA);
    fp_null(u2_PA);   fp_new(u2_PA);
    fp_null(h0_PAH);  fp_new(h0_PAH);
    fp_null(h1_PAH);  fp_new(h1_PAH);
    fp_null(u2_PAH);  fp_new(u2_PAH);
    fp_null(f_PAL);   fp_new(f_PAL);
    fp_null(f_PAL_sq);fp_new(f_PAL_sq);
    fp_null(u2_PAL);  fp_new(u2_PAL);
    fp_null(h0_PAL);  fp_new(h0_PAL);
    fp_null(f_A);     fp_new(f_A);
    fp_null(f_A_sq);  fp_new(f_A_sq);
    fp_null(h1_A);    fp_new(h1_A);
    fp_null(h0_X);    fp_new(h0_X);
    fp_null(h0_XH);   fp_new(h0_XH);
    fp_null(h1_XH);   fp_new(h1_XH);
    fp_null(u2_XH);   fp_new(u2_XH);
    fp_null(f_X);     fp_new(f_X);
    fp_null(f_X_sq);  fp_new(f_X_sq);
    fp_null(h1_X);    fp_new(h1_X);
    fp_null(h0_XL);   fp_new(h0_XL);
    fp_null(f_XL);    fp_new(f_XL);
    fp_null(f_XL_sq); fp_new(f_XL_sq);
    fp_null(u2_XL);   fp_new(u2_XL);
    fp_null(d_PA);    fp_new(d_PA);
    fp_null(d_XL);    fp_new(d_XL);
    fp_null(d_X);     fp_new(d_X);


    fp_copy(h0_A, u);
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

    /* =========================================================================
     * PAH block (i=35): lb=11, a=5, b=6, do_hlp=False (b=6<=leaf_w=8)
     * Square 6; no hlp capture.
     * PAH.H leaf: i=41, lb=5 -> >> (leaf_w-lb)=3
     * GPOW(35,5) [0 M]
     * PAH.L leaf: i=40, lb=6 -> >> (leaf_w-lb)=2; fhl[e1<<2]
     * c_PA = c_PAH + (e_PAH << 5)   [11-bit]
     * ========================================================================= */
    fp_copy(h0_PAH, h0_PA);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_PAH, h0_PAH);
    //countS += 6;

    /* PAH.H leaf: i=41, lb=5, >> 3 */
    bn_t tmp_bn;
    int _tmp;
    dig_t d;
    fp_prime_back(tmp_bn, h0_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xffff;
    _tmp=rlll[d];
    uint64_t c_PAH = _tmp>>3;
    //(uint64_t)rll_lookup(h0_PAH, rll) >> 3;

    GPOW_i35_e5(h1_PAH, (one_k << 5) - c_PAH, gw);
    SELECT(h1_PAH, gpp[40], c_PAH == 0, h1_PAH);
    //countM++;
    fp_mul(u2_PAH, h0_PA, h1_PAH);

    /* PAH.L leaf: i=40, lb=6, >> 2 */
    fp_prime_back(tmp_bn, u2_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xffff;
    _tmp=rlll[d];
    int _e1_PAH   = _tmp>> 2;
    //rll_lookup(u2_PAH, rll) >> 2;
    /* d1_PAH = fhl[_e1_PAH << 2]  (unused in the PA HIGH path) */
    uint64_t e_PAH = ((uint64_t)_e1_PAH - 1) & 63;
    uint64_t c_PA  = c_PAH + (e_PAH << 5);   /* 11-bit */

    /* =========================================================================
     * PA correct (i=23, a=11):
     * h1_PA   = GPOW(23,11) [2M];  u2_PA = h0_A * h1_PA
     * hlp1_PA = GPOW(29,11) [1M];  hlp_PA *= hlp1_PA
     * ========================================================================= */
    GPOW_i23_e11(h1_PA,   (one_k << 11) - c_PA, gw);
    SELECT(h1_PA,   gpp[34], c_PA == 0, h1_PA);
    GPOW_i29_e11(hlp1_PA, (one_k << 11) - c_PA, gw);
    SELECT(hlp1_PA, gpp[40], c_PA == 0, hlp1_PA);
    fp_mul(hlp_PA, hlp_PA, hlp1_PA); //countM++;
    //countM++;
    fp_mul(u2_PA, h0_A, h1_PA);

    /* =========================================================================
     * PAL block (i=34): lb=12, a=6, b=6
     * helper=updated_hlp_PA -> else branch: h0_PAL=hlp_PA, no squaring.
     * PAL.H leaf: i=40, lb=6, >> 2
     * f = GPOW(i-1=33, 2^6-c_PAL, 6)  [0 M]
     * u2_PAL = f_PAL^2 * u2_PA   [1S + mul]
     * PAL.L leaf: i=40, lb=6, >> 2; fhl[e1_PAL<<2]->d1_PAL
     * d_PA = d1_PAL * f_PAL   [countM]
     * e1_PA = c_PAL + (e_PAL << 6)   [12-bit]
     * ========================================================================= */
    fp_copy(h0_PAL, hlp_PA);   /* updated hlp_PA consumed */

    /* PAL.H leaf: i=40, lb=6, >> 2 */
    fp_prime_back(tmp_bn, h0_PAL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xffff;
    _tmp=rlll[d];
    uint64_t c_PAL = _tmp>> 2;
    //(uint64_t)rll_lookup(h0_PAL, rll) >> 2;

    GPOW_i33_e6(f_PAL, (one_k << 6) - c_PAL, gw);
    SELECT(f_PAL, gpp[39], c_PAL == 0, f_PAL);
    //countM++;

    fp_sqr(f_PAL_sq, f_PAL); //countS++;
    fp_mul(u2_PAL, f_PAL_sq, u2_PA);

    /* PAL.L leaf: i=40, lb=6, >> 2 */
    fp_prime_back(tmp_bn, u2_PAL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xffff;
    _tmp=rlll[d];
    int _e1_PAL = _tmp >> 2;
    //rll_lookup(u2_PAL, rll) >> 2;
    fp_copy(d_PA, fhl[_e1_PAL << 2]);
    fp_mul(d_PA, d_PA, f_PAL); //countM++;

    uint64_t e_PAL  = ((uint64_t)_e1_PAL - 1) & 63;
    uint64_t e1_PA  = c_PAL + (e_PAL << 6);         /* 12-bit */

    /* Combine PA node -> e0_A (23-bit) */
    uint64_t e_PA_adj = (e1_PA - 1) & ((one_k << 12) - 1);
    uint64_t e0_A     = c_PA + (e_PA_adj << 11);    /* 23-bit */

    /* =========================================================================
     * TOP correction (i=0, ret_d=True):
     * f_A    = GPOW(0, ((2^23-e0_A+1)>>1), 22)   [2 M]  + countM
     * h1_A   = f_A^2 * u   [1S]
     * hlp1_A = GPOW(12, 2^23-e0_A, 23)            [3 M]
     * hlp_A *= hlp1_A    + countM + countM
     * ========================================================================= */
    GPOW_i0_e22(f_A,    ((one_k << 23) - e0_A + 1) >> 1, gw);
    SELECT(f_A,    gpp[22], e0_A == 0, f_A);
    //countM++;
    fp_sqr(f_A_sq, f_A); //countS++;
    fp_mul(h1_A, f_A_sq, u);

    GPOW_i12_e23(hlp1_A, (one_k << 23) - e0_A, gw);
    SELECT(hlp1_A, gpp[35], e0_A == 0, hlp1_A);
    fp_mul(hlp_A, hlp_A, hlp1_A); //countM++;
    //countM++;

    /* =========================================================================
     * EXT block (i=23): lb=23, a=11, b=12
     * helper=updated_hlp_A -> else branch: h0_X=hlp_A, no squaring.
     * ========================================================================= */
    fp_copy(h0_X, hlp_A);   /* updated hlp_A consumed */

    /* =========================================================================
     * XH block (i=35): lb=11, a=5, b=6, do_hlp=False (b=6<=leaf_w=8)
     * Square 6; no hlp capture.
     * XH.H leaf: i=41, lb=5, >> 3
     * GPOW(35,5) [0M];  u2_XH = h0_X * h1_XH  [countM]
     * XH.L leaf: i=40, lb=6, >> 2  (d1_XH computed but not used in XH path)
     * c_X = c_XHH + (e_XH << 5)   [11-bit]
     * ========================================================================= */
    fp_copy(h0_XH, h0_X);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_XH, h0_XH);
    //countS += 6;

    /* XH.H leaf: i=41, lb=5, >> 3 */
    fp_prime_back(tmp_bn, h0_XH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xffff;
    _tmp=rlll[d];
    uint64_t c_XHH = _tmp >> 3;
    //(uint64_t)rll_lookup(h0_XH, rll) >> 3;

    GPOW_i35_e5(h1_XH, (one_k << 5) - c_XHH, gw);
    SELECT(h1_XH, gpp[40], c_XHH == 0, h1_XH);
    //countM++;
    fp_mul(u2_XH, h0_X, h1_XH);

    /* XH.L leaf: i=40, lb=6, >> 2 */
    fp_prime_back(tmp_bn, u2_XH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xffff;
    _tmp=rlll[d];
    int _e1_XH = _tmp>> 2;
    //rll_lookup(u2_XH, rll) >> 2;
    /* d1_XH = fhl[_e1_XH << 2]  (not needed for c_X combine) */
    uint64_t e_XH = ((uint64_t)_e1_XH - 1) & 63;
    uint64_t c_X  = c_XHH + (e_XH << 5);   /* 11-bit */

    /* =========================================================================
     * EXT correct (i=23, ret_d=True):
     * f_X  = GPOW(i-1=22, 2^11-c_X, 11)   [2 M]  + countM
     * h1_X = f_X^2 * h1_A   [1S]
     * (no hlp update: helper was provided to EXT, do_hlp=False)
     * ========================================================================= */
    GPOW_i22_e11(f_X, (one_k << 11) - c_X, gw);
    SELECT(f_X, gpp[33], c_X == 0, f_X);
    //countM++;
    fp_sqr(f_X_sq, f_X); //countS++;
    fp_mul(h1_X, f_X_sq, h1_A);

    /* =========================================================================
     * XL block (i=34): lb=12, a=6, b=6, do_hlp=False (b=6<=leaf_w=8)
     * helper=None -> square 6; no hlp capture.
     * XL.H leaf: i=40, lb=6, >> 2
     * f = GPOW(i-1=33, 2^6-c_XL, 6)  [0 M]  + countM
     * u2_XL = f_XL^2 * h1_X   [1S]
     * XL.L leaf: i=40, lb=6, >> 2; fhl[e1_XL<<2]->d1_XL
     * d_XL = d1_XL * f_XL   [countM]
     * e1_X = c_XL + (e_XL << 6)   [12-bit]
     * ========================================================================= */
    fp_copy(h0_XL, h1_X);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_XL, h0_XL);
    //countS += 6;

    /* XL.H leaf: i=40, lb=6, >> 2 */
    fp_prime_back(tmp_bn, h0_XL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xffff;
    _tmp=rlll[d];
    uint64_t c_XL = _tmp>>2;
    //(uint64_t)rll_lookup(h0_XL, rll) >> 2;

    GPOW_i33_e6(f_XL, (one_k << 6) - c_XL, gw);
    SELECT(f_XL, gpp[39], c_XL == 0, f_XL);
    //countM++;

    fp_sqr(f_XL_sq, f_XL); //countS++;
    fp_mul(u2_XL, f_XL_sq, h1_X);

    /* XL.L leaf: i=40, lb=6, >> 2 */
    fp_prime_back(tmp_bn, u2_XL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xffff;
    _tmp=rlll[d];
    int _e1_XL = _tmp>>2;
    //rll_lookup(u2_XL, rll) >> 2;
    fp_copy(d_XL, fhl[_e1_XL << 2]);
    fp_mul(d_XL, d_XL, f_XL); //countM++;

//    uint64_t e_XL = ((uint64_t)_e1_XL - 1) & 63;
//    uint64_t e1_X = c_XL + (e_XL << 6);   /* 12-bit */

    /* =========================================================================
     * Combine EXT: d_X = d_XL * f_X;  e1_A (23-bit)
     * ========================================================================= */
    fp_mul(d_X, d_XL, f_X); //countM++;
//    uint64_t e_X_adj = (e1_X - 1) & ((one_k << 12) - 1);
//    uint64_t e1_A    = c_X + (e_X_adj << 11);   /* 23-bit */

    /* =========================================================================
     * Combine TOP: d_final = d_X * f_A;  e_final (46-bit)
     * ========================================================================= */
    fp_mul(out_d, d_X, f_A); //countM++;
//    uint64_t e1_A_adj = (e1_A - 1) & ((one_k << 23) - 1);
//    (void)(e0_A + (e1_A_adj << 23));   /* e_final unused */

//    fp_copy(out_d, d_final);

    /* ---- free all ---- */
    fp_free(h0_A);    fp_free(hlp_A);   fp_free(hlp1_A);
    fp_free(h0_PA);   fp_free(hlp_PA);  fp_free(hlp1_PA);
    fp_free(h1_PA);   fp_free(u2_PA);
    fp_free(h0_PAH);  fp_free(h1_PAH);  fp_free(u2_PAH);
    fp_free(f_PAL);   fp_free(f_PAL_sq);fp_free(u2_PAL);
    fp_free(h0_PAL);
    fp_free(f_A);     fp_free(f_A_sq);  fp_free(h1_A);
    fp_free(h0_X);    fp_free(h0_XH);   fp_free(h1_XH);  fp_free(u2_XH);
    fp_free(f_X);     fp_free(f_X_sq);  fp_free(h1_X);
    fp_free(h0_XL);   fp_free(f_XL);    fp_free(f_XL_sq);fp_free(u2_XL);
    fp_free(d_PA);    fp_free(d_XL);    fp_free(d_X); 
}

/* =========================================================================
 * SQRT
 * ========================================================================= */
void SQRT(fp_t x, fp_t y, bn_t e,
          fp_t gw[nw][we],int rlll[65536], fp_t fhl[we], fp_t gpp[n])
{
    fp_t u, v, w_, d;
    fp_null(u); fp_null(v); fp_null(w_); fp_null(d);
    RLC_TRY {
        fp_new(u); fp_new(v); fp_new(w_); fp_new(d);
        /* e_exp = (m-1)/2  where m = (p-1)/2^n */
        /* Caller must supply or compute; placeholder: fp_exp(v, x, e_exp) */
        fp_exp(v, x, e);

        fp_mul(w_, x, v); //countM++;   /* w_ = x * v */
        fp_mul(u,  w_, v); //countM++;  /* u  = w_ * v */
        solve_dlp_pow2_flat(u, d, gw,rlll,  fhl, gpp);
        fp_mul(y, w_, d); //countM++;   /* y  = w_ * d */
//        fp_print(x);
//        fp_sqr(y, y);
//        fp_print(y);
    }
    RLC_CATCH_ANY { RLC_THROW(ERR_CAUGHT); }
    RLC_FINALLY   { fp_free(u); fp_free(v); fp_free(w_); fp_free(d); }
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
    fp_t gw[nw][we], rll[we], fhl[we], gpp[n], g, z, b, h, hh, y;
    bn_t tmp, m,e;
    int rlll[65536]={0};
    
    rlll[ 0x1 ]= 0 ;
    rlll[ 0xa97 ]= 1 ;
    rlll[ 0xa2b6 ]= 2 ;
    rlll[ 0x74fa ]= 3 ;
    rlll[ 0xa8a6 ]= 4 ;
    rlll[ 0x8659 ]= 5 ;
    rlll[ 0xb34e ]= 6 ;
    rlll[ 0xca50 ]= 7 ;
    rlll[ 0xeb2 ]= 8 ;
    rlll[ 0xc9bc ]= 9 ;
    rlll[ 0xd815 ]= 10 ;
    rlll[ 0x5583 ]= 11 ;
    rlll[ 0xf359 ]= 12 ;
    rlll[ 0x62ba ]= 13 ;
    rlll[ 0xff25 ]= 14 ;
    rlll[ 0x371a ]= 15 ;
    rlll[ 0xb237 ]= 16 ;
    rlll[ 0x58ea ]= 17 ;
    rlll[ 0x1e32 ]= 18 ;
    rlll[ 0x72a1 ]= 19 ;
    rlll[ 0x9d76 ]= 20 ;
    rlll[ 0x3a37 ]= 21 ;
    rlll[ 0x155f ]= 22 ;
    rlll[ 0x5b77 ]= 23 ;
    rlll[ 0xd52c ]= 24 ;
    rlll[ 0xc3d8 ]= 25 ;
    rlll[ 0x6e28 ]= 26 ;
    rlll[ 0xacee ]= 27 ;
    rlll[ 0x1b67 ]= 28 ;
    rlll[ 0x275b ]= 29 ;
    rlll[ 0x588f ]= 30 ;
    rlll[ 0x6b3 ]= 31 ;
    rlll[ 0x85ba ]= 32 ;
    rlll[ 0xc525 ]= 33 ;
    rlll[ 0x799d ]= 34 ;
    rlll[ 0x8b67 ]= 35 ;
    rlll[ 0xc61f ]= 36 ;
    rlll[ 0x8c84 ]= 37 ;
    rlll[ 0x2977 ]= 38 ;
    rlll[ 0x72d7 ]= 39 ;
    rlll[ 0xc0ed ]= 40 ;
    rlll[ 0x9b65 ]= 41 ;
    rlll[ 0xac ]= 42 ;
    rlll[ 0xefbc ]= 43 ;
    rlll[ 0xb575 ]= 44 ;
    rlll[ 0xf3e3 ]= 45 ;
    rlll[ 0x4017 ]= 46 ;
    rlll[ 0x913 ]= 47 ;
    rlll[ 0xd143 ]= 48 ;
    rlll[ 0xc0fd ]= 49 ;
    rlll[ 0xe93 ]= 50 ;
    rlll[ 0x5c50 ]= 51 ;
    rlll[ 0x1b8b ]= 52 ;
    rlll[ 0x589d ]= 53 ;
    rlll[ 0x41ce ]= 54 ;
    rlll[ 0x4a88 ]= 55 ;
    rlll[ 0x5e66 ]= 56 ;
    rlll[ 0x5e8d ]= 57 ;
    rlll[ 0xfabc ]= 58 ;
    rlll[ 0xea44 ]= 59 ;
    rlll[ 0x560e ]= 60 ;
    rlll[ 0x377c ]= 61 ;
    rlll[ 0x16ce ]= 62 ;
    rlll[ 0x3c21 ]= 63 ;
    rlll[ 0x1c63 ]= 64 ;
    rlll[ 0xdca9 ]= 65 ;
    rlll[ 0xd6f8 ]= 66 ;
    rlll[ 0x5023 ]= 67 ;
    rlll[ 0xb97c ]= 68 ;
    rlll[ 0x32f1 ]= 69 ;
    rlll[ 0x1877 ]= 70 ;
    rlll[ 0x2e41 ]= 71 ;
    rlll[ 0x430 ]= 72 ;
    rlll[ 0x3600 ]= 73 ;
    rlll[ 0x2f44 ]= 74 ;
    rlll[ 0x8ad ]= 75 ;
    rlll[ 0x3f1d ]= 76 ;
    rlll[ 0xc67f ]= 77 ;
    rlll[ 0xecad ]= 78 ;
    rlll[ 0x93c5 ]= 79 ;
    rlll[ 0x12b1 ]= 80 ;
    rlll[ 0xb0cc ]= 81 ;
    rlll[ 0x60e1 ]= 82 ;
    rlll[ 0x5749 ]= 83 ;
    rlll[ 0xf9e4 ]= 84 ;
    rlll[ 0xdf60 ]= 85 ;
    rlll[ 0x12d1 ]= 86 ;
    rlll[ 0x9587 ]= 87 ;
    rlll[ 0xfdba ]= 88 ;
    rlll[ 0x6b18 ]= 89 ;
    rlll[ 0x34bd ]= 90 ;
    rlll[ 0xea31 ]= 91 ;
    rlll[ 0xbff4 ]= 92 ;
    rlll[ 0x575a ]= 93 ;
    rlll[ 0x666a ]= 94 ;
    rlll[ 0x1d46 ]= 95 ;
    rlll[ 0x80f0 ]= 96 ;
    rlll[ 0x3df1 ]= 97 ;
    rlll[ 0x32a2 ]= 98 ;
    rlll[ 0x3e5b ]= 99 ;
    rlll[ 0x9ad2 ]= 100 ;
    rlll[ 0x44ea ]= 101 ;
    rlll[ 0xc50e ]= 102 ;
    rlll[ 0x5f1c ]= 103 ;
    rlll[ 0xd1ab ]= 104 ;
    rlll[ 0xdffe ]= 105 ;
    rlll[ 0x6b6a ]= 106 ;
    rlll[ 0x398f ]= 107 ;
    rlll[ 0x3401 ]= 108 ;
    rlll[ 0xe49f ]= 109 ;
    rlll[ 0x4041 ]= 110 ;
    rlll[ 0xe21c ]= 111 ;
    rlll[ 0xf15d ]= 112 ;
    rlll[ 0xfa47 ]= 113 ;
    rlll[ 0x8f03 ]= 114 ;
    rlll[ 0x231f ]= 115 ;
    rlll[ 0x5e55 ]= 116 ;
    rlll[ 0x4a3e ]= 117 ;
    rlll[ 0xbb19 ]= 118 ;
    rlll[ 0x35e2 ]= 119 ;
    rlll[ 0x22e6 ]= 120 ;
    rlll[ 0x264b ]= 121 ;
    rlll[ 0x4fe9 ]= 122 ;
    rlll[ 0xc31d ]= 123 ;
    rlll[ 0x78e2 ]= 124 ;
    rlll[ 0x610c ]= 125 ;
    rlll[ 0x8155 ]= 126 ;
    rlll[ 0x6054 ]= 127 ;
    rlll[ 0x0 ]= 128 ;
    rlll[ 0xf56a ]= 129 ;
    rlll[ 0x5d4b ]= 130 ;
    rlll[ 0x8b07 ]= 131 ;
    rlll[ 0x575b ]= 132 ;
    rlll[ 0x79a8 ]= 133 ;
    rlll[ 0x4cb3 ]= 134 ;
    rlll[ 0x35b1 ]= 135 ;
    rlll[ 0xf14f ]= 136 ;
    rlll[ 0x3645 ]= 137 ;
    rlll[ 0x27ec ]= 138 ;
    rlll[ 0xaa7e ]= 139 ;
    rlll[ 0xca8 ]= 140 ;
    rlll[ 0x9d47 ]= 141 ;
    rlll[ 0xdc ]= 142 ;
    rlll[ 0xc8e7 ]= 143 ;
    rlll[ 0x4dca ]= 144 ;
    rlll[ 0xa717 ]= 145 ;
    rlll[ 0xe1cf ]= 146 ;
    rlll[ 0x8d60 ]= 147 ;
    rlll[ 0x628b ]= 148 ;
    rlll[ 0xc5ca ]= 149 ;
    rlll[ 0xeaa2 ]= 150 ;
    rlll[ 0xa48a ]= 151 ;
    rlll[ 0x2ad5 ]= 152 ;
    rlll[ 0x3c29 ]= 153 ;
    rlll[ 0x91d9 ]= 154 ;
    rlll[ 0x5313 ]= 155 ;
    rlll[ 0xe49a ]= 156 ;
    rlll[ 0xd8a6 ]= 157 ;
    rlll[ 0xa772 ]= 158 ;
    rlll[ 0xf94e ]= 159 ;
    rlll[ 0x7a47 ]= 160 ;
    rlll[ 0x3adc ]= 161 ;
    rlll[ 0x8664 ]= 162 ;
    rlll[ 0x749a ]= 163 ;
    rlll[ 0x39e2 ]= 164 ;
    rlll[ 0x737d ]= 165 ;
    rlll[ 0xd68a ]= 166 ;
    rlll[ 0x8d2a ]= 167 ;
    rlll[ 0x3f14 ]= 168 ;
    rlll[ 0x649c ]= 169 ;
    rlll[ 0xff55 ]= 170 ;
    rlll[ 0x1045 ]= 171 ;
    rlll[ 0x4a8c ]= 172 ;
    rlll[ 0xc1e ]= 173 ;
    rlll[ 0xbfea ]= 174 ;
    rlll[ 0xf6ee ]= 175 ;
    rlll[ 0x2ebe ]= 176 ;
    rlll[ 0x3f04 ]= 177 ;
    rlll[ 0xf16e ]= 178 ;
    rlll[ 0xa3b1 ]= 179 ;
    rlll[ 0xe476 ]= 180 ;
    rlll[ 0xa764 ]= 181 ;
    rlll[ 0xbe33 ]= 182 ;
    rlll[ 0xb579 ]= 183 ;
    rlll[ 0xa19b ]= 184 ;
    rlll[ 0xa174 ]= 185 ;
    rlll[ 0x545 ]= 186 ;
    rlll[ 0x15bd ]= 187 ;
    rlll[ 0xa9f3 ]= 188 ;
    rlll[ 0xc885 ]= 189 ;
    rlll[ 0xe933 ]= 190 ;
    rlll[ 0xc3e0 ]= 191 ;
    rlll[ 0xe39e ]= 192 ;
    rlll[ 0x2358 ]= 193 ;
    rlll[ 0x2909 ]= 194 ;
    rlll[ 0xafde ]= 195 ;
    rlll[ 0x4685 ]= 196 ;
    rlll[ 0xcd10 ]= 197 ;
    rlll[ 0xe78a ]= 198 ;
    rlll[ 0xd1c0 ]= 199 ;
    rlll[ 0xfbd1 ]= 200 ;
    rlll[ 0xca01 ]= 201 ;
    rlll[ 0xd0bd ]= 202 ;
    rlll[ 0xf754 ]= 203 ;
    rlll[ 0xc0e4 ]= 204 ;
    rlll[ 0x3982 ]= 205 ;
    rlll[ 0x1354 ]= 206 ;
    rlll[ 0x6c3c ]= 207 ;
    rlll[ 0xed50 ]= 208 ;
    rlll[ 0x4f35 ]= 209 ;
    rlll[ 0x9f20 ]= 210 ;
    rlll[ 0xa8b8 ]= 211 ;
    rlll[ 0x61d ]= 212 ;
    rlll[ 0x20a1 ]= 213 ;
    rlll[ 0xed30 ]= 214 ;
    rlll[ 0x6a7a ]= 215 ;
    rlll[ 0x247 ]= 216 ;
    rlll[ 0x94e9 ]= 217 ;
    rlll[ 0xcb44 ]= 218 ;
    rlll[ 0x15d0 ]= 219 ;
    rlll[ 0x400d ]= 220 ;
    rlll[ 0xa8a7 ]= 221 ;
    rlll[ 0x9997 ]= 222 ;
    rlll[ 0xe2bb ]= 223 ;
    rlll[ 0x7f11 ]= 224 ;
    rlll[ 0xc210 ]= 225 ;
    rlll[ 0xcd5f ]= 226 ;
    rlll[ 0xc1a6 ]= 227 ;
    rlll[ 0x652f ]= 228 ;
    rlll[ 0xbb17 ]= 229 ;
    rlll[ 0x3af3 ]= 230 ;
    rlll[ 0xa0e5 ]= 231 ;
    rlll[ 0x2e56 ]= 232 ;
    rlll[ 0x2003 ]= 233 ;
    rlll[ 0x9497 ]= 234 ;
    rlll[ 0xc672 ]= 235 ;
    rlll[ 0xcc00 ]= 236 ;
    rlll[ 0x1b62 ]= 237 ;
    rlll[ 0xbfc0 ]= 238 ;
    rlll[ 0x1de5 ]= 239 ;
    rlll[ 0xea4 ]= 240 ;
    rlll[ 0x5ba ]= 241 ;
    rlll[ 0x70fe ]= 242 ;
    rlll[ 0xdce2 ]= 243 ;
    rlll[ 0xa1ac ]= 244 ;
    rlll[ 0xb5c3 ]= 245 ;
    rlll[ 0x44e8 ]= 246 ;
    rlll[ 0xca1f ]= 247 ;
    rlll[ 0xdd1b ]= 248 ;
    rlll[ 0xd9b6 ]= 249 ;
    rlll[ 0xb018 ]= 250 ;
    rlll[ 0x3ce4 ]= 251 ;
    rlll[ 0x871f ]= 252 ;
    rlll[ 0x9ef5 ]= 253 ;
    rlll[ 0x7eac ]= 254 ;
    rlll[ 0x9fad ]= 255 ;


    for (i = 0; i < nw; i++)
        for (j = 0; j < we; j++) { fp_null(gw[i][j]); fp_new(gw[i][j]); }
    for (i = 0; i < we; i++) { fp_null(rll[i]); fp_new(rll[i]); }
    for (i = 0; i < we; i++) { fp_null(fhl[i]); fp_new(fhl[i]); }
    for (i = 0; i < n;  i++) { fp_null(gpp[i]); fp_new(gpp[i]); }
    fp_null(b);  fp_new(b);  fp_null(y);  fp_new(y);
    fp_null(g);  fp_new(g);  fp_null(z);  fp_new(z);
    fp_null(h);  fp_new(h);  fp_null(hh); fp_new(hh);
    bn_null(tmp); bn_new(tmp);
    bn_null(m);   bn_new(m); bn_new(e);

    fp_rand(b);
    while (fp_is_sqr(b) != 1) fp_rand(b);

    /* z = 11 */
    bn_read_str(tmp,"5",1,16);
    fp_prime_conv(z,tmp);
    
    bn_read_str(m,"6b8e9185f1443ab18ec1701b28524ec688b67cc03d44e3c7bcd88bee82520005c2d7510c00000021423",96,16);
    fp_exp(g,z,m);
    
    bn_read_str(e,"35c748c2f8a21d58c760b80d94292763445b3e601ea271e3de6c45f741290002e16ba88600000010a11",83,16);
    
    //h=g^(2^(n-w))
    bn_t a1;
    bn_null(a1);
    bn_new(a1);
    bn_read_str(a1,"4000000000",11,16);
    fp_exp(h,g,a1);

//    fp_print(h);
    fp_srt(hh, h);
    fp_inv(hh, hh);
//    fp_print(hh);

    precomputation(g, h, hh, gw, rll, fhl, gpp);

    MEASURE(SQRT(b, y, e, gw,rlll,  fhl, gpp);)

    printf("RDTSC_clk_min=%f\n",    RDTSC_clk_min);
    printf("RDTSC_clk_median=%f\n", RDTSC_clk_median);
    printf("RDTSC_clk_max=%f\n",    RDTSC_clk_max);
    //printf("mult_count=%d\n",  countM);
    //printf("sqr_count=%d\n",   countS);

    for (i = 0; i < nw; i++)
        for (j = 0; j < we; j++) fp_free(gw[i][j]);
    for (i = 0; i < we; i++) fp_free(rll[i]);
    for (i = 0; i < we; i++) fp_free(fhl[i]);
    for (i = 0; i < n;  i++) fp_free(gpp[i]);
    fp_free(b); fp_free(y); fp_free(g); fp_free(z); fp_free(h); fp_free(hh);
    bn_free(tmp); bn_free(m);
    
    return 0;
}
