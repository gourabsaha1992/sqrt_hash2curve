#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include <immintrin.h>
#include <relic/relic_core.h>
#include "relic/relic_md.h"



#define w       6
#define we      64       /* 2^w */
#define n       46
#define nw      8        /* ceil(46/6) */
#define leaf_w  6

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
    e >>= w;
    fp_mul(t, t, gw[3][e & wm]); //countM++;
}


static void GPOW_i12_e23(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 23) - 1;
    uint64_t wm = we - 1;
    fp_copy(t, gw[2][e & wm]); e >>= w;
    fp_mul(t, t, gw[3][e & wm]); //countM++; 
    e >>= w;
    fp_mul(t, t, gw[4][e & wm]); //countM++; 
    e >>= w;
    fp_mul(t, t, gw[5][e & wm]); //countM++;
}


static void GPOW_i22_e11(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 11) - 1; e <<= 4;
    uint64_t wm = we - 1;
    fp_copy(t, gw[3][e & wm]); e >>= w;
    fp_mul(t, t, gw[4][e & wm]); //countM++; 
    e >>= w;
    fp_mul(t, t, gw[5][e & wm]); //countM++;
}

/* GPOW_i23_e11: ri=5, row=3, e<<=5, adj=16, rows=3, 2 muls */
static void GPOW_i23_e11(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 11) - 1; e <<= 5;
    uint64_t wm = we - 1;
    fp_copy(t, gw[3][e & wm]); e >>= w;
    fp_mul(t, t, gw[4][e & wm]); //countM++; 
    e >>= w;
    fp_mul(t, t, gw[5][e & wm]); //countM++;
}

/* GPOW_i29_e11: ri=5, row=4, e<<=5, adj=16, rows=3, 2 muls */
static void GPOW_i29_e11(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 11) - 1; e <<= 5;
    uint64_t wm = we - 1;
    fp_copy(t, gw[4][e & wm]); e >>= w;
    fp_mul(t, t, gw[5][e & wm]); //countM++; 
    e >>= w;
    fp_mul(t, t, gw[6][e & wm]); //countM++;
}

/* GPOW_i33_e6: ri=3, row=5, e<<=3, adj=9, rows=2, 1 mul */
static void GPOW_i33_e6(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 6) - 1; e <<= 3;
    uint64_t wm = we - 1;
    fp_copy(t, gw[5][e & wm]); e >>= w;
    fp_mul(t, t, gw[6][e & wm]); //countM++;
}

/* GPOW_i35_e5: ri=5, row=5, e<<=5, adj=10, rows=2, 1 mul */
static void GPOW_i35_e5(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 5) - 1; e <<= 5;
    uint64_t wm = we - 1;
    fp_copy(t, gw[5][e & wm]); e >>= w;
    fp_mul(t, t, gw[6][e & wm]); //countM++;
}


static int rll_lookup(fp_t x, fp_t rll[we])
{
    for (int _ii = 0; _ii < we; _ii++)
        if (fp_cmp(rll[_ii], x) == RLC_EQ) return _ii;
    return 0;
}


void solve_dlp_pow2_flat(fp_t u, fp_t out_d,
                         fp_t gw[nw][we],int rlll[4096], fp_t rll[we], fp_t fll[we],
                         fp_t gpp[n])
{
    uint64_t one_k = 1;

    fp_t h0_A,   hlp_A,   hlp1_A;
    fp_t h0_PA,  hlp_PA,  hlp1_PA, h1_PA, u2_PA;
    fp_t h0_PAH, h1_PAH,  u2_PAH;
    fp_t h0_PAL, f_PAL,   f_PAL_sq, u2_PAL;
    fp_t f_A,    f_A_sq,  h1_A;
    fp_t h0_X,   h0_XH,   h1_XH,   u2_XH;
    fp_t f_X,    f_X_sq,  h1_X;
    fp_t h0_XL,  f_XL,    f_XL_sq, u2_XL;
    fp_t d_PA,   d_XL,    d_X,     d_final;

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
    fp_null(h0_PAL);  fp_new(h0_PAL);
    fp_null(f_PAL);   fp_new(f_PAL);
    fp_null(f_PAL_sq);fp_new(f_PAL_sq);
    fp_null(u2_PAL);  fp_new(u2_PAL);
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
    fp_null(d_final); fp_new(d_final);

    
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

    
    fp_copy(h0_PAH, h0_PA);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_PAH, h0_PAH);
   // countS += 6;
    
    bn_t tmp_bn;
    int _tmp;
    dig_t d;
    /* PAH.H leaf: i=41, lb=5, >> 1 */
    fp_prime_back(tmp_bn, h0_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xfff;
    _tmp=rlll[d];
    uint64_t c_PAH = _tmp>>1;
    

    GPOW_i35_e5(h1_PAH, (one_k << 5) - c_PAH, gw);
    SELECT(h1_PAH, gpp[40], c_PAH == 0, h1_PAH);
    //countM++;
    fp_mul(u2_PAH, h0_PA, h1_PAH);

    
    fp_prime_back(tmp_bn, u2_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xfff;
    _tmp=rlll[d];
    int _e1_PAH = _tmp;
    
    uint64_t e_PAH = ((uint64_t)_e1_PAH - 1) & 63;
    uint64_t c_PA  = c_PAH + (e_PAH << 5);          /* 11-bit */

    
    GPOW_i23_e11(h1_PA,   (one_k << 11) - c_PA, gw);
    SELECT(h1_PA,   gpp[34], c_PA == 0, h1_PA);
    GPOW_i29_e11(hlp1_PA, (one_k << 11) - c_PA, gw);
    SELECT(hlp1_PA, gpp[40], c_PA == 0, hlp1_PA);
    fp_mul(hlp_PA, hlp_PA, hlp1_PA); //countM++;
    //countM++;
    fp_mul(u2_PA, h0_A, h1_PA);

    fp_copy(h0_PAL, hlp_PA);   /* updated hlp_PA consumed */

    /* PAL.H leaf: i=40, lb=6, >> 0 */
    fp_prime_back(tmp_bn, h0_PAL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xfff;
    _tmp=rlll[d];
    uint64_t c_PAL = _tmp;

    GPOW_i33_e6(f_PAL, (one_k << 6) - c_PAL, gw);
    SELECT(f_PAL, gpp[39], c_PAL == 0, f_PAL);
    //countM++;

    fp_sqr(f_PAL_sq, f_PAL); //countS++;
    fp_mul(u2_PAL, f_PAL_sq, u2_PA);
    fp_prime_back(tmp_bn, u2_PAL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xfff;
    _tmp=rlll[d];
    int _e1_PAL = _tmp;

    fp_copy(d_PA, fll[_e1_PAL]);          
    fp_mul(d_PA, d_PA, f_PAL); //countM++;

    uint64_t e_PAL  = ((uint64_t)_e1_PAL - 1) & 63;
    uint64_t e1_PA  = c_PAL + (e_PAL << 6);           /* 12-bit */

    uint64_t e_PA_adj = (e1_PA - 1) & ((one_k << 12) - 1);
    uint64_t e0_A     = c_PA + (e_PA_adj << 11);      /* 23-bit */

    GPOW_i0_e22(f_A,    ((one_k << 23) - e0_A + 1) >> 1, gw);
    SELECT(f_A,    gpp[22], e0_A == 0, f_A);
    //countM++;
    fp_sqr(f_A_sq, f_A); //countS++;
    fp_mul(h1_A, f_A_sq, u);

    GPOW_i12_e23(hlp1_A, (one_k << 23) - e0_A, gw);
    SELECT(hlp1_A, gpp[35], e0_A == 0, hlp1_A);
    fp_mul(hlp_A, hlp_A, hlp1_A); //countM++;
    //countM++;

    
    fp_copy(h0_X, hlp_A);   

    
    fp_copy(h0_XH, h0_X);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_XH, h0_XH);
    //countS += 6;

    
    fp_prime_back(tmp_bn, h0_XH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xfff;
    _tmp=rlll[d];
    uint64_t c_XHH =_tmp>>1; 

    GPOW_i35_e5(h1_XH, (one_k << 5) - c_XHH, gw);
    SELECT(h1_XH, gpp[40], c_XHH == 0, h1_XH);
    //countM++;
    fp_mul(u2_XH, h0_X, h1_XH);

    /* XH.L leaf: i=40, lb=6, >> 0 */
    fp_prime_back(tmp_bn, u2_XH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xfff;
    _tmp=rlll[d];
    int _e1_XH = _tmp;
    
    uint64_t e_XH = ((uint64_t)_e1_XH - 1) & 63;
    uint64_t c_X  = c_XHH + (e_XH << 5);   /* 11-bit */

    
    GPOW_i22_e11(f_X, (one_k << 11) - c_X, gw);
    SELECT(f_X, gpp[33], c_X == 0, f_X);
    //countM++;
    fp_sqr(f_X_sq, f_X); //countS++;
    fp_mul(h1_X, f_X_sq, h1_A);

    
    fp_copy(h0_XL, h1_X);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_XL, h0_XL);
    //countS += 6;

    /* XL.H leaf: i=40, lb=6, >> 0 */
    fp_prime_back(tmp_bn, h0_XL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xfff;
    _tmp=rlll[d];
    uint64_t c_XL = _tmp;
    //(uint64_t)rll_lookup(h0_XL, rll);   /* >> 0 */

    GPOW_i33_e6(f_XL, (one_k << 6) - c_XL, gw);
    SELECT(f_XL, gpp[39], c_XL == 0, f_XL);
    //countM++;

    fp_sqr(f_XL_sq, f_XL); //countS++;
    fp_mul(u2_XL, f_XL_sq, h1_X);

    /* XL.L leaf: i=40, lb=6, >> 0 */
    fp_prime_back(tmp_bn, u2_XL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xfff;
    _tmp=rlll[d];
    int _e1_XL = _tmp;
    
    fp_copy(d_XL, fll[_e1_XL]);            /* fhl[e1_XL << 0] */
    fp_mul(d_XL, d_XL, f_XL); //countM++;


    fp_mul(d_X, d_XL, f_X); //countM++;

    fp_mul(d_final, d_X, f_A); //countM++;


    fp_copy(out_d, d_final);

    /* ---- free all ---- */
    fp_free(h0_A);    fp_free(hlp_A);   fp_free(hlp1_A);
    fp_free(h0_PA);   fp_free(hlp_PA);  fp_free(hlp1_PA);
    fp_free(h1_PA);   fp_free(u2_PA);
    fp_free(h0_PAH);  fp_free(h1_PAH);  fp_free(u2_PAH);
    fp_free(h0_PAL);  fp_free(f_PAL);   fp_free(f_PAL_sq); fp_free(u2_PAL);
    fp_free(f_A);     fp_free(f_A_sq);  fp_free(h1_A);
    fp_free(h0_X);    fp_free(h0_XH);   fp_free(h1_XH);   fp_free(u2_XH);
    fp_free(f_X);     fp_free(f_X_sq);  fp_free(h1_X);
    fp_free(h0_XL);   fp_free(f_XL);    fp_free(f_XL_sq); fp_free(u2_XL);
    fp_free(d_PA);    fp_free(d_XL);    fp_free(d_X);     fp_free(d_final);
}

/* =========================================================================
 * SQRT
 * ========================================================================= */
void SQRT(fp_t x, fp_t y,bn_t e,
          fp_t gw[nw][we],int rlll[4096], fp_t rll[we], fp_t fll[we], fp_t gpp[n])
{
    fp_t u, v, w_, d;
    fp_null(u); fp_null(v); fp_null(w_); fp_null(d);
    RLC_TRY {
        fp_new(u); fp_new(v); fp_new(w_); fp_new(d);
        
        fp_exp(v, x, e);
        
        fp_mul(w_, x, v);  //countM++;
        
        fp_mul(u,  w_, v); //countM++;
        
        solve_dlp_pow2_flat(u, d, gw,rlll, rll, fll, gpp);
        fp_mul(y, w_, d);  //countM++;
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
    fp_t gw[nw][we], rll[we], fll[we], gpp[n], g, z, b, h, hh, y;
    bn_t tmp, m,e;
    
    int rlll[4096]={0};
    rlll[ 0x1 ]= 0 ;
  rlll[ 0x8a6 ]= 1 ;
  rlll[ 0xeb2 ]= 2 ;
  rlll[ 0x359 ]= 3 ;
  rlll[ 0x237 ]= 4 ;
  rlll[ 0xd76 ]= 5 ;
  rlll[ 0x52c ]= 6 ;
  rlll[ 0xb67 ]= 7 ;
  rlll[ 0x5ba ]= 8 ;
  rlll[ 0x61f ]= 9 ;
  rlll[ 0xed ]= 10 ;
  rlll[ 0x575 ]= 11 ;
  rlll[ 0x143 ]= 12 ;
  rlll[ 0xb8b ]= 13 ;
  rlll[ 0xe66 ]= 14 ;
  rlll[ 0x60e ]= 15 ;
  rlll[ 0xc63 ]= 16 ;
  rlll[ 0x97c ]= 17 ;
  rlll[ 0x430 ]= 18 ;
  rlll[ 0xf1d ]= 19 ;
  rlll[ 0x2b1 ]= 20 ;
  rlll[ 0x9e4 ]= 21 ;
  rlll[ 0xdba ]= 22 ;
  rlll[ 0xff4 ]= 23 ;
  rlll[ 0xf0 ]= 24 ;
  rlll[ 0xad2 ]= 25 ;
  rlll[ 0x1ab ]= 26 ;
  rlll[ 0x401 ]= 27 ;
  rlll[ 0x15d ]= 28 ;
  rlll[ 0xe55 ]= 29 ;
  rlll[ 0x2e6 ]= 30 ;
  rlll[ 0x8e2 ]= 31 ;
  rlll[ 0x0 ]= 32 ;
  rlll[ 0x75b ]= 33 ;
  rlll[ 0x14f ]= 34 ;
  rlll[ 0xca8 ]= 35 ;
  rlll[ 0xdca ]= 36 ;
  rlll[ 0x28b ]= 37 ;
  rlll[ 0xad5 ]= 38 ;
  rlll[ 0x49a ]= 39 ;
  rlll[ 0xa47 ]= 40 ;
  rlll[ 0x9e2 ]= 41 ;
  rlll[ 0xf14 ]= 42 ;
  rlll[ 0xa8c ]= 43 ;
  rlll[ 0xebe ]= 44 ;
  rlll[ 0x476 ]= 45 ;
  rlll[ 0x19b ]= 46 ;
  rlll[ 0x9f3 ]= 47 ;
  rlll[ 0x39e ]= 48 ;
  rlll[ 0x685 ]= 49 ;
  rlll[ 0xbd1 ]= 50 ;
  rlll[ 0xe4 ]= 51 ;
  rlll[ 0xd50 ]= 52 ;
  rlll[ 0x61d ]= 53 ;
  rlll[ 0x247 ]= 54 ;
  rlll[ 0xd ]= 55 ;
  rlll[ 0xf11 ]= 56 ;
  rlll[ 0x52f ]= 57 ;
  rlll[ 0xe56 ]= 58 ;
  rlll[ 0xc00 ]= 59 ;
  rlll[ 0xea4 ]= 60 ;
  rlll[ 0x1ac ]= 61 ;
  rlll[ 0xd1b ]= 62 ;
  rlll[ 0x71f ]= 63 ;


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

    bn_read_str(tmp,"5",1,16);
    fp_prime_conv(z,tmp);
    
    bn_read_str(m,"6b8e9185f1443ab18ec1701b28524ec688b67cc03d44e3c7bcd88bee82520005c2d7510c00000021423",96,16);
    fp_exp(g,z,m);
    
    bn_read_str(e,"35c748c2f8a21d58c760b80d94292763445b3e601ea271e3de6c45f741290002e16ba88600000010a11",83,16);
    
    //h=g^(2^(n-w))
    bn_t a1;
    bn_null(a1);
    bn_new(a1);
    bn_read_str(a1,"10000000000",11,16);
    fp_exp(h,g,a1);

    
    fp_srt(hh, h);
    fp_inv(hh, hh);

    precomputation(g, h, hh, gw, rll, fll, gpp);

    MEASURE(SQRT(b, y,e, gw, rlll, rll, fll, gpp);)

    printf("RDTSC_clk_min=%f\n",    RDTSC_clk_min);
    printf("RDTSC_clk_median=%f\n", RDTSC_clk_median);
    printf("RDTSC_clk_max=%f\n",    RDTSC_clk_max);
    //printf("mult_count=%d\n",  countM);
    //printf("sqr_count=%d\n",   countS);

    for (i = 0; i < nw; i++)
        for (j = 0; j < we; j++) fp_free(gw[i][j]);
    for (i = 0; i < we; i++) fp_free(rll[i]);
    for (i = 0; i < we; i++) fp_free(fll[i]);
    for (i = 0; i < n;  i++) fp_free(gpp[i]);
    fp_free(b); fp_free(y); fp_free(g); fp_free(z); fp_free(h); fp_free(h1);
    bn_free(tmp); bn_free(m);
    
    return 0;
}
