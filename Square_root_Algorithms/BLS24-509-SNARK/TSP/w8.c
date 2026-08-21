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
#define we  256      /* 2^w */
#define n   37
#define nw  5        /* ceil(37/8) */

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
}


static void GPOW_i10_e18(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 18) - 1; e <<= 2;
    uint64_t wm = we - 1;
    fp_copy(t, gw[1][e & wm]); e >>= w;
    fp_mul(t, t, gw[2][e & wm]); //countM++; 
e >>= w;
    fp_mul(t, t, gw[3][e & wm]); //countM++;
}


static void GPOW_i17_e9(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 9) - 1; e <<= 1;
    uint64_t wm = we - 1;
    fp_copy(t, gw[2][e & wm]); e >>= w;
    fp_mul(t, t, gw[3][e & wm]); //countM++;
}


static void GPOW_i19_e9(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 9) - 1; e <<= 3;
    uint64_t wm = we - 1;
    fp_copy(t, gw[2][e & wm]); e >>= w;
    fp_mul(t, t, gw[3][e & wm]); //countM++;
}


static void GPOW_i24_e9(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 9) - 1;
    uint64_t wm = we - 1;
    fp_copy(t, gw[3][e & wm]); e >>= w;
    fp_mul(t, t, gw[4][e & wm]); //countM++;
}


static void GPOW_i26_e5(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 5) - 1; e <<= 2;
    uint64_t wm = we - 1;
    fp_copy(t, gw[3][e & wm]);
}


static void GPOW_i28_e4(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 4) - 1; e <<= 4;
    uint64_t wm = we - 1;
    fp_copy(t, gw[3][e & wm]);
}


void DLPpow2ext(fp_t u_A, fp_t out_t,
                fp_t gw[nw][we], int rlll[32768], fp_t rll[we], fp_t fll[we], fp_t gpp[n])
{
    uint64_t one_k = 1;
    int _ii, _tmp;

    fp_t h0_A, hlp_A, hlp1_A;
    fp_t h0_PA, hlp_PA, hlp1_PA, h1_PA, u2_PA;
    fp_t h0_PAH, h1_PAH, u2_PAH;
    fp_t h0_PAL, h1_PAL, u2_PAL;
    fp_t f_A, f_A_sq;
    fp_t u_X0, h0_X0, h0_X0H, h1_X0H, u2_X0H;
    fp_t f_X0, f_X0_sq;
    fp_t u_X1, h0_X1;
    fp_t f_X1, f_X1_sq;
    fp_t u_X2;
    fp_t t_X1, t_X0, t_A;

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
    fp_null(h1_PAL);  fp_new(h1_PAL);
    fp_null(u2_PAL);  fp_new(u2_PAL);
    fp_null(f_A);     fp_new(f_A);
    fp_null(f_A_sq);  fp_new(f_A_sq);
    fp_null(u_X0);    fp_new(u_X0);
    fp_null(h0_X0);   fp_new(h0_X0);
    fp_null(h0_X0H);  fp_new(h0_X0H);
    fp_null(h1_X0H);  fp_new(h1_X0H);
    fp_null(u2_X0H);  fp_new(u2_X0H);
    fp_null(f_X0);    fp_new(f_X0);
    fp_null(f_X0_sq); fp_new(f_X0_sq);
    fp_null(u_X1);    fp_new(u_X1);
    fp_null(h0_X1);   fp_new(h0_X1);
    fp_null(f_X1);    fp_new(f_X1);
    fp_null(f_X1_sq); fp_new(f_X1_sq);
    fp_null(u_X2);    fp_new(u_X2);
    fp_null(t_X1);    fp_new(t_X1);
    fp_null(t_X0);    fp_new(t_X0);
    fp_null(t_A);     fp_new(t_A);

    
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
    for (int _j = 0; _j < 5; _j++) fp_sqr(h0_PAH, h0_PAH);
    //countS += 5;

    
	bn_t tmp_bn;
	bn_null(tmp_bn);bn_new(tmp_bn);
	dig_t d;
    fp_prime_back(tmp_bn, h0_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7fff;
    _tmp=rlll[d];

    uint64_t c_PAH = (uint64_t)_tmp >> 4;

    GPOW_i28_e4(h1_PAH, (one_k << 4) - c_PAH, gw);

    SELECT(h1_PAH, gpp[32], c_PAH == 0, h1_PAH);
    //countM++;
    fp_mul(u2_PAH, h0_PA, h1_PAH);   

   

    fp_prime_back(tmp_bn, u2_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7fff;
    _tmp=rlll[d];

    uint64_t d_PAH = (uint64_t)_tmp >> 3;
    uint64_t c_PA  = c_PAH + ((d_PAH - 1) % 32) * 16; 

    GPOW_i19_e9(h1_PA,   (one_k << 9) - c_PA, gw);
    SELECT(h1_PA,   gpp[28], c_PA == 0, h1_PA);
    GPOW_i24_e9(hlp1_PA, (one_k << 9) - c_PA, gw);
    SELECT(hlp1_PA, gpp[33], c_PA == 0, hlp1_PA);
    fp_mul(hlp_PA, hlp_PA, hlp1_PA); //countM++;
    //countM++;
    fp_mul(u2_PA, h0_A, h1_PA);

    
    fp_copy(h0_PAL, hlp_PA);   

    /* leaf-H: i=33, lb=4 -> >> 4 */
    fp_prime_back(tmp_bn, h0_PAL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7fff;
    _tmp=rlll[d];

    uint64_t c_PAL = (uint64_t)_tmp >> 4;

    GPOW_i28_e4(h1_PAL, (one_k << 4) - c_PAL, gw);
    SELECT(h1_PAL, gpp[32], c_PAL == 0, h1_PAL);
    //countM++;
    fp_mul(u2_PAL, u2_PA, h1_PAL);   

    
    fp_prime_back(tmp_bn, u2_PAL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7fff;
    _tmp=rlll[d];

    uint64_t d_PAL = (uint64_t)_tmp >> 3;
    uint64_t d_PA  = c_PAL + ((d_PAL - 1) % 32) * 16;   

    
    uint64_t c_A = c_PA + ((d_PA - 1) % 512) * 512;  

    
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

    /* sq 5 */
    fp_copy(h0_X0H, h0_X0);
    for (int _j = 0; _j < 5; _j++) fp_sqr(h0_X0H, h0_X0H);
    //countS += 5;

   
    fp_prime_back(tmp_bn, h0_X0H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7fff;
    _tmp=rlll[d];

    uint64_t c_X0H = (uint64_t)_tmp >> 4;

    GPOW_i28_e4(h1_X0H, (one_k << 4) - c_X0H, gw);
    SELECT(h1_X0H, gpp[32], c_X0H == 0, h1_X0H);
    //countM++;
    fp_mul(u2_X0H, h0_X0, h1_X0H);  

   
    fp_prime_back(tmp_bn, u2_X0H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7fff;
    _tmp=rlll[d];

    uint64_t d_X0H = (uint64_t)_tmp >> 3;
    uint64_t c_X0  = c_X0H + ((d_X0H - 1) % 32) * 16;

    
    GPOW_i17_e9(f_X0, (one_k << 9) - c_X0, gw);
    SELECT(f_X0, gpp[26], c_X0 == 0, f_X0);
    //countM++;
    //countS++;

    
    fp_sqr(f_X0_sq, f_X0);
    fp_mul(u_X1, u_X0, f_X0_sq); //countM++;
    fp_copy(h0_X1, u_X1);
    for (int _j = 0; _j < 5; _j++) fp_sqr(h0_X1, h0_X1);
    //countS += 5;

    
    fp_prime_back(tmp_bn, h0_X1);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7fff;
    _tmp=rlll[d];

    uint64_t c_X1 = (uint64_t)_tmp >> 3;

    GPOW_i26_e5(f_X1, (one_k << 5) - c_X1, gw);
    SELECT(f_X1, gpp[31], c_X1 == 0, f_X1);
    //countM++;
    //countS++;   
    fp_sqr(f_X1_sq, f_X1);
    fp_mul(u_X2, u_X1, f_X1_sq);
    fp_prime_back(tmp_bn, u_X2);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7fff;
    _tmp=rlll[d];

    uint64_t c1_X2 = (uint64_t)_tmp >> 3;
    fp_copy(t_X1, fll[c1_X2 << 3]);   

    
    fp_mul(t_X1, f_X1, t_X1); //countM++;

    fp_mul(t_X0, f_X0, t_X1); //countM++;

    fp_mul(out_t, f_A, t_X0); //countM++;

    /* ---- free all ---- */
    fp_free(h0_A);    fp_free(hlp_A);   fp_free(hlp1_A);
    fp_free(h0_PA);   fp_free(hlp_PA);  fp_free(hlp1_PA);
    fp_free(h1_PA);   fp_free(u2_PA);
    fp_free(h0_PAH);  fp_free(h1_PAH);  fp_free(u2_PAH);
    fp_free(h0_PAL);  fp_free(h1_PAL);  fp_free(u2_PAL);
    fp_free(f_A);     fp_free(f_A_sq);
    fp_free(u_X0);    fp_free(h0_X0);
    fp_free(h0_X0H);  fp_free(h1_X0H);  fp_free(u2_X0H);
    fp_free(f_X0);    fp_free(f_X0_sq);
    fp_free(u_X1);    fp_free(h0_X1);
    fp_free(f_X1);    fp_free(f_X1_sq);
    fp_free(u_X2);
    fp_free(t_X1);    fp_free(t_X0);    fp_free(t_A);
}

/* =========================================================================
 * sqrt_ext
 * ========================================================================= */
void sqrt_ext(fp_t x, fp_t y, bn_t e_exp,
              fp_t gw[nw][we],int rlll[32768], fp_t rll[we], fp_t fll[we], fp_t gpp[n])
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
        //fp_sqr(y, y);
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

	int rlll[32768]={0};

rlll[ 0x1 ]= 0 ;
rlll[ 0x2fa8 ]= 1 ;
rlll[ 0x4b7a ]= 2 ;
rlll[ 0x7a43 ]= 3 ;
rlll[ 0x3eb0 ]= 4 ;
rlll[ 0x613 ]= 5 ;
rlll[ 0x46e9 ]= 6 ;
rlll[ 0x7f40 ]= 7 ;
rlll[ 0x1719 ]= 8 ;
rlll[ 0x7055 ]= 9 ;
rlll[ 0x6ca2 ]= 10 ;
rlll[ 0xe80 ]= 11 ;
rlll[ 0x4130 ]= 12 ;
rlll[ 0x37e0 ]= 13 ;
rlll[ 0x77a5 ]= 14 ;
rlll[ 0x7561 ]= 15 ;
rlll[ 0x524b ]= 16 ;
rlll[ 0x6e9 ]= 17 ;
rlll[ 0x7802 ]= 18 ;
rlll[ 0x56dc ]= 19 ;
rlll[ 0x55eb ]= 20 ;
rlll[ 0x780d ]= 21 ;
rlll[ 0x3f8 ]= 22 ;
rlll[ 0x7797 ]= 23 ;
rlll[ 0x23c6 ]= 24 ;
rlll[ 0x5f82 ]= 25 ;
rlll[ 0x4221 ]= 26 ;
rlll[ 0x4f11 ]= 27 ;
rlll[ 0x3e74 ]= 28 ;
rlll[ 0x35f6 ]= 29 ;
rlll[ 0x3133 ]= 30 ;
rlll[ 0x7b82 ]= 31 ;
rlll[ 0x6a15 ]= 32 ;
rlll[ 0x547e ]= 33 ;
rlll[ 0x3d60 ]= 34 ;
rlll[ 0x7a37 ]= 35 ;
rlll[ 0x1424 ]= 36 ;
rlll[ 0x6889 ]= 37 ;
rlll[ 0x83a ]= 38 ;
rlll[ 0x45a7 ]= 39 ;
rlll[ 0x385b ]= 40 ;
rlll[ 0x4964 ]= 41 ;
rlll[ 0x3a0d ]= 42 ;
rlll[ 0x4683 ]= 43 ;
rlll[ 0x4c1d ]= 44 ;
rlll[ 0x75ac ]= 45 ;
rlll[ 0x86b ]= 46 ;
rlll[ 0x101e ]= 47 ;
rlll[ 0x1be ]= 48 ;
rlll[ 0x1cac ]= 49 ;
rlll[ 0x75e7 ]= 50 ;
rlll[ 0x7a08 ]= 51 ;
rlll[ 0x106d ]= 52 ;
rlll[ 0x403d ]= 53 ;
rlll[ 0x75be ]= 54 ;
rlll[ 0x67af ]= 55 ;
rlll[ 0x5a88 ]= 56 ;
rlll[ 0x222f ]= 57 ;
rlll[ 0x5c4d ]= 58 ;
rlll[ 0x53a1 ]= 59 ;
rlll[ 0x7d5b ]= 60 ;
rlll[ 0x5c07 ]= 61 ;
rlll[ 0x4ae8 ]= 62 ;
rlll[ 0x70a0 ]= 63 ;
rlll[ 0x271e ]= 64 ;
rlll[ 0x5b0c ]= 65 ;
rlll[ 0x41bf ]= 66 ;
rlll[ 0x43d1 ]= 67 ;
rlll[ 0x549f ]= 68 ;
rlll[ 0x7ab3 ]= 69 ;
rlll[ 0x70ab ]= 70 ;
rlll[ 0x40c9 ]= 71 ;
rlll[ 0x4677 ]= 72 ;
rlll[ 0x7769 ]= 73 ;
rlll[ 0x714e ]= 74 ;
rlll[ 0x5fd6 ]= 75 ;
rlll[ 0x4810 ]= 76 ;
rlll[ 0x168d ]= 77 ;
rlll[ 0x30e2 ]= 78 ;
rlll[ 0x72db ]= 79 ;
rlll[ 0x4cdf ]= 80 ;
rlll[ 0x6784 ]= 81 ;
rlll[ 0x5e0c ]= 82 ;
rlll[ 0x7e10 ]= 83 ;
rlll[ 0x323c ]= 84 ;
rlll[ 0xfb2 ]= 85 ;
rlll[ 0x45d5 ]= 86 ;
rlll[ 0x24b3 ]= 87 ;
rlll[ 0x5689 ]= 88 ;
rlll[ 0x123a ]= 89 ;
rlll[ 0x5b26 ]= 90 ;
rlll[ 0x2d0f ]= 91 ;
rlll[ 0x34b2 ]= 92 ;
rlll[ 0x5a9e ]= 93 ;
rlll[ 0x6ff6 ]= 94 ;
rlll[ 0xbb7 ]= 95 ;
rlll[ 0xbe4 ]= 96 ;
rlll[ 0x319f ]= 97 ;
rlll[ 0x573c ]= 98 ;
rlll[ 0x241e ]= 99 ;
rlll[ 0x10e7 ]= 100 ;
rlll[ 0x27c9 ]= 101 ;
rlll[ 0x45b4 ]= 102 ;
rlll[ 0x2b6d ]= 103 ;
rlll[ 0x7c24 ]= 104 ;
rlll[ 0x4a56 ]= 105 ;
rlll[ 0x2a97 ]= 106 ;
rlll[ 0x1705 ]= 107 ;
rlll[ 0x2a43 ]= 108 ;
rlll[ 0x5995 ]= 109 ;
rlll[ 0x429 ]= 110 ;
rlll[ 0x23da ]= 111 ;
rlll[ 0x73c ]= 112 ;
rlll[ 0x1c02 ]= 113 ;
rlll[ 0x37a2 ]= 114 ;
rlll[ 0x6f40 ]= 115 ;
rlll[ 0x503a ]= 116 ;
rlll[ 0x7b9f ]= 117 ;
rlll[ 0x437 ]= 118 ;
rlll[ 0x4887 ]= 119 ;
rlll[ 0x17e4 ]= 120 ;
rlll[ 0x11ea ]= 121 ;
rlll[ 0x19cb ]= 122 ;
rlll[ 0x4ec1 ]= 123 ;
rlll[ 0x5f02 ]= 124 ;
rlll[ 0xf5f ]= 125 ;
rlll[ 0x2502 ]= 126 ;
rlll[ 0x6cd ]= 127 ;
rlll[ 0x0 ]= 128 ;
rlll[ 0x5059 ]= 129 ;
rlll[ 0x3487 ]= 130 ;
rlll[ 0x5be ]= 131 ;
rlll[ 0x4151 ]= 132 ;
rlll[ 0x79ee ]= 133 ;
rlll[ 0x3918 ]= 134 ;
rlll[ 0xc1 ]= 135 ;
rlll[ 0x68e8 ]= 136 ;
rlll[ 0xfac ]= 137 ;
rlll[ 0x135f ]= 138 ;
rlll[ 0x7181 ]= 139 ;
rlll[ 0x3ed1 ]= 140 ;
rlll[ 0x4821 ]= 141 ;
rlll[ 0x85c ]= 142 ;
rlll[ 0xaa0 ]= 143 ;
rlll[ 0x2db6 ]= 144 ;
rlll[ 0x7918 ]= 145 ;
rlll[ 0x7ff ]= 146 ;
rlll[ 0x2925 ]= 147 ;
rlll[ 0x2a16 ]= 148 ;
rlll[ 0x7f4 ]= 149 ;
rlll[ 0x7c09 ]= 150 ;
rlll[ 0x86a ]= 151 ;
rlll[ 0x5c3b ]= 152 ;
rlll[ 0x207f ]= 153 ;
rlll[ 0x3de0 ]= 154 ;
rlll[ 0x30f0 ]= 155 ;
rlll[ 0x418d ]= 156 ;
rlll[ 0x4a0b ]= 157 ;
rlll[ 0x4ece ]= 158 ;
rlll[ 0x47f ]= 159 ;
rlll[ 0x15ec ]= 160 ;
rlll[ 0x2b83 ]= 161 ;
rlll[ 0x42a1 ]= 162 ;
rlll[ 0x5ca ]= 163 ;
rlll[ 0x6bdd ]= 164 ;
rlll[ 0x1778 ]= 165 ;
rlll[ 0x77c7 ]= 166 ;
rlll[ 0x3a5a ]= 167 ;
rlll[ 0x47a6 ]= 168 ;
rlll[ 0x369d ]= 169 ;
rlll[ 0x45f4 ]= 170 ;
rlll[ 0x397e ]= 171 ;
rlll[ 0x33e4 ]= 172 ;
rlll[ 0xa55 ]= 173 ;
rlll[ 0x7796 ]= 174 ;
rlll[ 0x6fe3 ]= 175 ;
rlll[ 0x7e43 ]= 176 ;
rlll[ 0x6355 ]= 177 ;
rlll[ 0xa1a ]= 178 ;
rlll[ 0x5f9 ]= 179 ;
rlll[ 0x6f94 ]= 180 ;
rlll[ 0x3fc4 ]= 181 ;
rlll[ 0xa43 ]= 182 ;
rlll[ 0x1852 ]= 183 ;
rlll[ 0x2579 ]= 184 ;
rlll[ 0x5dd2 ]= 185 ;
rlll[ 0x23b4 ]= 186 ;
rlll[ 0x2c60 ]= 187 ;
rlll[ 0x2a6 ]= 188 ;
rlll[ 0x23fa ]= 189 ;
rlll[ 0x3519 ]= 190 ;
rlll[ 0xf61 ]= 191 ;
rlll[ 0x58e3 ]= 192 ;
rlll[ 0x24f5 ]= 193 ;
rlll[ 0x3e42 ]= 194 ;
rlll[ 0x3c30 ]= 195 ;
rlll[ 0x2b62 ]= 196 ;
rlll[ 0x54e ]= 197 ;
rlll[ 0xf56 ]= 198 ;
rlll[ 0x3f38 ]= 199 ;
rlll[ 0x398a ]= 200 ;
rlll[ 0x898 ]= 201 ;
rlll[ 0xeb3 ]= 202 ;
rlll[ 0x202b ]= 203 ;
rlll[ 0x37f1 ]= 204 ;
rlll[ 0x6974 ]= 205 ;
rlll[ 0x4f1f ]= 206 ;
rlll[ 0xd26 ]= 207 ;
rlll[ 0x3322 ]= 208 ;
rlll[ 0x187d ]= 209 ;
rlll[ 0x21f5 ]= 210 ;
rlll[ 0x1f1 ]= 211 ;
rlll[ 0x4dc5 ]= 212 ;
rlll[ 0x704f ]= 213 ;
rlll[ 0x3a2c ]= 214 ;
rlll[ 0x5b4e ]= 215 ;
rlll[ 0x2978 ]= 216 ;
rlll[ 0x6dc7 ]= 217 ;
rlll[ 0x24db ]= 218 ;
rlll[ 0x52f2 ]= 219 ;
rlll[ 0x4b4f ]= 220 ;
rlll[ 0x2563 ]= 221 ;
rlll[ 0x100b ]= 222 ;
rlll[ 0x744a ]= 223 ;
rlll[ 0x741d ]= 224 ;
rlll[ 0x4e62 ]= 225 ;
rlll[ 0x28c5 ]= 226 ;
rlll[ 0x5be3 ]= 227 ;
rlll[ 0x6f1a ]= 228 ;
rlll[ 0x5838 ]= 229 ;
rlll[ 0x3a4d ]= 230 ;
rlll[ 0x5494 ]= 231 ;
rlll[ 0x3dd ]= 232 ;
rlll[ 0x35ab ]= 233 ;
rlll[ 0x556a ]= 234 ;
rlll[ 0x68fc ]= 235 ;
rlll[ 0x55be ]= 236 ;
rlll[ 0x266c ]= 237 ;
rlll[ 0x7bd8 ]= 238 ;
rlll[ 0x5c27 ]= 239 ;
rlll[ 0x78c5 ]= 240 ;
rlll[ 0x63ff ]= 241 ;
rlll[ 0x485f ]= 242 ;
rlll[ 0x10c1 ]= 243 ;
rlll[ 0x2fc7 ]= 244 ;
rlll[ 0x462 ]= 245 ;
rlll[ 0x7bca ]= 246 ;
rlll[ 0x377a ]= 247 ;
rlll[ 0x681d ]= 248 ;
rlll[ 0x6e17 ]= 249 ;
rlll[ 0x6636 ]= 250 ;
rlll[ 0x3140 ]= 251 ;
rlll[ 0x20ff ]= 252 ;
rlll[ 0x70a2 ]= 253 ;
rlll[ 0x5aff ]= 254 ;
rlll[ 0x7934 ]= 255 ;

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

//    fp_rand(b);
//    while (fp_is_sqr(b) != 1) fp_rand(b);
	fp_rand(b);
    while (fp_is_sqr(b) != 1) fp_rand(b);
    //fp_set_dig(b,4);

    /* g = z^m where z=11, m=(p-1)/2^37 */
        bn_read_str(tmp,"5",96,16);
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
