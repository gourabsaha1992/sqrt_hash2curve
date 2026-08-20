#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include <immintrin.h>
#include <relic/relic_core.h>
#include "relic/relic_md.h"


    fp_t h0_A;
    fp_t h0_23, h0_35, h0_41, h1, u2_41;
    fp_t h0_t, u2_t;
    fp_t u2_35, h0_40, u2_40, u2_23;
    fp_t h0_34, u2_34;
    fp_t f_A, f_A_sq, u_X0;
    fp_t h0_X23, h0_35b, h0_41b, u2_41b, h0_tb, u2_tb, u2_35b, h0_40b, u2_40b;
    fp_t f_X23, f_X23_sq, u_X1;
    fp_t h0_X34, h0_40c, u2_40c, h0_tc, u2_tc;
    fp_t f_X34, f_X34_sq, u_X2;
    fp_t h0_X40, h0_td, u2_td;
    fp_t f_X40, f_X40_sq, u_X3;
    fp_t h0_X43;
    fp_t f_X43, f_X43_sq, u_X4;
    fp_t d1_base_fp;
    fp_t t_X43, t_X40, t_X34, t_X23, t_A_fp;
    bn_t tmp_bn; 
    uint64_t c_41H, c_t, c_41L, c_35H;
    uint64_t c_40H, c_40L, c_35L, c_35, c_34H, c_34L, c_34, c_A;
    uint64_t c_X23, c_41Hb, c_tb_val, c_41Lb, c_41b_val;
    uint64_t c_40Hb, c_40Lb, c_35Lb;
    uint64_t c_X34, c_40Hc, c_40Lc;
    uint64_t c_X40, c_td_val;
    uint64_t c_X43;
    uint64_t c1_base;
    uint64_t e_X43, e_X40, e_X34, e_X23, e_final;
    dig_t d;
 

#define w_val   2
#define we      4           /* 2^w = 4 */
#define n       46
#define nw      23          /* ceil(46/2) = 23 */

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
 * SELECT helper
 * ========================================================================= */
static void SELECT(fp_t a0, fp_t a1, bool ctl, fp_t out)
{
    if (ctl) fp_copy(out, a1);
    else     fp_copy(out, a0);
}


static void GPOW_i41_e2(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 2) - 1;   /* mask to elen bits */
    e <<= 1;                 /* ri=1 */
    fp_copy(t, gw[20][e & 0x3]); e >>= 2;
    fp_mul(t, t, gw[21][e & 0x3]);  /* 1 M */
}

/* GPOW(43, e, 1): ri=43%2=1, e<<=1, i=43//2=21, eff_len=2 → 1 row gw[21], 0 extra M
 *   After shift: eff_len = 1+1 = 2; row 21: elen=2-2=0 ≤ 0 → done.   0 extra M. */
static void GPOW_i43_e1(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 1) - 1;
    e <<= 1;                 /* ri=1 */
    fp_copy(t, gw[21][e & 0x3]);  /* 0 extra M */
}

/* GPOW(35, e, 5): ri=35%2=1, e<<=1, i=35//2=17, eff_len=6 → 3 rows: gw[17],gw[18],gw[19]  2 M */
static void GPOW_i35_e5(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 5) - 1;
    e <<= 1;                 /* ri=1 */
    fp_copy(t, gw[17][e & 0x3]); e >>= 2;
    fp_mul(t, t, gw[18][e & 0x3]); e >>= 2;  /* 1 M */
    fp_mul(t, t, gw[19][e & 0x3]);            /* 2 M */
}

/* GPOW(40, e, 3): ri=40%2=0, i=40//2=20, eff_len=3 → 2 rows: gw[20],gw[21]  1 M */
static void GPOW_i40_e3(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 3) - 1;   /* ri=0, no shift */
    fp_copy(t, gw[20][e & 0x3]); e >>= 2;
    fp_mul(t, t, gw[21][e & 0x3]);  /* 1 M */
}

/* GPOW(34, e, 6): ri=34%2=0, i=34//2=17, eff_len=6 → 3 rows: gw[17],gw[18],gw[19]  2 M */
static void GPOW_i34_e6(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 6) - 1;   /* ri=0, no shift */
    fp_copy(t, gw[17][e & 0x3]); e >>= 2;
    fp_mul(t, t, gw[18][e & 0x3]); e >>= 2;  /* 1 M */
    fp_mul(t, t, gw[19][e & 0x3]);            /* 2 M */
}

/* GPOW(23, e, 11): ri=23%2=1, e<<=1, i=23//2=11, eff_len=12 → 6 rows: gw[11..16]  5 M */
static void GPOW_i23_e11(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 11) - 1;
    e <<= 1;                 /* ri=1 */
    fp_copy(t, gw[11][e & 0x3]); e >>= 2;
    fp_mul(t, t, gw[12][e & 0x3]); e >>= 2;  /* 1 M */
    fp_mul(t, t, gw[13][e & 0x3]); e >>= 2;  /* 2 M */
    fp_mul(t, t, gw[14][e & 0x3]); e >>= 2;  /* 3 M */
    fp_mul(t, t, gw[15][e & 0x3]); e >>= 2;  /* 4 M */
    fp_mul(t, t, gw[16][e & 0x3]);            /* 5 M */
}

/* GPOW(33, e, 6): ri=33%2=1, e<<=1, i=33//2=16, eff_len=7 → 4 rows: gw[16],gw[17],gw[18],gw[19]  3 M */
static void GPOW_i33_e6(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 6) - 1;
    e <<= 1;                 /* ri=1 */
    fp_copy(t, gw[16][e & 0x3]); e >>= 2;
    fp_mul(t, t, gw[17][e & 0x3]); e >>= 2;  /* 1 M */
    fp_mul(t, t, gw[18][e & 0x3]); e >>= 2;  /* 2 M */
    fp_mul(t, t, gw[19][e & 0x3]);            /* 3 M */
}

/* GPOW(39, e, 3): ri=39%2=1, e<<=1, i=39//2=19, eff_len=4 → 2 rows: gw[19],gw[20]  1 M */
static void GPOW_i39_e3(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 3) - 1;
    e <<= 1;                 /* ri=1 */
    fp_copy(t, gw[19][e & 0x3]); e >>= 2;
    fp_mul(t, t, gw[20][e & 0x3]);  /* 1 M */
}

/* GPOW(42, e, 1): ri=42%2=0, i=42//2=21, eff_len=1 → 1 row gw[21]  0 extra M */
static void GPOW_i42_e1(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 1) - 1;   /* ri=0, no shift */
    fp_copy(t, gw[21][e & 0x3]);
}

/* GPOW(22, e, 11): ri=22%2=0, i=22//2=11, eff_len=11 → 6 rows: gw[11..16]  5 M */
static void GPOW_i22_e11(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 11) - 1;  /* ri=0, no shift */
    fp_copy(t, gw[11][e & 0x3]); e >>= 2;
    fp_mul(t, t, gw[12][e & 0x3]); e >>= 2;  /* 1 M */
    fp_mul(t, t, gw[13][e & 0x3]); e >>= 2;  /* 2 M */
    fp_mul(t, t, gw[14][e & 0x3]); e >>= 2;  /* 3 M */
    fp_mul(t, t, gw[15][e & 0x3]); e >>= 2;  /* 4 M */
    fp_mul(t, t, gw[16][e & 0x3]);            /* 5 M */
}

/* GPOW(0, e, 22): ri=0%2=0, i=0, eff_len=22 → 11 rows: gw[0..10]  10 M */
static void GPOW_i0_e22(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= (1ULL << 22) - 1;  /* ri=0 */
    fp_copy(t, gw[0][e & 0x3]);  e >>= 2;
    fp_mul(t, t, gw[1][e & 0x3]);  e >>= 2;   /* 1 M */
    fp_mul(t, t, gw[2][e & 0x3]);  e >>= 2;   /* 2 M */
    fp_mul(t, t, gw[3][e & 0x3]);  e >>= 2;   /* 3 M */
    fp_mul(t, t, gw[4][e & 0x3]);  e >>= 2;   /* 4 M */
    fp_mul(t, t, gw[5][e & 0x3]);  e >>= 2;   /* 5 M */
    fp_mul(t, t, gw[6][e & 0x3]);  e >>= 2;   /* 6 M */
    fp_mul(t, t, gw[7][e & 0x3]);  e >>= 2;   /* 7 M */
    fp_mul(t, t, gw[8][e & 0x3]);  e >>= 2;   /* 8 M */
    fp_mul(t, t, gw[9][e & 0x3]);  e >>= 2;   /* 9 M */
    fp_mul(t, t, gw[10][e & 0x3]);             /* 10 M */
}


static uint64_t lookup_rll(fp_t x, fp_t rll[we])
{
    int v;
    for (v = 0; v < we; v++) {
        if (fp_cmp(x, rll[v]) == RLC_EQ) return (uint64_t)v;
    }
    /* Should never reach here for valid input */
    return 0;
}


void precomputation(fp_t g, fp_t h, fp_t hh,
                    fp_t rll[we], fp_t fll[we],
                    fp_t gw[nw][we], fp_t gpp[n])
{
    bn_t temp, one;
    bn_null(temp); bn_new(temp);
    bn_null(one);  bn_new(one);
    bn_set_dig(one, 1);

    /* rll[v] = h^v */
    for (int v = 0; v < we; v++) {
        bn_set_dig(temp, v);
        fp_exp(rll[v], h, temp);
    }
    
    for (int v = 0; v < we; v++) {
        bn_set_dig(temp, v);
        fp_exp(fll[v], hh, temp);
    }
    /* gw[i][j] = g^(j * 2^(i*w)) */
    for (int i = 0; i < nw; i++) {
        for (int j = 0; j < we; j++) {
            bn_lsh(temp, one, i * w_val);
            bn_mul_dig(temp, temp, j);
            fp_exp(gw[i][j], g, temp);
        }
    }
    /* gpp[i] = g^(2^i) */
    fp_copy(gpp[0], g);
    for (int i = 1; i < n; i++)
        fp_sqr(gpp[i], gpp[i - 1]);

    bn_free(temp);
    bn_free(one);
}


void DLPpow2ext(fp_t u_A, fp_t out_t,
                fp_t rll[we], fp_t fll[we],
                fp_t gw[nw][we], fp_t gpp[n], int rlll[4])
{
    const uint64_t one_k = 1ULL;

    fp_copy(h0_A, u_A);
    for (int _j = 0; _j < 23; _j++) fp_sqr(h0_A, h0_A);

    
    fp_copy(h0_23, h0_A);
    for (int _j = 0; _j < 12; _j++) fp_sqr(h0_23, h0_23);

    
    fp_copy(h0_35, h0_23);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_35, h0_35);

    
    fp_copy(h0_41, h0_35);
    for (int _j = 0; _j < 3; _j++) fp_sqr(h0_41, h0_41);

    
    fp_prime_back(tmp_bn, h0_41);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_41H=rlll[d];
    
    GPOW_i41_e2(h1, (one_k << 2) - c_41H, gw);
    SELECT(h1, gpp[43], c_41H == 0, h1);
    fp_mul(u2_41, h0_35, h1);   /* 1 M */

    
    fp_copy(h0_t, u2_41);
    for (int _j = 0; _j < 2; _j++) fp_sqr(h0_t, h0_t);

    
    fp_prime_back(tmp_bn, h0_t);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_t = rlll[d] >> 1;
    GPOW_i43_e1(h1, (one_k << 1) - c_t, gw);
    SELECT(h1, gpp[44], c_t == 0, h1);
    fp_mul(u2_t, u2_41, h1);    /* 1 M */

    
    c_41L = c_t + (((lookup_rll(u2_t, rll) - 1) & 0x3ULL) << 1);   /* 3-bit */
    c_35H = c_41H + (((c_41L - 1) & 0x7ULL) << 2);                  /* 5-bit */

    
    GPOW_i35_e5(h1, (one_k << 5) - c_35H, gw);
    SELECT(h1, gpp[40], c_35H == 0, h1);
    fp_mul(u2_35, h0_23, h1);   /* 1 M */

    
    fp_copy(h0_40, u2_35);
    for (int _j = 0; _j < 3; _j++) fp_sqr(h0_40, h0_40);

    
    fp_copy(h0_t, h0_40);
    for (int _j = 0; _j < 2; _j++) fp_sqr(h0_t, h0_t);
    fp_prime_back(tmp_bn, h0_t);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_t = rlll[d] >> 1;
    GPOW_i43_e1(h1, (one_k << 1) - c_t, gw);
    SELECT(h1, gpp[44], c_t == 0, h1);
    fp_mul(u2_t, h0_40, h1);    /* 1 M */
    c_40H = c_t + (((lookup_rll(u2_t, rll) - 1) & 0x3ULL) << 1);   /* 3-bit */

    /* DLP@40 correction → u2_40 */
    GPOW_i40_e3(h1, (one_k << 3) - c_40H, gw);
    SELECT(h1, gpp[43], c_40H == 0, h1);
    fp_mul(u2_40, u2_35, h1);   /* 1 M */

    /* ── DLP@43 [L-child of DLP@40] ───────────────────────────────────── */
    fp_copy(h0_t, u2_40);
    for (int _j = 0; _j < 2; _j++) fp_sqr(h0_t, h0_t);
    fp_prime_back(tmp_bn, h0_t);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_t = rlll[d] >> 1;
    GPOW_i43_e1(h1, (one_k << 1) - c_t, gw);
    SELECT(h1, gpp[44], c_t == 0, h1);
    fp_mul(u2_t, u2_40, h1);    /* 1 M */
    fp_prime_back(tmp_bn, u2_t);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_40L = c_t + (((rlll[d] - 1) & 0x3ULL) << 1);   /* 3-bit */
    c_35L = c_40H + (((c_40L - 1) & 0x7ULL) << 3);                  /* 6-bit */

    c_35 = c_35H + (((c_35L - 1) & 0x3FULL) << 5);                  /* 11-bit (= c_PAH in w=4 code) */

    /* ── DLP@23 correction → u2_23 ─────────────────────────────────────── */
    GPOW_i23_e11(h1, (one_k << 11) - c_35, gw);
    SELECT(h1, gpp[34], c_35 == 0, h1);
    fp_mul(u2_23, h0_A, h1);    /* 1 M */

    /* ── DLP@34 [L-child of DLP@23]: Square 6 ─────────────────────────── */
    fp_copy(h0_34, u2_23);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_34, h0_34);

    /* ── DLP@40 [H-child of DLP@34] ───────────────────────────────────── */
    fp_copy(h0_40, h0_34);
    for (int _j = 0; _j < 3; _j++) fp_sqr(h0_40, h0_40);
    /* DLP@43 [H of DLP@40] */
    fp_copy(h0_t, h0_40);
    for (int _j = 0; _j < 2; _j++) fp_sqr(h0_t, h0_t);
    fp_prime_back(tmp_bn, h0_t);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_t = rlll[d] >> 1;
    GPOW_i43_e1(h1, (one_k << 1) - c_t, gw);
    SELECT(h1, gpp[44], c_t == 0, h1);
    fp_mul(u2_t, h0_40, h1);    /* 1 M */
    
    fp_prime_back(tmp_bn, u2_t);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_40H = c_t + (((rlll[d] - 1) & 0x3ULL) << 1);
    GPOW_i40_e3(h1, (one_k << 3) - c_40H, gw);
    SELECT(h1, gpp[43], c_40H == 0, h1);
    fp_mul(u2_40, h0_34, h1);   /* 1 M */
    /* DLP@43 [L of DLP@40] */
    fp_copy(h0_t, u2_40);
    for (int _j = 0; _j < 2; _j++) fp_sqr(h0_t, h0_t);
    fp_prime_back(tmp_bn, h0_t);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_t = rlll[d] >> 1;
    GPOW_i43_e1(h1, (one_k << 1) - c_t, gw);
    SELECT(h1, gpp[44], c_t == 0, h1);
    fp_mul(u2_t, u2_40, h1);    /* 1 M */
    fp_prime_back(tmp_bn,u2_t);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_40L = c_t + (((rlll[d] - 1) & 0x3ULL) << 1);
    c_34H = c_40H + (((c_40L - 1) & 0x7ULL) << 3);                  /* 6-bit */

    /* DLP@34 correction → u2_34 */
    GPOW_i34_e6(h1, (one_k << 6) - c_34H, gw);
    SELECT(h1, gpp[40], c_34H == 0, h1);
    fp_mul(u2_34, u2_23, h1);   /* 1 M */

    /* ── DLP@40 [L-child of DLP@34] ───────────────────────────────────── */
    fp_copy(h0_40, u2_34);
    for (int _j = 0; _j < 3; _j++) fp_sqr(h0_40, h0_40);
    /* DLP@43 [H of DLP@40] */
    fp_copy(h0_t, h0_40);
    for (int _j = 0; _j < 2; _j++) fp_sqr(h0_t, h0_t);
    fp_prime_back(tmp_bn, h0_t);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_t = rlll[d] >> 1;
    GPOW_i43_e1(h1, (one_k << 1) - c_t, gw);
    SELECT(h1, gpp[44], c_t == 0, h1);
    fp_mul(u2_t, h0_40, h1);    /* 1 M */
    fp_prime_back(tmp_bn, u2_t);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_40H = c_t + (((rlll[d] - 1) & 0x3ULL) << 1);
    GPOW_i40_e3(h1, (one_k << 3) - c_40H, gw);
    SELECT(h1, gpp[43], c_40H == 0, h1);
    fp_mul(u2_40, u2_34, h1);   /* 1 M */
    /* DLP@43 [L of DLP@40] */
    fp_copy(h0_t, u2_40);
    for (int _j = 0; _j < 2; _j++) fp_sqr(h0_t, h0_t);
    fp_prime_back(tmp_bn, h0_t);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_t = rlll[d] >> 1;
    GPOW_i43_e1(h1, (one_k << 1) - c_t, gw);
    SELECT(h1, gpp[44], c_t == 0, h1);
    fp_mul(u2_t, u2_40, h1);    /* 1 M */
    fp_prime_back(tmp_bn, u2_t);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_40L = c_t + (((rlll[d] - 1) & 0x3ULL) << 1);
    c_34L = c_40H + (((c_40L - 1) & 0x7ULL) << 3);                  /* 6-bit */

    c_34 = c_34H + (((c_34L - 1) & 0x3FULL) << 6);                  /* 12-bit */
    c_A  = c_35  + (((c_34  - 1) & 0xFFFULL) << 11);                /* 23-bit */


    GPOW_i0_e22(f_A, ((one_k << 23) - c_A + 1) >> 1, gw);
    SELECT(f_A, gpp[22], c_A == 0, f_A);
    fp_sqr(f_A_sq, f_A);
    fp_mul(u_X0, u_A, f_A_sq);  /* 1 M  (f_A^2 * u_A) */


    fp_copy(h0_X23, u_X0);
    for (int _j = 0; _j < 12; _j++) fp_sqr(h0_X23, h0_X23);

    /* DLP@35(h0_X23) → c_X23 */
    fp_copy(h0_35b, h0_X23);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_35b, h0_35b);

    /* DLP@41 */
    fp_copy(h0_41b, h0_35b);
    for (int _j = 0; _j < 3; _j++) fp_sqr(h0_41b, h0_41b);
    fp_prime_back(tmp_bn, h0_41b);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_41Hb =  rlll[d];   
    GPOW_i41_e2(h1, (one_k << 2) - c_41Hb, gw);
    SELECT(h1, gpp[43], c_41Hb == 0, h1);
    fp_mul(u2_41b, h0_35b, h1);   /* 1 M */

    /* DLP@43 [L of DLP@41] */
    fp_copy(h0_tb, u2_41b);
    for (int _j = 0; _j < 2; _j++) fp_sqr(h0_tb, h0_tb);
    fp_prime_back(tmp_bn, h0_tb);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_tb_val = rlll[d] >> 1;
    GPOW_i43_e1(h1, (one_k << 1) - c_tb_val, gw);
    SELECT(h1, gpp[44], c_tb_val == 0, h1);
    fp_mul(u2_tb, u2_41b, h1);    /* 1 M */
    fp_prime_back(tmp_bn, u2_tb);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_41Lb    = c_tb_val + (((rlll[d] - 1) & 0x3ULL) << 1);
    c_41b_val = c_41Hb + (((c_41Lb - 1) & 0x7ULL) << 2);            /* 5-bit */

    GPOW_i35_e5(h1, (one_k << 5) - c_41b_val, gw);
    SELECT(h1, gpp[40], c_41b_val == 0, h1);
    fp_mul(u2_35b, h0_X23, h1);   /* 1 M */

    /* DLP@40 [L of DLP@35] */
    fp_copy(h0_40b, u2_35b);
    for (int _j = 0; _j < 3; _j++) fp_sqr(h0_40b, h0_40b);
    /* DLP@43 [H of DLP@40] */
    fp_copy(h0_tb, h0_40b);
    for (int _j = 0; _j < 2; _j++) fp_sqr(h0_tb, h0_tb);
    fp_prime_back(tmp_bn, h0_tb);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_tb_val = rlll[d] >> 1;
    GPOW_i43_e1(h1, (one_k << 1) - c_tb_val, gw);
    SELECT(h1, gpp[44], c_tb_val == 0, h1);
    fp_mul(u2_tb, h0_40b, h1);    /* 1 M */
    fp_prime_back(tmp_bn, u2_tb);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_40Hb = c_tb_val + (((rlll[d] - 1) & 0x3ULL) << 1);
    GPOW_i40_e3(h1, (one_k << 3) - c_40Hb, gw);
    SELECT(h1, gpp[43], c_40Hb == 0, h1);
    fp_mul(u2_40b, u2_35b, h1);   /* 1 M */
    /* DLP@43 [L of DLP@40] */
    fp_copy(h0_tb, u2_40b);
    for (int _j = 0; _j < 2; _j++) fp_sqr(h0_tb, h0_tb);
    fp_prime_back(tmp_bn, h0_tb);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_tb_val = rlll[d] >> 1;
    GPOW_i43_e1(h1, (one_k << 1) - c_tb_val, gw);
    SELECT(h1, gpp[44], c_tb_val == 0, h1);
    fp_mul(u2_tb, u2_40b, h1);    /* 1 M */
    fp_prime_back(tmp_bn, u2_tb);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_40Lb = c_tb_val + (((rlll[d] - 1) & 0x3ULL) << 1);
    c_35Lb = c_40Hb + (((c_40Lb - 1) & 0x7ULL) << 3);
    c_X23  = c_41b_val + (((c_35Lb - 1) & 0x3FULL) << 5);           /* 11-bit */

    /* f_X23: GPOW(22, 2^11 - c_X23, 11) */
    GPOW_i22_e11(f_X23, (one_k << 11) - c_X23, gw);
    SELECT(f_X23, gpp[33], c_X23 == 0, f_X23);
    fp_sqr(f_X23_sq, f_X23);
    fp_mul(u_X1, u_X0, f_X23_sq);   /* 1 M */

    /* =====================================================================
     * EXT2@34: Square 6 → h0_X34
     * ===================================================================== */
    fp_copy(h0_X34, u_X1);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_X34, h0_X34);

    /* DLP@40(h0_X34) → c_X34 */
    fp_copy(h0_40c, h0_X34);
    for (int _j = 0; _j < 3; _j++) fp_sqr(h0_40c, h0_40c);
    /* DLP@43 [H of DLP@40] */
    fp_copy(h0_tc, h0_40c);
    for (int _j = 0; _j < 2; _j++) fp_sqr(h0_tc, h0_tc);
    fp_prime_back(tmp_bn, h0_tc);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_tb_val = rlll[d] >> 1;
    GPOW_i43_e1(h1, (one_k << 1) - c_tb_val, gw);
    SELECT(h1, gpp[44], c_tb_val == 0, h1);
    fp_mul(u2_tc, h0_40c, h1);    /* 1 M */
    fp_prime_back(tmp_bn, u2_tc);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_40Hc = c_tb_val + (((rlll[d] - 1) & 0x3ULL) << 1);
    GPOW_i40_e3(h1, (one_k << 3) - c_40Hc, gw);
    SELECT(h1, gpp[43], c_40Hc == 0, h1);
    fp_mul(u2_40c, h0_X34, h1);   /* 1 M */
    /* DLP@43 [L of DLP@40] */
    fp_copy(h0_tc, u2_40c);
    for (int _j = 0; _j < 2; _j++) fp_sqr(h0_tc, h0_tc);
    fp_prime_back(tmp_bn, h0_tc);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_tb_val = rlll[d] >> 1;
    GPOW_i43_e1(h1, (one_k << 1) - c_tb_val, gw);
    SELECT(h1, gpp[44], c_tb_val == 0, h1);
    fp_mul(u2_tc, u2_40c, h1);    /* 1 M */
    fp_prime_back(tmp_bn, u2_tc);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_40Lc = c_tb_val + (((rlll[d] - 1) & 0x3ULL) << 1);
    c_X34  = c_40Hc + (((c_40Lc - 1) & 0x7ULL) << 3);               /* 6-bit */

    /* f_X34: GPOW(33, 2^6 - c_X34, 6) */
    GPOW_i33_e6(f_X34, (one_k << 6) - c_X34, gw);
    SELECT(f_X34, gpp[39], c_X34 == 0, f_X34);
    fp_sqr(f_X34_sq, f_X34);
    fp_mul(u_X2, u_X1, f_X34_sq);   /* 1 M */

    
    fp_copy(h0_X40, u_X2);
    for (int _j = 0; _j < 3; _j++) fp_sqr(h0_X40, h0_X40);

    /* DLP@43(h0_X40) → c_X40 */
    fp_copy(h0_td, h0_X40);
    for (int _j = 0; _j < 2; _j++) fp_sqr(h0_td, h0_td);
    fp_prime_back(tmp_bn, h0_td);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_td_val = rlll[d] >> 1;
    GPOW_i43_e1(h1, (one_k << 1) - c_td_val, gw);
    SELECT(h1, gpp[44], c_td_val == 0, h1);
    fp_mul(u2_td, h0_X40, h1);    /* 1 M */
    fp_prime_back(tmp_bn, u2_td);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_X40 = c_td_val + (((rlll[d] - 1) & 0x3ULL) << 1);  /* 3-bit */

    /* f_X40: GPOW(39, 2^3 - c_X40, 3) */
    GPOW_i39_e3(f_X40, (one_k << 3) - c_X40, gw);
    SELECT(f_X40, gpp[42], c_X40 == 0, f_X40);
    fp_sqr(f_X40_sq, f_X40);
    fp_mul(u_X3, u_X2, f_X40_sq);   /* 1 M */

    
    fp_copy(h0_X43, u_X3);
    for (int _j = 0; _j < 2; _j++) fp_sqr(h0_X43, h0_X43);

    /* inner DLP@45: LEAF lb=1, shift=1  →  c_X43 = lookup(h0_X43) >> 1 */
    fp_prime_back(tmp_bn, h0_X43);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c_X43 = rlll[d] >> 1;

    /* f_X43: GPOW(42, 2^1 - c_X43, 1) */
    GPOW_i42_e1(f_X43, (one_k << 1) - c_X43, gw);
    SELECT(f_X43, gpp[43], c_X43 == 0, f_X43);
    fp_sqr(f_X43_sq, f_X43);
    fp_mul(u_X4, u_X3, f_X43_sq);   /* 1 M */

    
     fp_prime_back(tmp_bn, u_X4);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    c1_base = rlll[d];
    fp_copy(d1_base_fp, fll[c1_base]);


    fp_mul(t_X43, f_X43, d1_base_fp);

    fp_mul(t_X40, f_X40, t_X43);

    fp_mul(t_X34, f_X34, t_X40);

    fp_mul(t_X23, f_X23, t_X34);

    fp_mul(out_t, f_A, t_X23); 

}


void sqrt_ext(fp_t x, fp_t y, bn_t e_exp,
              fp_t rll[we], fp_t fll[we],
              fp_t gw[nw][we], fp_t gpp[n],int rlll[4])
{
    fp_t u, v, w_, t;
    fp_null(u); fp_null(v); fp_null(w_); fp_null(t);
    RLC_TRY {
        fp_new(u); fp_new(v); fp_new(w_); fp_new(t);
        fp_exp(v,  x, e_exp);      /* v = x^((m-1)/2) */
        fp_mul(w_, x, v);          /* w_ = x * v  =  x^((m+1)/2) */
        fp_mul(u,  w_, v);         /* u  = w_ * v =  x^m           */
        DLPpow2ext(u, t, rll, fll, gw, gpp,rlll);
        fp_mul(y, w_, t);          /* y = w_ * t */
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
    if (core_init() != RLC_OK) { core_clean(); return 1; }
    if (ep_param_set_any_pairf() != RLC_OK) {
        printf("Curve initialisation failed\n");
        core_clean(); return 1;
    }

    int i, j;
    fp_t rll[we], fll[we], gw[nw][we], gpp[n];
    fp_t g, z, b, h, hh, y;
    bn_t tmp, m, e;
    int rlll[4]={0};
    rlll[ 0x1 ]= 0 ;
    rlll[ 0x3 ]= 1 ;
    rlll[ 0x0 ]= 2 ;
    rlll[ 0x2 ]= 3 ;

    for (i = 0; i < nw; i++)
        for (j = 0; j < we; j++) { fp_null(gw[i][j]); }
    for (i = 0; i < we; i++) { fp_null(rll[i]); fp_null(fll[i]); }
    for (i = 0; i < n;  i++) { fp_null(gpp[i]); }
    fp_null(b); fp_null(y); fp_null(g); fp_null(z); fp_null(h); fp_null(hh);
    bn_null(tmp); bn_null(m); bn_null(e);

    RLC_TRY {
        bn_new(e); fp_new(g); fp_new(b);
        bn_new(m); fp_new(y); fp_new(z);
        bn_new(tmp); fp_new(hh); fp_new(h);
        for (i = 0; i < nw; i++)
            for (j = 0; j < we; j++) { fp_new(gw[i][j]); }
        for (i = 0; i < we; i++) { fp_new(rll[i]); fp_new(fll[i]); }
        for (i = 0; i < n;  i++) { fp_new(gpp[i]); }

        /* z = 5  (primitive element used in Sage) */
        bn_read_str(tmp, "5", 1, 16);
        fp_prime_conv(z, tmp);

        
        bn_read_str(m,
            "6b8e9185f1443ab18ec1701b28524ec688b67cc03d44e3c7bcd88bee82520005c2d7510c00000021423",
            84, 16);
        fp_exp(g, z, m);        

        
        bn_read_str(e,
            "35c748c2f8a21d58c760b80d94292763445b3e601ea271e3de6c45f741290002e16ba88600000010a11",
            83, 16);

        /* h = g^(2^(n-w)) = g^(2^44) */
        bn_t a1;
        bn_null(a1); bn_new(a1);
        bn_read_str(a1, "100000000000", 12, 16);   /* 2^44 in hex */
        fp_exp(h, g, a1);
        bn_free(a1);

        /* hh = (sqrt(h))^{-1} */
        fp_srt(hh, h);
        fp_inv(hh, hh);

        precomputation(g, h, hh, rll, fll, gw, gpp);

        /* pick a random quadratic residue */
        fp_rand(b);
        while (fp_is_sqr(b) != 1) fp_rand(b);

        MEASURE(sqrt_ext(b, y, e, rll, fll, gw, gpp,rlll);)

        printf("RDTSC_clk_min    = %f\n", RDTSC_clk_min);
        printf("RDTSC_clk_median = %f\n", RDTSC_clk_median);
        printf("RDTSC_clk_max    = %f\n", RDTSC_clk_max);

        
        fp_t y2; fp_null(y2); fp_new(y2);
        
        fp_free(y2);
    }
    RLC_CATCH_ANY { RLC_THROW(ERR_CAUGHT); }
    RLC_FINALLY {
        for (i = 0; i < nw; i++)
            for (j = 0; j < we; j++) fp_free(gw[i][j]);
        for (i = 0; i < we; i++) { fp_free(rll[i]); fp_free(fll[i]); }
        for (i = 0; i < n;  i++) fp_free(gpp[i]);
        fp_free(b); fp_free(y); fp_free(g); fp_free(z); fp_free(h); fp_free(hh);
        bn_free(tmp); bn_free(e); bn_free(m);
    }
    core_clean();
    return 0;
}
