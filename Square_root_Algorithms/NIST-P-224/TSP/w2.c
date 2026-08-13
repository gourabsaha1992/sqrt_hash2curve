#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include <immintrin.h>
#include <relic/relic_core.h>
#include "relic/relic_md.h"

/* n=96, w=2, nw=48, we=4
 * p = 2^224 - 2^96 + 1
 * do_hlp=False EVERYWHERE (cost >= nlb1 at every DLP node)
 * Base cases: i=95 lb=1 >>1  |  i=94 lb=2 >>0
 * DLP nodes: DLP93(sq2), DLP90(sq3), DLP84(sq6), DLP72(sq12), DLP48(sq24)
 * EXT2 (HIGH-only sub-calls, unwind combines):
 *   X0(i=48): sq24, c_X0=DLP72(h0_X0), f_X0=GPOW_i47_e24
 *   X1(i=72): sq12, c_X1=DLP84(h0_X1), f_X1=GPOW_i71_e12
 *   X2(i=84): sq6,  c_X2=DLP90(h0_X2), f_X2=GPOW_i83_e6
 *   X3(i=90): sq3,  c_X3=DLP93(h0_X3), f_X3=GPOW_i89_e3
 *   X4(i=93): sq2,  c_X4=BASE(95)>>1,   f_X4=GPOW_i92_e1
 *   BASE(i=94): lb=2=w -> fll lookup >>0
 * Expected: M=172, S=310 */

#define w   2
#define we  4        /* 2^w */
#define n   96
#define nw  48       /* ceil(96/2) */

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

/* GPOW_i0_e47: ri=0, row=0, adj=47, rows=24, 23 muls  [f_A] */
static void GPOW_i0_e47(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 47) - 1;
    uint64_t wm = we - 1;
    fp_copy(t, gw[0][e & wm]);
    e >>= w; fp_mul(t, t, gw[1][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[2][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[3][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[4][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[5][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[6][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[7][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[8][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[9][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[10][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[11][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[12][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[13][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[14][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[15][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[16][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[17][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[18][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[19][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[20][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[21][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[22][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[23][e & wm]); //countM++;
}

/* GPOW_i47_e24: ri=1, row=23, e<<=1, adj=25, rows=13, 12 muls  [f_X0] */
static void GPOW_i47_e24(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 24) - 1;
    e <<= 1;
    uint64_t wm = we - 1;
    fp_copy(t, gw[23][e & wm]);
    e >>= w; fp_mul(t, t, gw[24][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[25][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[26][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[27][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[28][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[29][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[30][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[31][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[32][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[33][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[34][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[35][e & wm]); //countM++;
}

/* GPOW_i48_e24: ri=0, row=24, adj=24, rows=12, 11 muls  [c_A correct (PA at i=48)] */
static void GPOW_i48_e24(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 24) - 1;
    uint64_t wm = we - 1;
    fp_copy(t, gw[24][e & wm]);
    e >>= w; fp_mul(t, t, gw[25][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[26][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[27][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[28][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[29][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[30][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[31][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[32][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[33][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[34][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[35][e & wm]); //countM++;
}

/* GPOW_i71_e12: ri=1, row=35, e<<=1, adj=13, rows=7, 6 muls  [f_X1] */
static void GPOW_i71_e12(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 12) - 1;
    e <<= 1;
    uint64_t wm = we - 1;
    fp_copy(t, gw[35][e & wm]);
    e >>= w; fp_mul(t, t, gw[36][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[37][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[38][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[39][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[40][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[41][e & wm]); //countM++;
}

/* GPOW_i72_e12: ri=0, row=36, adj=12, rows=6, 5 muls  [c_PAH correct (DLP72); c_X0P correct] */
static void GPOW_i72_e12(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 12) - 1;
    uint64_t wm = we - 1;
    fp_copy(t, gw[36][e & wm]);
    e >>= w; fp_mul(t, t, gw[37][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[38][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[39][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[40][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[41][e & wm]); //countM++;
}

/* GPOW_i83_e6: ri=1, row=41, e<<=1, adj=7, rows=4, 3 muls  [f_X2] */
static void GPOW_i83_e6(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 6) - 1;
    e <<= 1;
    uint64_t wm = we - 1;
    fp_copy(t, gw[41][e & wm]);
    e >>= w; fp_mul(t, t, gw[42][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[43][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[44][e & wm]); //countM++;
}

/* GPOW_i84_e6: ri=0, row=42, adj=6, rows=3, 2 muls  [c_PAHH/PAHL/PALH/PALL/X0PH/X0PL/X1P correct (DLP84)] */
static void GPOW_i84_e6(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 6) - 1;
    uint64_t wm = we - 1;
    fp_copy(t, gw[42][e & wm]);
    e >>= w; fp_mul(t, t, gw[43][e & wm]); //countM++;
    e >>= w; fp_mul(t, t, gw[44][e & wm]); //countM++;
}

/* GPOW_i89_e3: ri=1, row=44, e<<=1, adj=4, rows=2, 1 muls  [f_X3] */
static void GPOW_i89_e3(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 3) - 1;
    e <<= 1;
    uint64_t wm = we - 1;
    fp_copy(t, gw[44][e & wm]);
    e >>= w; fp_mul(t, t, gw[45][e & wm]); //countM++;
}

/* GPOW_i90_e3: ri=0, row=45, adj=3, rows=2, 1 muls  [c_PAHHH/.../X2P correct (DLP90)] */
static void GPOW_i90_e3(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 3) - 1;
    uint64_t wm = we - 1;
    fp_copy(t, gw[45][e & wm]);
    e >>= w; fp_mul(t, t, gw[46][e & wm]); //countM++;
}

/* GPOW_i92_e1: ri=0, row=46, adj=1, rows=1, 0 muls  [f_X4] */
static void GPOW_i92_e1(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 1) - 1;
    uint64_t wm = we - 1;
    fp_copy(t, gw[46][e & wm]);
}

/* GPOW_i93_e1: ri=1, row=46, e<<=1, adj=2, rows=1, 0 muls  [DLP93 leaf corrector] */
static void GPOW_i93_e1(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 1) - 1;
    e <<= 1;
    uint64_t wm = we - 1;
    fp_copy(t, gw[46][e & wm]);
}

void DLPpow2ext(fp_t u_A, fp_t out_t,
                fp_t gw[nw][we], int rlll[16], fp_t rll[we], fp_t fll[we], fp_t gpp[n])
{
    uint64_t one_k = 1;
    int _ii, _tmp;
    fp_t f_A, f_A_sq, u_X0, f_X0, f_X0_sq, u_X1, f_X1, f_X1_sq;
    fp_t u_X2, f_X2, f_X2_sq, u_X3, f_X3, f_X3_sq, u_X4, f_X4, f_X4_sq;
    fp_t u_X5, t_X4, t_X3, t_X2, t_X1, t_X0, t_A;
    fp_null(f_A); fp_new(f_A);
    fp_null(f_A_sq); fp_new(f_A_sq);
    fp_null(u_X0); fp_new(u_X0);
    fp_null(f_X0); fp_new(f_X0);
    fp_null(f_X0_sq); fp_new(f_X0_sq);
    fp_null(u_X1); fp_new(u_X1);
    fp_null(f_X1); fp_new(f_X1);
    fp_null(f_X1_sq); fp_new(f_X1_sq);
    fp_null(u_X2); fp_new(u_X2);
    fp_null(f_X2); fp_new(f_X2);
    fp_null(f_X2_sq); fp_new(f_X2_sq);
    fp_null(u_X3); fp_new(u_X3);
    fp_null(f_X3); fp_new(f_X3);
    fp_null(f_X3_sq); fp_new(f_X3_sq);
    fp_null(u_X4); fp_new(u_X4);
    fp_null(f_X4); fp_new(f_X4);
    fp_null(f_X4_sq); fp_new(f_X4_sq);
    fp_null(u_X5); fp_new(u_X5);
    fp_null(t_X4); fp_new(t_X4);
    fp_null(t_X3); fp_new(t_X3);
    fp_null(t_X2); fp_new(t_X2);
    fp_null(t_X1); fp_new(t_X1);
    fp_null(t_X0); fp_new(t_X0);
    fp_null(t_A); fp_new(t_A);

    /* ===== TOP: sq48, do_hlp_A=False ===== */
    fp_t h0_A; fp_null(h0_A); fp_new(h0_A);
    fp_copy(h0_A, u_A);
    for (int _j = 0; _j < 48; _j++) fp_sqr(h0_A, h0_A); //countS += 48;

    /* ===== c_A = DLP(h0_A, i=48) ===== */
    /* DLP48: sq24, DLP72(H)->c_PAH, GPOW_i48_e24, DLP72(L)->d_PA, combine */
    fp_t h0_PA, h1_PA, u2_PA;
    fp_null(h0_PA); fp_new(h0_PA); fp_null(h1_PA); fp_new(h1_PA); fp_null(u2_PA); fp_new(u2_PA);
    fp_copy(h0_PA, h0_A);
    for (int _j = 0; _j < 24; _j++) fp_sqr(h0_PA, h0_PA); //countS += 24;
    /* DLP72 [PAH]: sq12, DLP84(H), GPOW_i72_e12, DLP84(L) */
    fp_t h0_PAH, h1_PAH, u2_PAH;
    fp_null(h0_PAH); fp_new(h0_PAH); fp_null(h1_PAH); fp_new(h1_PAH); fp_null(u2_PAH); fp_new(u2_PAH);
    fp_copy(h0_PAH, h0_PA);
    for (int _sq = 0; _sq < 12; _sq++) fp_sqr(h0_PAH, h0_PAH); //countS += 12;
    /* DLP84 [PAHH]: sq6, DLP90(H), GPOW_i84_e6, DLP90(L) */
    fp_t h0_PAHH, h1_PAHH, u2_PAHH;
    fp_null(h0_PAHH); fp_new(h0_PAHH); fp_null(h1_PAHH); fp_new(h1_PAHH); fp_null(u2_PAHH); fp_new(u2_PAHH);
    fp_copy(h0_PAHH, h0_PAH);
    for (int _sq = 0; _sq < 6; _sq++) fp_sqr(h0_PAHH, h0_PAHH); //countS += 6;
    /* DLP90 [PAHHH]: sq3, DLP93(H), GPOW_i90_e3, DLP93(L) */
    fp_t h0_PAHHH, h1_PAHHH, u2_PAHHH;
    fp_null(h0_PAHHH); fp_new(h0_PAHHH); fp_null(h1_PAHHH); fp_new(h1_PAHHH); fp_null(u2_PAHHH); fp_new(u2_PAHHH);
    fp_copy(h0_PAHHH, h0_PAHH);
    fp_sqr(h0_PAHHH, h0_PAHHH); fp_sqr(h0_PAHHH, h0_PAHHH); fp_sqr(h0_PAHHH, h0_PAHHH); //countS += 3;
    /* DLP93 [PAHHHH]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_PAHHHH, h1_PAHHHH, u2_PAHHHH;
    fp_null(h0_PAHHHH); fp_new(h0_PAHHHH); fp_null(h1_PAHHHH); fp_new(h1_PAHHHH); fp_null(u2_PAHHHH); fp_new(u2_PAHHHH);
    fp_copy(h0_PAHHHH, h0_PAHHH);
    fp_sqr(h0_PAHHHH, h0_PAHHHH); fp_sqr(h0_PAHHHH, h0_PAHHHH); //countS += 2;
	bn_t tmp_bn;
	dig_t d;
	bn_null(tmp_bn);bn_new(tmp_bn);	
	fp_prime_back(tmp_bn, h0_PAHHHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_PAHHHH) == RLC_EQ) _tmp = _ii;
    uint64_t c_PAHHHHH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_PAHHHH, (one_k << 1) - c_PAHHHHH, gw);
    SELECT(h1_PAHHHH, gpp[94], c_PAHHHHH == 0, h1_PAHHHH);
    //countM++;
    fp_mul(u2_PAHHHH, h0_PAHHH, h1_PAHHHH);
	fp_prime_back(tmp_bn, u2_PAHHHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_PAHHHH) == RLC_EQ) _tmp = _ii;
    uint64_t d_PAHHHHL = (uint64_t)_tmp >> 0;
    uint64_t c_PAHHHH = c_PAHHHHH + ((d_PAHHHHL - 1) % 4) * 2;
    fp_free(h0_PAHHHH); fp_free(h1_PAHHHH); fp_free(u2_PAHHHH);
    GPOW_i90_e3(h1_PAHHH, (one_k << 3) - c_PAHHHH, gw);
    SELECT(h1_PAHHH, gpp[93], c_PAHHHH == 0, h1_PAHHH);
    //countM++;
    fp_mul(u2_PAHHH, h0_PAHH, h1_PAHHH);
    /* DLP93 [PAHHHL]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_PAHHHL, h1_PAHHHL, u2_PAHHHL;
    fp_null(h0_PAHHHL); fp_new(h0_PAHHHL); fp_null(h1_PAHHHL); fp_new(h1_PAHHHL); fp_null(u2_PAHHHL); fp_new(u2_PAHHHL);
    fp_copy(h0_PAHHHL, u2_PAHHH);
    fp_sqr(h0_PAHHHL, h0_PAHHHL); fp_sqr(h0_PAHHHL, h0_PAHHHL); //countS += 2;
	fp_prime_back(tmp_bn, h0_PAHHHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_PAHHHL) == RLC_EQ) _tmp = _ii;
    uint64_t c_PAHHHLH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_PAHHHL, (one_k << 1) - c_PAHHHLH, gw);
    SELECT(h1_PAHHHL, gpp[94], c_PAHHHLH == 0, h1_PAHHHL);
    //countM++;
    fp_mul(u2_PAHHHL, u2_PAHHH, h1_PAHHHL);
	fp_prime_back(tmp_bn, u2_PAHHHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_PAHHHL) == RLC_EQ) _tmp = _ii;
    uint64_t d_PAHHHLL = (uint64_t)_tmp >> 0;
    uint64_t d_PAHHHL = c_PAHHHLH + ((d_PAHHHLL - 1) % 4) * 2;
    fp_free(h0_PAHHHL); fp_free(h1_PAHHHL); fp_free(u2_PAHHHL);
    uint64_t c_PAHHH = c_PAHHHH + ((d_PAHHHL - 1) % 8) * 8;
    fp_free(h0_PAHHH); fp_free(h1_PAHHH); fp_free(u2_PAHHH);
    GPOW_i84_e6(h1_PAHH, (one_k << 6) - c_PAHHH, gw);
    SELECT(h1_PAHH, gpp[90], c_PAHHH == 0, h1_PAHH);
    //countM++;
    fp_mul(u2_PAHH, h0_PAH, h1_PAHH);
    /* DLP90 [PAHHL]: sq3, DLP93(H), GPOW_i90_e3, DLP93(L) */
    fp_t h0_PAHHL, h1_PAHHL, u2_PAHHL;
    fp_null(h0_PAHHL); fp_new(h0_PAHHL); fp_null(h1_PAHHL); fp_new(h1_PAHHL); fp_null(u2_PAHHL); fp_new(u2_PAHHL);
    fp_copy(h0_PAHHL, u2_PAHH);
    fp_sqr(h0_PAHHL, h0_PAHHL); fp_sqr(h0_PAHHL, h0_PAHHL); fp_sqr(h0_PAHHL, h0_PAHHL); //countS += 3;
    /* DLP93 [PAHHLH]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_PAHHLH, h1_PAHHLH, u2_PAHHLH;
    fp_null(h0_PAHHLH); fp_new(h0_PAHHLH); fp_null(h1_PAHHLH); fp_new(h1_PAHHLH); fp_null(u2_PAHHLH); fp_new(u2_PAHHLH);
    fp_copy(h0_PAHHLH, h0_PAHHL);
    fp_sqr(h0_PAHHLH, h0_PAHHLH); fp_sqr(h0_PAHHLH, h0_PAHHLH); //countS += 2;
	fp_prime_back(tmp_bn, h0_PAHHLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_PAHHLH) == RLC_EQ) _tmp = _ii;
    uint64_t c_PAHHLHH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_PAHHLH, (one_k << 1) - c_PAHHLHH, gw);
    SELECT(h1_PAHHLH, gpp[94], c_PAHHLHH == 0, h1_PAHHLH);
    //countM++;
    fp_mul(u2_PAHHLH, h0_PAHHL, h1_PAHHLH);
	fp_prime_back(tmp_bn, u2_PAHHLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_PAHHLH) == RLC_EQ) _tmp = _ii;
    uint64_t d_PAHHLHL = (uint64_t)_tmp >> 0;
    uint64_t c_PAHHLH = c_PAHHLHH + ((d_PAHHLHL - 1) % 4) * 2;
    fp_free(h0_PAHHLH); fp_free(h1_PAHHLH); fp_free(u2_PAHHLH);
    GPOW_i90_e3(h1_PAHHL, (one_k << 3) - c_PAHHLH, gw);
    SELECT(h1_PAHHL, gpp[93], c_PAHHLH == 0, h1_PAHHL);
    //countM++;
    fp_mul(u2_PAHHL, u2_PAHH, h1_PAHHL);
    /* DLP93 [PAHHLL]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_PAHHLL, h1_PAHHLL, u2_PAHHLL;
    fp_null(h0_PAHHLL); fp_new(h0_PAHHLL); fp_null(h1_PAHHLL); fp_new(h1_PAHHLL); fp_null(u2_PAHHLL); fp_new(u2_PAHHLL);
    fp_copy(h0_PAHHLL, u2_PAHHL);
    fp_sqr(h0_PAHHLL, h0_PAHHLL); fp_sqr(h0_PAHHLL, h0_PAHHLL); //countS += 2;
	fp_prime_back(tmp_bn, h0_PAHHLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_PAHHLL) == RLC_EQ) _tmp = _ii;
    uint64_t c_PAHHLLH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_PAHHLL, (one_k << 1) - c_PAHHLLH, gw);
    SELECT(h1_PAHHLL, gpp[94], c_PAHHLLH == 0, h1_PAHHLL);
    //countM++;
    fp_mul(u2_PAHHLL, u2_PAHHL, h1_PAHHLL);
	fp_prime_back(tmp_bn, u2_PAHHLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_PAHHLL) == RLC_EQ) _tmp = _ii;
    uint64_t d_PAHHLLL = (uint64_t)_tmp >> 0;
    uint64_t d_PAHHLL = c_PAHHLLH + ((d_PAHHLLL - 1) % 4) * 2;
    fp_free(h0_PAHHLL); fp_free(h1_PAHHLL); fp_free(u2_PAHHLL);
    uint64_t d_PAHHL = c_PAHHLH + ((d_PAHHLL - 1) % 8) * 8;
    fp_free(h0_PAHHL); fp_free(h1_PAHHL); fp_free(u2_PAHHL);
    uint64_t c_PAHH = c_PAHHH + ((d_PAHHL - 1) % 64) * 64;
    fp_free(h0_PAHH); fp_free(h1_PAHH); fp_free(u2_PAHH);
    GPOW_i72_e12(h1_PAH, (one_k << 12) - c_PAHH, gw);
    SELECT(h1_PAH, gpp[84], c_PAHH == 0, h1_PAH);
    //countM++;
    fp_mul(u2_PAH, h0_PA, h1_PAH);
    /* DLP84 [PAHL]: sq6, DLP90(H), GPOW_i84_e6, DLP90(L) */
    fp_t h0_PAHL, h1_PAHL, u2_PAHL;
    fp_null(h0_PAHL); fp_new(h0_PAHL); fp_null(h1_PAHL); fp_new(h1_PAHL); fp_null(u2_PAHL); fp_new(u2_PAHL);
    fp_copy(h0_PAHL, u2_PAH);
    for (int _sq = 0; _sq < 6; _sq++) fp_sqr(h0_PAHL, h0_PAHL); //countS += 6;
    /* DLP90 [PAHLH]: sq3, DLP93(H), GPOW_i90_e3, DLP93(L) */
    fp_t h0_PAHLH, h1_PAHLH, u2_PAHLH;
    fp_null(h0_PAHLH); fp_new(h0_PAHLH); fp_null(h1_PAHLH); fp_new(h1_PAHLH); fp_null(u2_PAHLH); fp_new(u2_PAHLH);
    fp_copy(h0_PAHLH, h0_PAHL);
    fp_sqr(h0_PAHLH, h0_PAHLH); fp_sqr(h0_PAHLH, h0_PAHLH); fp_sqr(h0_PAHLH, h0_PAHLH); //countS += 3;
    /* DLP93 [PAHLHH]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_PAHLHH, h1_PAHLHH, u2_PAHLHH;
    fp_null(h0_PAHLHH); fp_new(h0_PAHLHH); fp_null(h1_PAHLHH); fp_new(h1_PAHLHH); fp_null(u2_PAHLHH); fp_new(u2_PAHLHH);
    fp_copy(h0_PAHLHH, h0_PAHLH);
    fp_sqr(h0_PAHLHH, h0_PAHLHH); fp_sqr(h0_PAHLHH, h0_PAHLHH); //countS += 2;
	fp_prime_back(tmp_bn, h0_PAHLHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_PAHLHH) == RLC_EQ) _tmp = _ii;
    uint64_t c_PAHLHHH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_PAHLHH, (one_k << 1) - c_PAHLHHH, gw);
    SELECT(h1_PAHLHH, gpp[94], c_PAHLHHH == 0, h1_PAHLHH);
    //countM++;
    fp_mul(u2_PAHLHH, h0_PAHLH, h1_PAHLHH);
	fp_prime_back(tmp_bn, u2_PAHLHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_PAHLHH) == RLC_EQ) _tmp = _ii;
    uint64_t d_PAHLHHL = (uint64_t)_tmp >> 0;
    uint64_t c_PAHLHH = c_PAHLHHH + ((d_PAHLHHL - 1) % 4) * 2;
    fp_free(h0_PAHLHH); fp_free(h1_PAHLHH); fp_free(u2_PAHLHH);
    GPOW_i90_e3(h1_PAHLH, (one_k << 3) - c_PAHLHH, gw);
    SELECT(h1_PAHLH, gpp[93], c_PAHLHH == 0, h1_PAHLH);
    //countM++;
    fp_mul(u2_PAHLH, h0_PAHL, h1_PAHLH);
    /* DLP93 [PAHLHL]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_PAHLHL, h1_PAHLHL, u2_PAHLHL;
    fp_null(h0_PAHLHL); fp_new(h0_PAHLHL); fp_null(h1_PAHLHL); fp_new(h1_PAHLHL); fp_null(u2_PAHLHL); fp_new(u2_PAHLHL);
    fp_copy(h0_PAHLHL, u2_PAHLH);
    fp_sqr(h0_PAHLHL, h0_PAHLHL); fp_sqr(h0_PAHLHL, h0_PAHLHL); //countS += 2;
	fp_prime_back(tmp_bn, h0_PAHLHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_PAHLHL) == RLC_EQ) _tmp = _ii;
    uint64_t c_PAHLHLH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_PAHLHL, (one_k << 1) - c_PAHLHLH, gw);
    SELECT(h1_PAHLHL, gpp[94], c_PAHLHLH == 0, h1_PAHLHL);
    //countM++;
    fp_mul(u2_PAHLHL, u2_PAHLH, h1_PAHLHL);
	fp_prime_back(tmp_bn, u2_PAHLHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_PAHLHL) == RLC_EQ) _tmp = _ii;
    uint64_t d_PAHLHLL = (uint64_t)_tmp >> 0;
    uint64_t d_PAHLHL = c_PAHLHLH + ((d_PAHLHLL - 1) % 4) * 2;
    fp_free(h0_PAHLHL); fp_free(h1_PAHLHL); fp_free(u2_PAHLHL);
    uint64_t c_PAHLH = c_PAHLHH + ((d_PAHLHL - 1) % 8) * 8;
    fp_free(h0_PAHLH); fp_free(h1_PAHLH); fp_free(u2_PAHLH);
    GPOW_i84_e6(h1_PAHL, (one_k << 6) - c_PAHLH, gw);
    SELECT(h1_PAHL, gpp[90], c_PAHLH == 0, h1_PAHL);
    //countM++;
    fp_mul(u2_PAHL, u2_PAH, h1_PAHL);
    /* DLP90 [PAHLL]: sq3, DLP93(H), GPOW_i90_e3, DLP93(L) */
    fp_t h0_PAHLL, h1_PAHLL, u2_PAHLL;
    fp_null(h0_PAHLL); fp_new(h0_PAHLL); fp_null(h1_PAHLL); fp_new(h1_PAHLL); fp_null(u2_PAHLL); fp_new(u2_PAHLL);
    fp_copy(h0_PAHLL, u2_PAHL);
    fp_sqr(h0_PAHLL, h0_PAHLL); fp_sqr(h0_PAHLL, h0_PAHLL); fp_sqr(h0_PAHLL, h0_PAHLL); //countS += 3;
    /* DLP93 [PAHLLH]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_PAHLLH, h1_PAHLLH, u2_PAHLLH;
    fp_null(h0_PAHLLH); fp_new(h0_PAHLLH); fp_null(h1_PAHLLH); fp_new(h1_PAHLLH); fp_null(u2_PAHLLH); fp_new(u2_PAHLLH);
    fp_copy(h0_PAHLLH, h0_PAHLL);
    fp_sqr(h0_PAHLLH, h0_PAHLLH); fp_sqr(h0_PAHLLH, h0_PAHLLH); //countS += 2;
	fp_prime_back(tmp_bn, h0_PAHLLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_PAHLLH) == RLC_EQ) _tmp = _ii;
    uint64_t c_PAHLLHH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_PAHLLH, (one_k << 1) - c_PAHLLHH, gw);
    SELECT(h1_PAHLLH, gpp[94], c_PAHLLHH == 0, h1_PAHLLH);
    //countM++;
    fp_mul(u2_PAHLLH, h0_PAHLL, h1_PAHLLH);
	fp_prime_back(tmp_bn, u2_PAHLLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_PAHLLH) == RLC_EQ) _tmp = _ii;
    uint64_t d_PAHLLHL = (uint64_t)_tmp >> 0;
    uint64_t c_PAHLLH = c_PAHLLHH + ((d_PAHLLHL - 1) % 4) * 2;
    fp_free(h0_PAHLLH); fp_free(h1_PAHLLH); fp_free(u2_PAHLLH);
    GPOW_i90_e3(h1_PAHLL, (one_k << 3) - c_PAHLLH, gw);
    SELECT(h1_PAHLL, gpp[93], c_PAHLLH == 0, h1_PAHLL);
    //countM++;
    fp_mul(u2_PAHLL, u2_PAHL, h1_PAHLL);
    /* DLP93 [PAHLLL]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_PAHLLL, h1_PAHLLL, u2_PAHLLL;
    fp_null(h0_PAHLLL); fp_new(h0_PAHLLL); fp_null(h1_PAHLLL); fp_new(h1_PAHLLL); fp_null(u2_PAHLLL); fp_new(u2_PAHLLL);
    fp_copy(h0_PAHLLL, u2_PAHLL);
    fp_sqr(h0_PAHLLL, h0_PAHLLL); fp_sqr(h0_PAHLLL, h0_PAHLLL); //countS += 2;
	fp_prime_back(tmp_bn, h0_PAHLLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_PAHLLL) == RLC_EQ) _tmp = _ii;
    uint64_t c_PAHLLLH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_PAHLLL, (one_k << 1) - c_PAHLLLH, gw);
    SELECT(h1_PAHLLL, gpp[94], c_PAHLLLH == 0, h1_PAHLLL);
    //countM++;
    fp_mul(u2_PAHLLL, u2_PAHLL, h1_PAHLLL);
	fp_prime_back(tmp_bn, u2_PAHLLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_PAHLLL) == RLC_EQ) _tmp = _ii;
    uint64_t d_PAHLLLL = (uint64_t)_tmp >> 0;
    uint64_t d_PAHLLL = c_PAHLLLH + ((d_PAHLLLL - 1) % 4) * 2;
    fp_free(h0_PAHLLL); fp_free(h1_PAHLLL); fp_free(u2_PAHLLL);
    uint64_t d_PAHLL = c_PAHLLH + ((d_PAHLLL - 1) % 8) * 8;
    fp_free(h0_PAHLL); fp_free(h1_PAHLL); fp_free(u2_PAHLL);
    uint64_t d_PAHL = c_PAHLH + ((d_PAHLL - 1) % 64) * 64;
    fp_free(h0_PAHL); fp_free(h1_PAHL); fp_free(u2_PAHL);
    uint64_t c_PAH = c_PAHH + ((d_PAHL - 1) % 4096) * 4096;
    fp_free(h0_PAH); fp_free(h1_PAH); fp_free(u2_PAH);
    GPOW_i48_e24(h1_PA, (one_k << 24) - c_PAH, gw);
    SELECT(h1_PA, gpp[72], c_PAH == 0, h1_PA);
    //countM++;
    fp_mul(u2_PA, h0_A, h1_PA);
    /* DLP72 [PAL]: sq12, DLP84(H), GPOW_i72_e12, DLP84(L) */
    fp_t h0_PAL, h1_PAL, u2_PAL;
    fp_null(h0_PAL); fp_new(h0_PAL); fp_null(h1_PAL); fp_new(h1_PAL); fp_null(u2_PAL); fp_new(u2_PAL);
    fp_copy(h0_PAL, u2_PA);
    for (int _sq = 0; _sq < 12; _sq++) fp_sqr(h0_PAL, h0_PAL); //countS += 12;
    /* DLP84 [PALH]: sq6, DLP90(H), GPOW_i84_e6, DLP90(L) */
    fp_t h0_PALH, h1_PALH, u2_PALH;
    fp_null(h0_PALH); fp_new(h0_PALH); fp_null(h1_PALH); fp_new(h1_PALH); fp_null(u2_PALH); fp_new(u2_PALH);
    fp_copy(h0_PALH, h0_PAL);
    for (int _sq = 0; _sq < 6; _sq++) fp_sqr(h0_PALH, h0_PALH); //countS += 6;
    /* DLP90 [PALHH]: sq3, DLP93(H), GPOW_i90_e3, DLP93(L) */
    fp_t h0_PALHH, h1_PALHH, u2_PALHH;
    fp_null(h0_PALHH); fp_new(h0_PALHH); fp_null(h1_PALHH); fp_new(h1_PALHH); fp_null(u2_PALHH); fp_new(u2_PALHH);
    fp_copy(h0_PALHH, h0_PALH);
    fp_sqr(h0_PALHH, h0_PALHH); fp_sqr(h0_PALHH, h0_PALHH); fp_sqr(h0_PALHH, h0_PALHH); //countS += 3;
    /* DLP93 [PALHHH]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_PALHHH, h1_PALHHH, u2_PALHHH;
    fp_null(h0_PALHHH); fp_new(h0_PALHHH); fp_null(h1_PALHHH); fp_new(h1_PALHHH); fp_null(u2_PALHHH); fp_new(u2_PALHHH);
    fp_copy(h0_PALHHH, h0_PALHH);
    fp_sqr(h0_PALHHH, h0_PALHHH); fp_sqr(h0_PALHHH, h0_PALHHH); //countS += 2;
	fp_prime_back(tmp_bn, h0_PALHHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_PALHHH) == RLC_EQ) _tmp = _ii;
    uint64_t c_PALHHHH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_PALHHH, (one_k << 1) - c_PALHHHH, gw);
    SELECT(h1_PALHHH, gpp[94], c_PALHHHH == 0, h1_PALHHH);
    //countM++;
    fp_mul(u2_PALHHH, h0_PALHH, h1_PALHHH);
	fp_prime_back(tmp_bn, u2_PALHHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_PALHHH) == RLC_EQ) _tmp = _ii;
    uint64_t d_PALHHHL = (uint64_t)_tmp >> 0;
    uint64_t c_PALHHH = c_PALHHHH + ((d_PALHHHL - 1) % 4) * 2;
    fp_free(h0_PALHHH); fp_free(h1_PALHHH); fp_free(u2_PALHHH);
    GPOW_i90_e3(h1_PALHH, (one_k << 3) - c_PALHHH, gw);
    SELECT(h1_PALHH, gpp[93], c_PALHHH == 0, h1_PALHH);
    //countM++;
    fp_mul(u2_PALHH, h0_PALH, h1_PALHH);
    /* DLP93 [PALHHL]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_PALHHL, h1_PALHHL, u2_PALHHL;
    fp_null(h0_PALHHL); fp_new(h0_PALHHL); fp_null(h1_PALHHL); fp_new(h1_PALHHL); fp_null(u2_PALHHL); fp_new(u2_PALHHL);
    fp_copy(h0_PALHHL, u2_PALHH);
    fp_sqr(h0_PALHHL, h0_PALHHL); fp_sqr(h0_PALHHL, h0_PALHHL); //countS += 2;
	fp_prime_back(tmp_bn, h0_PALHHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_PALHHL) == RLC_EQ) _tmp = _ii;
    uint64_t c_PALHHLH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_PALHHL, (one_k << 1) - c_PALHHLH, gw);
    SELECT(h1_PALHHL, gpp[94], c_PALHHLH == 0, h1_PALHHL);
    //countM++;
    fp_mul(u2_PALHHL, u2_PALHH, h1_PALHHL);
	fp_prime_back(tmp_bn, u2_PALHHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_PALHHL) == RLC_EQ) _tmp = _ii;
    uint64_t d_PALHHLL = (uint64_t)_tmp >> 0;
    uint64_t d_PALHHL = c_PALHHLH + ((d_PALHHLL - 1) % 4) * 2;
    fp_free(h0_PALHHL); fp_free(h1_PALHHL); fp_free(u2_PALHHL);
    uint64_t c_PALHH = c_PALHHH + ((d_PALHHL - 1) % 8) * 8;
    fp_free(h0_PALHH); fp_free(h1_PALHH); fp_free(u2_PALHH);
    GPOW_i84_e6(h1_PALH, (one_k << 6) - c_PALHH, gw);
    SELECT(h1_PALH, gpp[90], c_PALHH == 0, h1_PALH);
    //countM++;
    fp_mul(u2_PALH, h0_PAL, h1_PALH);
    /* DLP90 [PALHL]: sq3, DLP93(H), GPOW_i90_e3, DLP93(L) */
    fp_t h0_PALHL, h1_PALHL, u2_PALHL;
    fp_null(h0_PALHL); fp_new(h0_PALHL); fp_null(h1_PALHL); fp_new(h1_PALHL); fp_null(u2_PALHL); fp_new(u2_PALHL);
    fp_copy(h0_PALHL, u2_PALH);
    fp_sqr(h0_PALHL, h0_PALHL); fp_sqr(h0_PALHL, h0_PALHL); fp_sqr(h0_PALHL, h0_PALHL); //countS += 3;
    /* DLP93 [PALHLH]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_PALHLH, h1_PALHLH, u2_PALHLH;
    fp_null(h0_PALHLH); fp_new(h0_PALHLH); fp_null(h1_PALHLH); fp_new(h1_PALHLH); fp_null(u2_PALHLH); fp_new(u2_PALHLH);
    fp_copy(h0_PALHLH, h0_PALHL);
    fp_sqr(h0_PALHLH, h0_PALHLH); fp_sqr(h0_PALHLH, h0_PALHLH); //countS += 2;
	fp_prime_back(tmp_bn, h0_PALHLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_PALHLH) == RLC_EQ) _tmp = _ii;
    uint64_t c_PALHLHH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_PALHLH, (one_k << 1) - c_PALHLHH, gw);
    SELECT(h1_PALHLH, gpp[94], c_PALHLHH == 0, h1_PALHLH);
    //countM++;
    fp_mul(u2_PALHLH, h0_PALHL, h1_PALHLH);
	fp_prime_back(tmp_bn, u2_PALHLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_PALHLH) == RLC_EQ) _tmp = _ii;
    uint64_t d_PALHLHL = (uint64_t)_tmp >> 0;
    uint64_t c_PALHLH = c_PALHLHH + ((d_PALHLHL - 1) % 4) * 2;
    fp_free(h0_PALHLH); fp_free(h1_PALHLH); fp_free(u2_PALHLH);
    GPOW_i90_e3(h1_PALHL, (one_k << 3) - c_PALHLH, gw);
    SELECT(h1_PALHL, gpp[93], c_PALHLH == 0, h1_PALHL);
    //countM++;
    fp_mul(u2_PALHL, u2_PALH, h1_PALHL);
    /* DLP93 [PALHLL]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_PALHLL, h1_PALHLL, u2_PALHLL;
    fp_null(h0_PALHLL); fp_new(h0_PALHLL); fp_null(h1_PALHLL); fp_new(h1_PALHLL); fp_null(u2_PALHLL); fp_new(u2_PALHLL);
    fp_copy(h0_PALHLL, u2_PALHL);
    fp_sqr(h0_PALHLL, h0_PALHLL); fp_sqr(h0_PALHLL, h0_PALHLL); //countS += 2;
	fp_prime_back(tmp_bn, h0_PALHLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_PALHLL) == RLC_EQ) _tmp = _ii;
    uint64_t c_PALHLLH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_PALHLL, (one_k << 1) - c_PALHLLH, gw);
    SELECT(h1_PALHLL, gpp[94], c_PALHLLH == 0, h1_PALHLL);
    //countM++;
    fp_mul(u2_PALHLL, u2_PALHL, h1_PALHLL);
	fp_prime_back(tmp_bn, u2_PALHLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_PALHLL) == RLC_EQ) _tmp = _ii;
    uint64_t d_PALHLLL = (uint64_t)_tmp >> 0;
    uint64_t d_PALHLL = c_PALHLLH + ((d_PALHLLL - 1) % 4) * 2;
    fp_free(h0_PALHLL); fp_free(h1_PALHLL); fp_free(u2_PALHLL);
    uint64_t d_PALHL = c_PALHLH + ((d_PALHLL - 1) % 8) * 8;
    fp_free(h0_PALHL); fp_free(h1_PALHL); fp_free(u2_PALHL);
    uint64_t c_PALH = c_PALHH + ((d_PALHL - 1) % 64) * 64;
    fp_free(h0_PALH); fp_free(h1_PALH); fp_free(u2_PALH);
    GPOW_i72_e12(h1_PAL, (one_k << 12) - c_PALH, gw);
    SELECT(h1_PAL, gpp[84], c_PALH == 0, h1_PAL);
    //countM++;
    fp_mul(u2_PAL, u2_PA, h1_PAL);
    /* DLP84 [PALL]: sq6, DLP90(H), GPOW_i84_e6, DLP90(L) */
    fp_t h0_PALL, h1_PALL, u2_PALL;
    fp_null(h0_PALL); fp_new(h0_PALL); fp_null(h1_PALL); fp_new(h1_PALL); fp_null(u2_PALL); fp_new(u2_PALL);
    fp_copy(h0_PALL, u2_PAL);
    for (int _sq = 0; _sq < 6; _sq++) fp_sqr(h0_PALL, h0_PALL); //countS += 6;
    /* DLP90 [PALLH]: sq3, DLP93(H), GPOW_i90_e3, DLP93(L) */
    fp_t h0_PALLH, h1_PALLH, u2_PALLH;
    fp_null(h0_PALLH); fp_new(h0_PALLH); fp_null(h1_PALLH); fp_new(h1_PALLH); fp_null(u2_PALLH); fp_new(u2_PALLH);
    fp_copy(h0_PALLH, h0_PALL);
    fp_sqr(h0_PALLH, h0_PALLH); fp_sqr(h0_PALLH, h0_PALLH); fp_sqr(h0_PALLH, h0_PALLH); //countS += 3;
    /* DLP93 [PALLHH]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_PALLHH, h1_PALLHH, u2_PALLHH;
    fp_null(h0_PALLHH); fp_new(h0_PALLHH); fp_null(h1_PALLHH); fp_new(h1_PALLHH); fp_null(u2_PALLHH); fp_new(u2_PALLHH);
    fp_copy(h0_PALLHH, h0_PALLH);
    fp_sqr(h0_PALLHH, h0_PALLHH); fp_sqr(h0_PALLHH, h0_PALLHH); //countS += 2;
	fp_prime_back(tmp_bn, h0_PALLHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_PALLHH) == RLC_EQ) _tmp = _ii;
    uint64_t c_PALLHHH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_PALLHH, (one_k << 1) - c_PALLHHH, gw);
    SELECT(h1_PALLHH, gpp[94], c_PALLHHH == 0, h1_PALLHH);
    //countM++;
    fp_mul(u2_PALLHH, h0_PALLH, h1_PALLHH);
	fp_prime_back(tmp_bn, u2_PALLHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_PALLHH) == RLC_EQ) _tmp = _ii;
    uint64_t d_PALLHHL = (uint64_t)_tmp >> 0;
    uint64_t c_PALLHH = c_PALLHHH + ((d_PALLHHL - 1) % 4) * 2;
    fp_free(h0_PALLHH); fp_free(h1_PALLHH); fp_free(u2_PALLHH);
    GPOW_i90_e3(h1_PALLH, (one_k << 3) - c_PALLHH, gw);
    SELECT(h1_PALLH, gpp[93], c_PALLHH == 0, h1_PALLH);
    //countM++;
    fp_mul(u2_PALLH, h0_PALL, h1_PALLH);
    /* DLP93 [PALLHL]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_PALLHL, h1_PALLHL, u2_PALLHL;
    fp_null(h0_PALLHL); fp_new(h0_PALLHL); fp_null(h1_PALLHL); fp_new(h1_PALLHL); fp_null(u2_PALLHL); fp_new(u2_PALLHL);
    fp_copy(h0_PALLHL, u2_PALLH);
    fp_sqr(h0_PALLHL, h0_PALLHL); fp_sqr(h0_PALLHL, h0_PALLHL); //countS += 2;
	fp_prime_back(tmp_bn, h0_PALLHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_PALLHL) == RLC_EQ) _tmp = _ii;
    uint64_t c_PALLHLH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_PALLHL, (one_k << 1) - c_PALLHLH, gw);
    SELECT(h1_PALLHL, gpp[94], c_PALLHLH == 0, h1_PALLHL);
    //countM++;
    fp_mul(u2_PALLHL, u2_PALLH, h1_PALLHL);
	fp_prime_back(tmp_bn, u2_PALLHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_PALLHL) == RLC_EQ) _tmp = _ii;
    uint64_t d_PALLHLL = (uint64_t)_tmp >> 0;
    uint64_t d_PALLHL = c_PALLHLH + ((d_PALLHLL - 1) % 4) * 2;
    fp_free(h0_PALLHL); fp_free(h1_PALLHL); fp_free(u2_PALLHL);
    uint64_t c_PALLH = c_PALLHH + ((d_PALLHL - 1) % 8) * 8;
    fp_free(h0_PALLH); fp_free(h1_PALLH); fp_free(u2_PALLH);
    GPOW_i84_e6(h1_PALL, (one_k << 6) - c_PALLH, gw);
    SELECT(h1_PALL, gpp[90], c_PALLH == 0, h1_PALL);
    //countM++;
    fp_mul(u2_PALL, u2_PAL, h1_PALL);
    /* DLP90 [PALLL]: sq3, DLP93(H), GPOW_i90_e3, DLP93(L) */
    fp_t h0_PALLL, h1_PALLL, u2_PALLL;
    fp_null(h0_PALLL); fp_new(h0_PALLL); fp_null(h1_PALLL); fp_new(h1_PALLL); fp_null(u2_PALLL); fp_new(u2_PALLL);
    fp_copy(h0_PALLL, u2_PALL);
    fp_sqr(h0_PALLL, h0_PALLL); fp_sqr(h0_PALLL, h0_PALLL); fp_sqr(h0_PALLL, h0_PALLL); //countS += 3;
    /* DLP93 [PALLLH]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_PALLLH, h1_PALLLH, u2_PALLLH;
    fp_null(h0_PALLLH); fp_new(h0_PALLLH); fp_null(h1_PALLLH); fp_new(h1_PALLLH); fp_null(u2_PALLLH); fp_new(u2_PALLLH);
    fp_copy(h0_PALLLH, h0_PALLL);
    fp_sqr(h0_PALLLH, h0_PALLLH); fp_sqr(h0_PALLLH, h0_PALLLH); //countS += 2;
	fp_prime_back(tmp_bn, h0_PALLLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_PALLLH) == RLC_EQ) _tmp = _ii;
    uint64_t c_PALLLHH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_PALLLH, (one_k << 1) - c_PALLLHH, gw);
    SELECT(h1_PALLLH, gpp[94], c_PALLLHH == 0, h1_PALLLH);
    //countM++;
    fp_mul(u2_PALLLH, h0_PALLL, h1_PALLLH);
	fp_prime_back(tmp_bn, u2_PALLLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_PALLLH) == RLC_EQ) _tmp = _ii;
    uint64_t d_PALLLHL = (uint64_t)_tmp >> 0;
    uint64_t c_PALLLH = c_PALLLHH + ((d_PALLLHL - 1) % 4) * 2;
    fp_free(h0_PALLLH); fp_free(h1_PALLLH); fp_free(u2_PALLLH);
    GPOW_i90_e3(h1_PALLL, (one_k << 3) - c_PALLLH, gw);
    SELECT(h1_PALLL, gpp[93], c_PALLLH == 0, h1_PALLL);
    //countM++;
    fp_mul(u2_PALLL, u2_PALL, h1_PALLL);
    /* DLP93 [PALLLL]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_PALLLL, h1_PALLLL, u2_PALLLL;
    fp_null(h0_PALLLL); fp_new(h0_PALLLL); fp_null(h1_PALLLL); fp_new(h1_PALLLL); fp_null(u2_PALLLL); fp_new(u2_PALLLL);
    fp_copy(h0_PALLLL, u2_PALLL);
    fp_sqr(h0_PALLLL, h0_PALLLL); fp_sqr(h0_PALLLL, h0_PALLLL); //countS += 2;
	fp_prime_back(tmp_bn, h0_PALLLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_PALLLL) == RLC_EQ) _tmp = _ii;
    uint64_t c_PALLLLH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_PALLLL, (one_k << 1) - c_PALLLLH, gw);
    SELECT(h1_PALLLL, gpp[94], c_PALLLLH == 0, h1_PALLLL);
    //countM++;
    fp_mul(u2_PALLLL, u2_PALLL, h1_PALLLL);
	fp_prime_back(tmp_bn, u2_PALLLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_PALLLL) == RLC_EQ) _tmp = _ii;
    uint64_t d_PALLLLL = (uint64_t)_tmp >> 0;
    uint64_t d_PALLLL = c_PALLLLH + ((d_PALLLLL - 1) % 4) * 2;
    fp_free(h0_PALLLL); fp_free(h1_PALLLL); fp_free(u2_PALLLL);
    uint64_t d_PALLL = c_PALLLH + ((d_PALLLL - 1) % 8) * 8;
    fp_free(h0_PALLL); fp_free(h1_PALLL); fp_free(u2_PALLL);
    uint64_t d_PALL = c_PALLH + ((d_PALLL - 1) % 64) * 64;
    fp_free(h0_PALL); fp_free(h1_PALL); fp_free(u2_PALL);
    uint64_t d_PA = c_PALH + ((d_PALL - 1) % 4096) * 4096;
    fp_free(h0_PAL); fp_free(h1_PAL); fp_free(u2_PAL);
    uint64_t c_A = c_PAH + ((d_PA - 1) % ((one_k<<24))) * (one_k<<24);
    fp_free(h0_PA); fp_free(h1_PA); fp_free(u2_PA); fp_free(h0_A);

    /* ===== f_A: GPOW_i0_e47 [23 muls], no hlp ===== */
    GPOW_i0_e47(f_A, ((one_k << 48) - c_A + 1) >> 1, gw);
    SELECT(f_A, gpp[47], c_A == 0, f_A);
    //countM++; countS++;
    fp_sqr(f_A_sq, f_A); fp_mul(u_X0, u_A, f_A_sq);

    /* ===== EXT2 X0: i=48, sq24, c_X0=DLP72(h0_X0), f_X0=GPOW_i47_e24 ===== */
    fp_t h0_X0; fp_null(h0_X0); fp_new(h0_X0);
    fp_copy(h0_X0, u_X0);
    for (int _j = 0; _j < 24; _j++) fp_sqr(h0_X0, h0_X0); //countS += 24;
    /* DLP72 [X0P]: sq12, DLP84(H), GPOW_i72_e12, DLP84(L) */
    fp_t h0_X0P, h1_X0P, u2_X0P;
    fp_null(h0_X0P); fp_new(h0_X0P); fp_null(h1_X0P); fp_new(h1_X0P); fp_null(u2_X0P); fp_new(u2_X0P);
    fp_copy(h0_X0P, h0_X0);
    for (int _sq = 0; _sq < 12; _sq++) fp_sqr(h0_X0P, h0_X0P); //countS += 12;
    /* DLP84 [X0PH]: sq6, DLP90(H), GPOW_i84_e6, DLP90(L) */
    fp_t h0_X0PH, h1_X0PH, u2_X0PH;
    fp_null(h0_X0PH); fp_new(h0_X0PH); fp_null(h1_X0PH); fp_new(h1_X0PH); fp_null(u2_X0PH); fp_new(u2_X0PH);
    fp_copy(h0_X0PH, h0_X0P);
    for (int _sq = 0; _sq < 6; _sq++) fp_sqr(h0_X0PH, h0_X0PH); //countS += 6;
    /* DLP90 [X0PHH]: sq3, DLP93(H), GPOW_i90_e3, DLP93(L) */
    fp_t h0_X0PHH, h1_X0PHH, u2_X0PHH;
    fp_null(h0_X0PHH); fp_new(h0_X0PHH); fp_null(h1_X0PHH); fp_new(h1_X0PHH); fp_null(u2_X0PHH); fp_new(u2_X0PHH);
    fp_copy(h0_X0PHH, h0_X0PH);
    fp_sqr(h0_X0PHH, h0_X0PHH); fp_sqr(h0_X0PHH, h0_X0PHH); fp_sqr(h0_X0PHH, h0_X0PHH); //countS += 3;
    /* DLP93 [X0PHHH]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_X0PHHH, h1_X0PHHH, u2_X0PHHH;
    fp_null(h0_X0PHHH); fp_new(h0_X0PHHH); fp_null(h1_X0PHHH); fp_new(h1_X0PHHH); fp_null(u2_X0PHHH); fp_new(u2_X0PHHH);
    fp_copy(h0_X0PHHH, h0_X0PHH);
    fp_sqr(h0_X0PHHH, h0_X0PHHH); fp_sqr(h0_X0PHHH, h0_X0PHHH); //countS += 2;
	fp_prime_back(tmp_bn, h0_X0PHHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_X0PHHH) == RLC_EQ) _tmp = _ii;
    uint64_t c_X0PHHHH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_X0PHHH, (one_k << 1) - c_X0PHHHH, gw);
    SELECT(h1_X0PHHH, gpp[94], c_X0PHHHH == 0, h1_X0PHHH);
    //countM++;
    fp_mul(u2_X0PHHH, h0_X0PHH, h1_X0PHHH);
	fp_prime_back(tmp_bn, u2_X0PHHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_X0PHHH) == RLC_EQ) _tmp = _ii;
    uint64_t d_X0PHHHL = (uint64_t)_tmp >> 0;
    uint64_t c_X0PHHH = c_X0PHHHH + ((d_X0PHHHL - 1) % 4) * 2;
    fp_free(h0_X0PHHH); fp_free(h1_X0PHHH); fp_free(u2_X0PHHH);
    GPOW_i90_e3(h1_X0PHH, (one_k << 3) - c_X0PHHH, gw);
    SELECT(h1_X0PHH, gpp[93], c_X0PHHH == 0, h1_X0PHH);
    //countM++;
    fp_mul(u2_X0PHH, h0_X0PH, h1_X0PHH);
    /* DLP93 [X0PHHL]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_X0PHHL, h1_X0PHHL, u2_X0PHHL;
    fp_null(h0_X0PHHL); fp_new(h0_X0PHHL); fp_null(h1_X0PHHL); fp_new(h1_X0PHHL); fp_null(u2_X0PHHL); fp_new(u2_X0PHHL);
    fp_copy(h0_X0PHHL, u2_X0PHH);
    fp_sqr(h0_X0PHHL, h0_X0PHHL); fp_sqr(h0_X0PHHL, h0_X0PHHL); //countS += 2;
	fp_prime_back(tmp_bn, h0_X0PHHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_X0PHHL) == RLC_EQ) _tmp = _ii;
    uint64_t c_X0PHHLH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_X0PHHL, (one_k << 1) - c_X0PHHLH, gw);
    SELECT(h1_X0PHHL, gpp[94], c_X0PHHLH == 0, h1_X0PHHL);
    //countM++;
    fp_mul(u2_X0PHHL, u2_X0PHH, h1_X0PHHL);
	fp_prime_back(tmp_bn, u2_X0PHHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_X0PHHL) == RLC_EQ) _tmp = _ii;
    uint64_t d_X0PHHLL = (uint64_t)_tmp >> 0;
    uint64_t d_X0PHHL = c_X0PHHLH + ((d_X0PHHLL - 1) % 4) * 2;
    fp_free(h0_X0PHHL); fp_free(h1_X0PHHL); fp_free(u2_X0PHHL);
    uint64_t c_X0PHH = c_X0PHHH + ((d_X0PHHL - 1) % 8) * 8;
    fp_free(h0_X0PHH); fp_free(h1_X0PHH); fp_free(u2_X0PHH);
    GPOW_i84_e6(h1_X0PH, (one_k << 6) - c_X0PHH, gw);
    SELECT(h1_X0PH, gpp[90], c_X0PHH == 0, h1_X0PH);
    //countM++;
    fp_mul(u2_X0PH, h0_X0P, h1_X0PH);
    /* DLP90 [X0PHL]: sq3, DLP93(H), GPOW_i90_e3, DLP93(L) */
    fp_t h0_X0PHL, h1_X0PHL, u2_X0PHL;
    fp_null(h0_X0PHL); fp_new(h0_X0PHL); fp_null(h1_X0PHL); fp_new(h1_X0PHL); fp_null(u2_X0PHL); fp_new(u2_X0PHL);
    fp_copy(h0_X0PHL, u2_X0PH);
    fp_sqr(h0_X0PHL, h0_X0PHL); fp_sqr(h0_X0PHL, h0_X0PHL); fp_sqr(h0_X0PHL, h0_X0PHL); //countS += 3;
    /* DLP93 [X0PHLH]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_X0PHLH, h1_X0PHLH, u2_X0PHLH;
    fp_null(h0_X0PHLH); fp_new(h0_X0PHLH); fp_null(h1_X0PHLH); fp_new(h1_X0PHLH); fp_null(u2_X0PHLH); fp_new(u2_X0PHLH);
    fp_copy(h0_X0PHLH, h0_X0PHL);
    fp_sqr(h0_X0PHLH, h0_X0PHLH); fp_sqr(h0_X0PHLH, h0_X0PHLH); //countS += 2;
	fp_prime_back(tmp_bn, h0_X0PHLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_X0PHLH) == RLC_EQ) _tmp = _ii;
    uint64_t c_X0PHLHH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_X0PHLH, (one_k << 1) - c_X0PHLHH, gw);
    SELECT(h1_X0PHLH, gpp[94], c_X0PHLHH == 0, h1_X0PHLH);
    //countM++;
    fp_mul(u2_X0PHLH, h0_X0PHL, h1_X0PHLH);
	fp_prime_back(tmp_bn, u2_X0PHLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_X0PHLH) == RLC_EQ) _tmp = _ii;
    uint64_t d_X0PHLHL = (uint64_t)_tmp >> 0;
    uint64_t c_X0PHLH = c_X0PHLHH + ((d_X0PHLHL - 1) % 4) * 2;
    fp_free(h0_X0PHLH); fp_free(h1_X0PHLH); fp_free(u2_X0PHLH);
    GPOW_i90_e3(h1_X0PHL, (one_k << 3) - c_X0PHLH, gw);
    SELECT(h1_X0PHL, gpp[93], c_X0PHLH == 0, h1_X0PHL);
    //countM++;
    fp_mul(u2_X0PHL, u2_X0PH, h1_X0PHL);
    /* DLP93 [X0PHLL]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_X0PHLL, h1_X0PHLL, u2_X0PHLL;
    fp_null(h0_X0PHLL); fp_new(h0_X0PHLL); fp_null(h1_X0PHLL); fp_new(h1_X0PHLL); fp_null(u2_X0PHLL); fp_new(u2_X0PHLL);
    fp_copy(h0_X0PHLL, u2_X0PHL);
    fp_sqr(h0_X0PHLL, h0_X0PHLL); fp_sqr(h0_X0PHLL, h0_X0PHLL); //countS += 2;
	fp_prime_back(tmp_bn, h0_X0PHLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_X0PHLL) == RLC_EQ) _tmp = _ii;
    uint64_t c_X0PHLLH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_X0PHLL, (one_k << 1) - c_X0PHLLH, gw);
    SELECT(h1_X0PHLL, gpp[94], c_X0PHLLH == 0, h1_X0PHLL);
    //countM++;
    fp_mul(u2_X0PHLL, u2_X0PHL, h1_X0PHLL);
	fp_prime_back(tmp_bn, u2_X0PHLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_X0PHLL) == RLC_EQ) _tmp = _ii;
    uint64_t d_X0PHLLL = (uint64_t)_tmp >> 0;
    uint64_t d_X0PHLL = c_X0PHLLH + ((d_X0PHLLL - 1) % 4) * 2;
    fp_free(h0_X0PHLL); fp_free(h1_X0PHLL); fp_free(u2_X0PHLL);
    uint64_t d_X0PHL = c_X0PHLH + ((d_X0PHLL - 1) % 8) * 8;
    fp_free(h0_X0PHL); fp_free(h1_X0PHL); fp_free(u2_X0PHL);
    uint64_t c_X0PH = c_X0PHH + ((d_X0PHL - 1) % 64) * 64;
    fp_free(h0_X0PH); fp_free(h1_X0PH); fp_free(u2_X0PH);
    GPOW_i72_e12(h1_X0P, (one_k << 12) - c_X0PH, gw);
    SELECT(h1_X0P, gpp[84], c_X0PH == 0, h1_X0P);
    //countM++;
    fp_mul(u2_X0P, h0_X0, h1_X0P);
    /* DLP84 [X0PL]: sq6, DLP90(H), GPOW_i84_e6, DLP90(L) */
    fp_t h0_X0PL, h1_X0PL, u2_X0PL;
    fp_null(h0_X0PL); fp_new(h0_X0PL); fp_null(h1_X0PL); fp_new(h1_X0PL); fp_null(u2_X0PL); fp_new(u2_X0PL);
    fp_copy(h0_X0PL, u2_X0P);
    for (int _sq = 0; _sq < 6; _sq++) fp_sqr(h0_X0PL, h0_X0PL); //countS += 6;
    /* DLP90 [X0PLH]: sq3, DLP93(H), GPOW_i90_e3, DLP93(L) */
    fp_t h0_X0PLH, h1_X0PLH, u2_X0PLH;
    fp_null(h0_X0PLH); fp_new(h0_X0PLH); fp_null(h1_X0PLH); fp_new(h1_X0PLH); fp_null(u2_X0PLH); fp_new(u2_X0PLH);
    fp_copy(h0_X0PLH, h0_X0PL);
    fp_sqr(h0_X0PLH, h0_X0PLH); fp_sqr(h0_X0PLH, h0_X0PLH); fp_sqr(h0_X0PLH, h0_X0PLH); //countS += 3;
    /* DLP93 [X0PLHH]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_X0PLHH, h1_X0PLHH, u2_X0PLHH;
    fp_null(h0_X0PLHH); fp_new(h0_X0PLHH); fp_null(h1_X0PLHH); fp_new(h1_X0PLHH); fp_null(u2_X0PLHH); fp_new(u2_X0PLHH);
    fp_copy(h0_X0PLHH, h0_X0PLH);
    fp_sqr(h0_X0PLHH, h0_X0PLHH); fp_sqr(h0_X0PLHH, h0_X0PLHH); //countS += 2;
	fp_prime_back(tmp_bn, h0_X0PLHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_X0PLHH) == RLC_EQ) _tmp = _ii;
    uint64_t c_X0PLHHH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_X0PLHH, (one_k << 1) - c_X0PLHHH, gw);
    SELECT(h1_X0PLHH, gpp[94], c_X0PLHHH == 0, h1_X0PLHH);
    //countM++;
    fp_mul(u2_X0PLHH, h0_X0PLH, h1_X0PLHH);
	fp_prime_back(tmp_bn, u2_X0PLHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_X0PLHH) == RLC_EQ) _tmp = _ii;
    uint64_t d_X0PLHHL = (uint64_t)_tmp >> 0;
    uint64_t c_X0PLHH = c_X0PLHHH + ((d_X0PLHHL - 1) % 4) * 2;
    fp_free(h0_X0PLHH); fp_free(h1_X0PLHH); fp_free(u2_X0PLHH);
    GPOW_i90_e3(h1_X0PLH, (one_k << 3) - c_X0PLHH, gw);
    SELECT(h1_X0PLH, gpp[93], c_X0PLHH == 0, h1_X0PLH);
    //countM++;
    fp_mul(u2_X0PLH, h0_X0PL, h1_X0PLH);
    /* DLP93 [X0PLHL]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_X0PLHL, h1_X0PLHL, u2_X0PLHL;
    fp_null(h0_X0PLHL); fp_new(h0_X0PLHL); fp_null(h1_X0PLHL); fp_new(h1_X0PLHL); fp_null(u2_X0PLHL); fp_new(u2_X0PLHL);
    fp_copy(h0_X0PLHL, u2_X0PLH);
    fp_sqr(h0_X0PLHL, h0_X0PLHL); fp_sqr(h0_X0PLHL, h0_X0PLHL); //countS += 2;
	fp_prime_back(tmp_bn, h0_X0PLHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_X0PLHL) == RLC_EQ) _tmp = _ii;
    uint64_t c_X0PLHLH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_X0PLHL, (one_k << 1) - c_X0PLHLH, gw);
    SELECT(h1_X0PLHL, gpp[94], c_X0PLHLH == 0, h1_X0PLHL);
    //countM++;
    fp_mul(u2_X0PLHL, u2_X0PLH, h1_X0PLHL);
	fp_prime_back(tmp_bn, u2_X0PLHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_X0PLHL) == RLC_EQ) _tmp = _ii;
    uint64_t d_X0PLHLL = (uint64_t)_tmp >> 0;
    uint64_t d_X0PLHL = c_X0PLHLH + ((d_X0PLHLL - 1) % 4) * 2;
    fp_free(h0_X0PLHL); fp_free(h1_X0PLHL); fp_free(u2_X0PLHL);
    uint64_t c_X0PLH = c_X0PLHH + ((d_X0PLHL - 1) % 8) * 8;
    fp_free(h0_X0PLH); fp_free(h1_X0PLH); fp_free(u2_X0PLH);
    GPOW_i84_e6(h1_X0PL, (one_k << 6) - c_X0PLH, gw);
    SELECT(h1_X0PL, gpp[90], c_X0PLH == 0, h1_X0PL);
    //countM++;
    fp_mul(u2_X0PL, u2_X0P, h1_X0PL);
    /* DLP90 [X0PLL]: sq3, DLP93(H), GPOW_i90_e3, DLP93(L) */
    fp_t h0_X0PLL, h1_X0PLL, u2_X0PLL;
    fp_null(h0_X0PLL); fp_new(h0_X0PLL); fp_null(h1_X0PLL); fp_new(h1_X0PLL); fp_null(u2_X0PLL); fp_new(u2_X0PLL);
    fp_copy(h0_X0PLL, u2_X0PL);
    fp_sqr(h0_X0PLL, h0_X0PLL); fp_sqr(h0_X0PLL, h0_X0PLL); fp_sqr(h0_X0PLL, h0_X0PLL); //countS += 3;
    /* DLP93 [X0PLLH]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_X0PLLH, h1_X0PLLH, u2_X0PLLH;
    fp_null(h0_X0PLLH); fp_new(h0_X0PLLH); fp_null(h1_X0PLLH); fp_new(h1_X0PLLH); fp_null(u2_X0PLLH); fp_new(u2_X0PLLH);
    fp_copy(h0_X0PLLH, h0_X0PLL);
    fp_sqr(h0_X0PLLH, h0_X0PLLH); fp_sqr(h0_X0PLLH, h0_X0PLLH); //countS += 2;
	fp_prime_back(tmp_bn, h0_X0PLLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_X0PLLH) == RLC_EQ) _tmp = _ii;
    uint64_t c_X0PLLHH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_X0PLLH, (one_k << 1) - c_X0PLLHH, gw);
    SELECT(h1_X0PLLH, gpp[94], c_X0PLLHH == 0, h1_X0PLLH);
    //countM++;
    fp_mul(u2_X0PLLH, h0_X0PLL, h1_X0PLLH);
	fp_prime_back(tmp_bn, u2_X0PLLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_X0PLLH) == RLC_EQ) _tmp = _ii;
    uint64_t d_X0PLLHL = (uint64_t)_tmp >> 0;
    uint64_t c_X0PLLH = c_X0PLLHH + ((d_X0PLLHL - 1) % 4) * 2;
    fp_free(h0_X0PLLH); fp_free(h1_X0PLLH); fp_free(u2_X0PLLH);
    GPOW_i90_e3(h1_X0PLL, (one_k << 3) - c_X0PLLH, gw);
    SELECT(h1_X0PLL, gpp[93], c_X0PLLH == 0, h1_X0PLL);
    //countM++;
    fp_mul(u2_X0PLL, u2_X0PL, h1_X0PLL);
    /* DLP93 [X0PLLL]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_X0PLLL, h1_X0PLLL, u2_X0PLLL;
    fp_null(h0_X0PLLL); fp_new(h0_X0PLLL); fp_null(h1_X0PLLL); fp_new(h1_X0PLLL); fp_null(u2_X0PLLL); fp_new(u2_X0PLLL);
    fp_copy(h0_X0PLLL, u2_X0PLL);
    fp_sqr(h0_X0PLLL, h0_X0PLLL); fp_sqr(h0_X0PLLL, h0_X0PLLL); //countS += 2;
	fp_prime_back(tmp_bn, h0_X0PLLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_X0PLLL) == RLC_EQ) _tmp = _ii;
    uint64_t c_X0PLLLH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_X0PLLL, (one_k << 1) - c_X0PLLLH, gw);
    SELECT(h1_X0PLLL, gpp[94], c_X0PLLLH == 0, h1_X0PLLL);
    //countM++;
    fp_mul(u2_X0PLLL, u2_X0PLL, h1_X0PLLL);
	fp_prime_back(tmp_bn, u2_X0PLLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_X0PLLL) == RLC_EQ) _tmp = _ii;
    uint64_t d_X0PLLLL = (uint64_t)_tmp >> 0;
    uint64_t d_X0PLLL = c_X0PLLLH + ((d_X0PLLLL - 1) % 4) * 2;
    fp_free(h0_X0PLLL); fp_free(h1_X0PLLL); fp_free(u2_X0PLLL);
    uint64_t d_X0PLL = c_X0PLLH + ((d_X0PLLL - 1) % 8) * 8;
    fp_free(h0_X0PLL); fp_free(h1_X0PLL); fp_free(u2_X0PLL);
    uint64_t d_X0PL = c_X0PLH + ((d_X0PLL - 1) % 64) * 64;
    fp_free(h0_X0PL); fp_free(h1_X0PL); fp_free(u2_X0PL);
    uint64_t c_X0 = c_X0PH + ((d_X0PL - 1) % 4096) * 4096;
    fp_free(h0_X0P); fp_free(h1_X0P); fp_free(u2_X0P);
    fp_free(h0_X0);
    GPOW_i47_e24(f_X0, (one_k << 24) - c_X0, gw);
    SELECT(f_X0, gpp[71], c_X0 == 0, f_X0);
    //countM++; countS++;
    fp_sqr(f_X0_sq, f_X0); fp_mul(u_X1, u_X0, f_X0_sq);

    /* ===== EXT2 X1: i=72, sq12, c_X1=DLP84(h0_X1), f_X1=GPOW_i71_e12 ===== */
    fp_t h0_X1; fp_null(h0_X1); fp_new(h0_X1);
    fp_copy(h0_X1, u_X1);
    for (int _j = 0; _j < 12; _j++) fp_sqr(h0_X1, h0_X1); //countS += 12;
    /* DLP84 [X1P]: sq6, DLP90(H), GPOW_i84_e6, DLP90(L) */
    fp_t h0_X1P, h1_X1P, u2_X1P;
    fp_null(h0_X1P); fp_new(h0_X1P); fp_null(h1_X1P); fp_new(h1_X1P); fp_null(u2_X1P); fp_new(u2_X1P);
    fp_copy(h0_X1P, h0_X1);
    for (int _sq = 0; _sq < 6; _sq++) fp_sqr(h0_X1P, h0_X1P); //countS += 6;
    /* DLP90 [X1PH]: sq3, DLP93(H), GPOW_i90_e3, DLP93(L) */
    fp_t h0_X1PH, h1_X1PH, u2_X1PH;
    fp_null(h0_X1PH); fp_new(h0_X1PH); fp_null(h1_X1PH); fp_new(h1_X1PH); fp_null(u2_X1PH); fp_new(u2_X1PH);
    fp_copy(h0_X1PH, h0_X1P);
    fp_sqr(h0_X1PH, h0_X1PH); fp_sqr(h0_X1PH, h0_X1PH); fp_sqr(h0_X1PH, h0_X1PH); //countS += 3;
    /* DLP93 [X1PHH]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_X1PHH, h1_X1PHH, u2_X1PHH;
    fp_null(h0_X1PHH); fp_new(h0_X1PHH); fp_null(h1_X1PHH); fp_new(h1_X1PHH); fp_null(u2_X1PHH); fp_new(u2_X1PHH);
    fp_copy(h0_X1PHH, h0_X1PH);
    fp_sqr(h0_X1PHH, h0_X1PHH); fp_sqr(h0_X1PHH, h0_X1PHH); //countS += 2;
	fp_prime_back(tmp_bn, h0_X1PHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_X1PHH) == RLC_EQ) _tmp = _ii;
    uint64_t c_X1PHHH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_X1PHH, (one_k << 1) - c_X1PHHH, gw);
    SELECT(h1_X1PHH, gpp[94], c_X1PHHH == 0, h1_X1PHH);
    //countM++;
    fp_mul(u2_X1PHH, h0_X1PH, h1_X1PHH);
	fp_prime_back(tmp_bn, u2_X1PHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_X1PHH) == RLC_EQ) _tmp = _ii;
    uint64_t d_X1PHHL = (uint64_t)_tmp >> 0;
    uint64_t c_X1PHH = c_X1PHHH + ((d_X1PHHL - 1) % 4) * 2;
    fp_free(h0_X1PHH); fp_free(h1_X1PHH); fp_free(u2_X1PHH);
    GPOW_i90_e3(h1_X1PH, (one_k << 3) - c_X1PHH, gw);
    SELECT(h1_X1PH, gpp[93], c_X1PHH == 0, h1_X1PH);
    //countM++;
    fp_mul(u2_X1PH, h0_X1P, h1_X1PH);
    /* DLP93 [X1PHL]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_X1PHL, h1_X1PHL, u2_X1PHL;
    fp_null(h0_X1PHL); fp_new(h0_X1PHL); fp_null(h1_X1PHL); fp_new(h1_X1PHL); fp_null(u2_X1PHL); fp_new(u2_X1PHL);
    fp_copy(h0_X1PHL, u2_X1PH);
    fp_sqr(h0_X1PHL, h0_X1PHL); fp_sqr(h0_X1PHL, h0_X1PHL); //countS += 2;
	fp_prime_back(tmp_bn, h0_X1PHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_X1PHL) == RLC_EQ) _tmp = _ii;
    uint64_t c_X1PHLH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_X1PHL, (one_k << 1) - c_X1PHLH, gw);
    SELECT(h1_X1PHL, gpp[94], c_X1PHLH == 0, h1_X1PHL);
    //countM++;
    fp_mul(u2_X1PHL, u2_X1PH, h1_X1PHL);
	fp_prime_back(tmp_bn, u2_X1PHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_X1PHL) == RLC_EQ) _tmp = _ii;
    uint64_t d_X1PHLL = (uint64_t)_tmp >> 0;
    uint64_t d_X1PHL = c_X1PHLH + ((d_X1PHLL - 1) % 4) * 2;
    fp_free(h0_X1PHL); fp_free(h1_X1PHL); fp_free(u2_X1PHL);
    uint64_t c_X1PH = c_X1PHH + ((d_X1PHL - 1) % 8) * 8;
    fp_free(h0_X1PH); fp_free(h1_X1PH); fp_free(u2_X1PH);
    GPOW_i84_e6(h1_X1P, (one_k << 6) - c_X1PH, gw);
    SELECT(h1_X1P, gpp[90], c_X1PH == 0, h1_X1P);
    //countM++;
    fp_mul(u2_X1P, h0_X1, h1_X1P);
    /* DLP90 [X1PL]: sq3, DLP93(H), GPOW_i90_e3, DLP93(L) */
    fp_t h0_X1PL, h1_X1PL, u2_X1PL;
    fp_null(h0_X1PL); fp_new(h0_X1PL); fp_null(h1_X1PL); fp_new(h1_X1PL); fp_null(u2_X1PL); fp_new(u2_X1PL);
    fp_copy(h0_X1PL, u2_X1P);
    fp_sqr(h0_X1PL, h0_X1PL); fp_sqr(h0_X1PL, h0_X1PL); fp_sqr(h0_X1PL, h0_X1PL); //countS += 3;
    /* DLP93 [X1PLH]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_X1PLH, h1_X1PLH, u2_X1PLH;
    fp_null(h0_X1PLH); fp_new(h0_X1PLH); fp_null(h1_X1PLH); fp_new(h1_X1PLH); fp_null(u2_X1PLH); fp_new(u2_X1PLH);
    fp_copy(h0_X1PLH, h0_X1PL);
    fp_sqr(h0_X1PLH, h0_X1PLH); fp_sqr(h0_X1PLH, h0_X1PLH); //countS += 2;
	fp_prime_back(tmp_bn, h0_X1PLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_X1PLH) == RLC_EQ) _tmp = _ii;
    uint64_t c_X1PLHH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_X1PLH, (one_k << 1) - c_X1PLHH, gw);
    SELECT(h1_X1PLH, gpp[94], c_X1PLHH == 0, h1_X1PLH);
    //countM++;
    fp_mul(u2_X1PLH, h0_X1PL, h1_X1PLH);
	fp_prime_back(tmp_bn, u2_X1PLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_X1PLH) == RLC_EQ) _tmp = _ii;
    uint64_t d_X1PLHL = (uint64_t)_tmp >> 0;
    uint64_t c_X1PLH = c_X1PLHH + ((d_X1PLHL - 1) % 4) * 2;
    fp_free(h0_X1PLH); fp_free(h1_X1PLH); fp_free(u2_X1PLH);
    GPOW_i90_e3(h1_X1PL, (one_k << 3) - c_X1PLH, gw);
    SELECT(h1_X1PL, gpp[93], c_X1PLH == 0, h1_X1PL);
    //countM++;
    fp_mul(u2_X1PL, u2_X1P, h1_X1PL);
    /* DLP93 [X1PLL]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_X1PLL, h1_X1PLL, u2_X1PLL;
    fp_null(h0_X1PLL); fp_new(h0_X1PLL); fp_null(h1_X1PLL); fp_new(h1_X1PLL); fp_null(u2_X1PLL); fp_new(u2_X1PLL);
    fp_copy(h0_X1PLL, u2_X1PL);
    fp_sqr(h0_X1PLL, h0_X1PLL); fp_sqr(h0_X1PLL, h0_X1PLL); //countS += 2;
	fp_prime_back(tmp_bn, h0_X1PLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_X1PLL) == RLC_EQ) _tmp = _ii;
    uint64_t c_X1PLLH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_X1PLL, (one_k << 1) - c_X1PLLH, gw);
    SELECT(h1_X1PLL, gpp[94], c_X1PLLH == 0, h1_X1PLL);
    //countM++;
    fp_mul(u2_X1PLL, u2_X1PL, h1_X1PLL);
	fp_prime_back(tmp_bn, u2_X1PLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_X1PLL) == RLC_EQ) _tmp = _ii;
    uint64_t d_X1PLLL = (uint64_t)_tmp >> 0;
    uint64_t d_X1PLL = c_X1PLLH + ((d_X1PLLL - 1) % 4) * 2;
    fp_free(h0_X1PLL); fp_free(h1_X1PLL); fp_free(u2_X1PLL);
    uint64_t d_X1PL = c_X1PLH + ((d_X1PLL - 1) % 8) * 8;
    fp_free(h0_X1PL); fp_free(h1_X1PL); fp_free(u2_X1PL);
    uint64_t c_X1 = c_X1PH + ((d_X1PL - 1) % 64) * 64;
    fp_free(h0_X1P); fp_free(h1_X1P); fp_free(u2_X1P);
    fp_free(h0_X1);
    GPOW_i71_e12(f_X1, (one_k << 12) - c_X1, gw);
    SELECT(f_X1, gpp[83], c_X1 == 0, f_X1);
    //countM++; //countS++;
    fp_sqr(f_X1_sq, f_X1); fp_mul(u_X2, u_X1, f_X1_sq);

    /* ===== EXT2 X2: i=84, sq6, c_X2=DLP90(h0_X2), f_X2=GPOW_i83_e6 ===== */
    fp_t h0_X2; fp_null(h0_X2); fp_new(h0_X2);
    fp_copy(h0_X2, u_X2);
    for (int _j = 0; _j < 6; _j++) fp_sqr(h0_X2, h0_X2); //countS += 6;
    /* DLP90 [X2P]: sq3, DLP93(H), GPOW_i90_e3, DLP93(L) */
    fp_t h0_X2P, h1_X2P, u2_X2P;
    fp_null(h0_X2P); fp_new(h0_X2P); fp_null(h1_X2P); fp_new(h1_X2P); fp_null(u2_X2P); fp_new(u2_X2P);
    fp_copy(h0_X2P, h0_X2);
    fp_sqr(h0_X2P, h0_X2P); fp_sqr(h0_X2P, h0_X2P); fp_sqr(h0_X2P, h0_X2P); //countS += 3;
    /* DLP93 [X2PH]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_X2PH, h1_X2PH, u2_X2PH;
    fp_null(h0_X2PH); fp_new(h0_X2PH); fp_null(h1_X2PH); fp_new(h1_X2PH); fp_null(u2_X2PH); fp_new(u2_X2PH);
    fp_copy(h0_X2PH, h0_X2P);
    fp_sqr(h0_X2PH, h0_X2PH); fp_sqr(h0_X2PH, h0_X2PH); //countS += 2;
	fp_prime_back(tmp_bn, h0_X2PH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_X2PH) == RLC_EQ) _tmp = _ii;
    uint64_t c_X2PHH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_X2PH, (one_k << 1) - c_X2PHH, gw);
    SELECT(h1_X2PH, gpp[94], c_X2PHH == 0, h1_X2PH);
    //countM++;
    fp_mul(u2_X2PH, h0_X2P, h1_X2PH);
	fp_prime_back(tmp_bn, u2_X2PH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_X2PH) == RLC_EQ) _tmp = _ii;
    uint64_t d_X2PHL = (uint64_t)_tmp >> 0;
    uint64_t c_X2PH = c_X2PHH + ((d_X2PHL - 1) % 4) * 2;
    fp_free(h0_X2PH); fp_free(h1_X2PH); fp_free(u2_X2PH);
    GPOW_i90_e3(h1_X2P, (one_k << 3) - c_X2PH, gw);
    SELECT(h1_X2P, gpp[93], c_X2PH == 0, h1_X2P);
    //countM++;
    fp_mul(u2_X2P, h0_X2, h1_X2P);
    /* DLP93 [X2PL]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_X2PL, h1_X2PL, u2_X2PL;
    fp_null(h0_X2PL); fp_new(h0_X2PL); fp_null(h1_X2PL); fp_new(h1_X2PL); fp_null(u2_X2PL); fp_new(u2_X2PL);
    fp_copy(h0_X2PL, u2_X2P);
    fp_sqr(h0_X2PL, h0_X2PL); fp_sqr(h0_X2PL, h0_X2PL); //countS += 2;
	fp_prime_back(tmp_bn, h0_X2PL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_X2PL) == RLC_EQ) _tmp = _ii;
    uint64_t c_X2PLH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_X2PL, (one_k << 1) - c_X2PLH, gw);
    SELECT(h1_X2PL, gpp[94], c_X2PLH == 0, h1_X2PL);
    //countM++;
    fp_mul(u2_X2PL, u2_X2P, h1_X2PL);
	fp_prime_back(tmp_bn, u2_X2PL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_X2PL) == RLC_EQ) _tmp = _ii;
    uint64_t d_X2PLL = (uint64_t)_tmp >> 0;
    uint64_t d_X2PL = c_X2PLH + ((d_X2PLL - 1) % 4) * 2;
    fp_free(h0_X2PL); fp_free(h1_X2PL); fp_free(u2_X2PL);
    uint64_t c_X2 = c_X2PH + ((d_X2PL - 1) % 8) * 8;
    fp_free(h0_X2P); fp_free(h1_X2P); fp_free(u2_X2P);
    fp_free(h0_X2);
    GPOW_i83_e6(f_X2, (one_k << 6) - c_X2, gw);
    SELECT(f_X2, gpp[89], c_X2 == 0, f_X2);
    //countM++; countS++;
    fp_sqr(f_X2_sq, f_X2); fp_mul(u_X3, u_X2, f_X2_sq);

    /* ===== EXT2 X3: i=90, sq3, c_X3=DLP93(h0_X3) [HIGH only], f_X3=GPOW_i89_e3 ===== */
    fp_t h0_X3; fp_null(h0_X3); fp_new(h0_X3);
    fp_copy(h0_X3, u_X3);
    fp_sqr(h0_X3, h0_X3); fp_sqr(h0_X3, h0_X3); fp_sqr(h0_X3, h0_X3); //countS += 3;
    /* DLP93 [X3P]: sq2, BASE(95)>>1, GPOW_i93_e1, BASE(94)>>0 */
    fp_t h0_X3P, h1_X3P, u2_X3P;
    fp_null(h0_X3P); fp_new(h0_X3P); fp_null(h1_X3P); fp_new(h1_X3P); fp_null(u2_X3P); fp_new(u2_X3P);
    fp_copy(h0_X3P, h0_X3);
    fp_sqr(h0_X3P, h0_X3P); fp_sqr(h0_X3P, h0_X3P); //countS += 2;
	fp_prime_back(tmp_bn, h0_X3P);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_X3P) == RLC_EQ) _tmp = _ii;
    uint64_t c_X3PH = (uint64_t)_tmp >> 1;
    GPOW_i93_e1(h1_X3P, (one_k << 1) - c_X3PH, gw);
    SELECT(h1_X3P, gpp[94], c_X3PH == 0, h1_X3P);
    //countM++;
    fp_mul(u2_X3P, h0_X3, h1_X3P);
	fp_prime_back(tmp_bn, u2_X3P);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u2_X3P) == RLC_EQ) _tmp = _ii;
    uint64_t d_X3PL = (uint64_t)_tmp >> 0;
    uint64_t c_X3 = c_X3PH + ((d_X3PL - 1) % 4) * 2;
    fp_free(h0_X3P); fp_free(h1_X3P); fp_free(u2_X3P);
    fp_free(h0_X3);
    GPOW_i89_e3(f_X3, (one_k << 3) - c_X3, gw);
    SELECT(f_X3, gpp[92], c_X3 == 0, f_X3);
    //countM++; countS++;
    fp_sqr(f_X3_sq, f_X3); fp_mul(u_X4, u_X3, f_X3_sq);

    /* ===== EXT2 X4: i=93, sq2, c_X4=BASE(95)>>1 [HIGH only], f_X4=GPOW_i92_e1 ===== */
    fp_t h0_X4; fp_null(h0_X4); fp_new(h0_X4);
    fp_copy(h0_X4, u_X4);
    fp_sqr(h0_X4, h0_X4); fp_sqr(h0_X4, h0_X4); //countS += 2;
	fp_prime_back(tmp_bn, h0_X4);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], h0_X4) == RLC_EQ) _tmp = _ii;
    uint64_t c_X4 = (uint64_t)_tmp >> 1;
    fp_free(h0_X4);
    GPOW_i92_e1(f_X4, (one_k << 1) - c_X4, gw);
    SELECT(f_X4, gpp[93], c_X4 == 0, f_X4);
    //countM++; countS++;
    fp_sqr(f_X4_sq, f_X4); fp_mul(u_X5, u_X4, f_X4_sq);

    /* ===== EXT2 BASE: i=94, lb=2=w -> rll>>0, fll[c1<<0] ===== */
	fp_prime_back(tmp_bn, u_X5);
    bn_get_dig(&d, tmp_bn);
    d=d&0xf;
    _tmp=rlll[d];
    //_tmp = 0; for (_ii = 0; _ii < we; _ii++) if (fp_cmp(rll[_ii], u_X5) == RLC_EQ) _tmp = _ii;
    uint64_t c1_X5 = (uint64_t)_tmp >> 0;
    fp_copy(t_X4, fll[c1_X5]);   /* d1_X5 = fll[c1_X5<<0] = fll[c1_X5] */
    //uint64_t e_X5 = c1_X5;

    /* ===== Unwind EXT2 chain ===== */
    fp_mul(t_X4, f_X4, t_X4); //countM++;
    //uint64_t e_X4 = c_X4 + ((e_X5 - 1) %  4) *  2;   /* b=2, a=1 */
    fp_mul(t_X3, f_X3, t_X4); //countM++;
    //uint64_t e_X3 = c_X3 + ((e_X4 - 1) %  8) *  8;   /* b=3, a=3 */
    fp_mul(t_X2, f_X2, t_X3); //countM++;
    //uint64_t e_X2 = c_X2 + ((e_X3 - 1) % 64) * 64;   /* b=6, a=6 */
    fp_mul(t_X1, f_X1, t_X2); //countM++;
    //uint64_t e_X1 = c_X1 + ((e_X2 - 1) % 4096) * 4096;   /* b=12, a=12 */
    fp_mul(t_X0, f_X0, t_X1); //countM++;
    //uint64_t e_X0 = c_X0 + ((e_X1 - 1) % ((one_k<<12))) * (one_k<<12);   /* b=12, a=12 */
    fp_mul(out_t, f_A, t_X0); //countM++;
    //(void)(c_A + ((e_X0 - 1) % ((one_k<<24))) * (one_k<<24));   /* e_final unused */
    //fp_copy(out_t, t_A);

	bn_free(tmp_bn);
    fp_free(f_A);
    fp_free(f_A_sq);
    fp_free(u_X0);
    fp_free(f_X0);
    fp_free(f_X0_sq);
    fp_free(u_X1);
    fp_free(f_X1);
    fp_free(f_X1_sq);
    fp_free(u_X2);
    fp_free(f_X2);
    fp_free(f_X2_sq);
    fp_free(u_X3);
    fp_free(f_X3);
    fp_free(f_X3_sq);
    fp_free(u_X4);
    fp_free(f_X4);
    fp_free(f_X4_sq);
    fp_free(u_X5);
    fp_free(t_X4);
    fp_free(t_X3);
    fp_free(t_X2);
    fp_free(t_X1);
    fp_free(t_X0);
    fp_free(t_A);
}

void sqrt_ext(fp_t x, fp_t y, bn_t e_exp,
              fp_t gw[nw][we],int rlll[16], fp_t rll[we], fp_t fll[we], fp_t gpp[n])
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

void main(void)
{
    if (core_init() != RLC_OK) { core_clean(); return; }
    if (fp_param_set_any_pmers() != RLC_OK) { core_clean(); return; }

    int i, j;
    fp_t gw[nw][we], rll[we], fll[we], gpp[n], g, z, b, h, hh, y;
    bn_t tmp, m, e;
	int rlll[16]={0};
	rlll[ 0x1 ]= 0 ;
	rlll[ 0x9 ]= 1 ;
	rlll[ 0x0 ]= 2 ;
	rlll[ 0x8 ]= 3 ;

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

 //   fp_set_dig(b,4);
//    fp_print(b);

    bn_read_str(tmp, "b", 1, 16);
    fp_prime_conv(z, tmp);
    bn_read_str(m, "ffffffffffffffffffffffffffffffff", 32, 16);
    fp_exp(g, z, m);
    bn_read_str(e, "7fffffffffffffffffffffffffffffff", 32, 16);
    bn_read_str(tmp, "400000000000000000000000", 24, 16);
    fp_exp(h, g, tmp);
    fp_srt(hh, h); fp_inv(hh, hh);
    precomputation(g, h, hh, gw, rll, fll, gpp);

    MEASURE(sqrt_ext(b, y, e, gw,rlll, rll, fll, gpp);)

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
