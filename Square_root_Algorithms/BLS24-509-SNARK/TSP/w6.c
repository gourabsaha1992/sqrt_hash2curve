/*
 * Square root in GF(p),  n=37, w=6 — fully inlined, no recursion, no stack
 *
 * Direct C/RELIC translation of sqrt_n37_w6.sage.
 *
 * Fixed tree for n=37, w=6  (all split points are constants):
 *
 *  DLPpow2ext top (i=0): lb=37, a=18, b=19, nlb1=10, do_hlp=True
 *    square 19, capture hlp_A at j=10
 *    c_A = DLPpow2(h0_A, i=19):                    <- PA block
 *      [PA] lb=18, a=9, b=9, nlb1=5, do_hlp=True
 *        square 9, capture hlp_PA at j=5
 *        c_PA = DLPpow2(h0_PA, i=28):              <- PAH block (no helper)
 *          lb=9, a=4, b=5, do_hlp=False (b=5 <= w=6)
 *          square 5
 *          leaf-H  i=33, lb=4 -> rll lookup, shift >> 2
 *          GPOW(28, 2^4 - c_PAH, 4)                  [1 M]
 *          leaf-L  i=32, lb=5 -> rll lookup, shift >> 1
 *          c_PA = c_PAH + ((d_PAH-1) % 2^5) * 2^4
 *        correct PA: GPOW(19, 2^9-c_PA, 9)           [1 M]
 *                    GPOW(24, 2^9-c_PA, 9)            [1 M]  <- updates hlp_PA
 *        d_PA = DLPpow2(u2_PA, i=28, helper=updated_hlp_PA): <- PAL block
 *          (helper consumed -> h0 = updated_hlp_PA, no squarings)
 *          leaf-H  i=33, lb=4 -> rll lookup, shift >> 2
 *          GPOW(28, 2^4 - c_PAL, 4)                  [1 M]
 *          leaf-L  i=32, lb=5 -> rll lookup, shift >> 1
 *          d_PA = c_PAL + ((d_PAL-1) % 2^5) * 2^4
 *        c_A = c_PA + ((d_PA-1) % 2^9) * 2^9
 *    f_A = GPOW(0, ((2^18-c_A+1)>>1), 17)            [1 M]
 *           GPOW(10, 2^18-c_A, 18)                   [1 M]  <- updates hlp_A
 *    1 extra squaring (f_A^2 inside u_X0)
 *
 *  EXT2 depth 0 (i=18): lb=19, a=9, b=10
 *    helper = updated_hlp_A (consumed) -> h0_X0 = updated_hlp_A, no squarings
 *    c_X0 = DLPpow2(h0_X0, i=28):  same shape as PAH block
 *      square 5
 *      leaf-H  i=33, lb=4 -> rll, shift >> 2
 *      GPOW(28, 2^4-c_X0H, 4)                        [1 M]
 *      leaf-L  i=32, lb=5 -> rll, shift >> 1
 *      c_X0 = c_X0H + ((d_X0H-1) % 2^5) * 2^4
 *    f_X0 = GPOW(17, 2^9-c_X0, 9)                    [1 M]
 *    hlp_X0 = None -> no hlp update passed to depth 1
 *    1 extra squaring
 *
 *  EXT2 depth 1 (i=27): lb=10, a=5, b=5
 *    helper = None; do_hlp=False (b=5 <= w=6) -> square 5, no hlp capture
 *    c_X1 = DLPpow2(h0_X1, i=32): LEAF lb=5, shift >> 1
 *    f_X1 = GPOW(26, 2^5-c_X1, 5)                    [1 M]
 *    1 extra squaring
 *
 *  EXT2 BASE (i=32, lb=5 <= w=6):
 *    u_X2 = u_X1 * f_X1^2
 *    rll lookup -> c1_X2 (shift >> 1);  fll[c1_X2 << 1] -> d1_X2
 *
 *  Unwind:
 *    t_X1    = f_X1 * d1_X2;       e_X1 = c_X1 + ((c1_X2-1) % 2^5)  * 2^5
 *    t_X0    = f_X0 * t_X1;        e_X0 = c_X0 + ((e_X1-1)  % 2^10) * 2^9
 *    t_A     = f_A  * t_X0;     e_final = c_A  + ((e_X0-1)  % 2^19) * 2^18
 *
 * Parameters:
 *   p  = 0x17452A017CBDD682A502A1E13A9D671D27958EECD2C33C3A36C2ADE221D9B956BEA4F49B2B5EE7D7D72AD20065DCB2B8E9CA0015B5152C00000811E000000001
 *   n  = 37,  w = 6,  we = 2^w = 64,  nw = ceil(37/6) = 7
 */

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include <immintrin.h>
#include <relic/relic_core.h>
#include "relic/relic_md.h"

#define w   6
#define we  64          /* 2^6 */
#define n   37
#define nw  7           /* ceil(37/6) */

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
 *
 * Sage:
 *   h  = g^(2^(n-w))          = g^(2^31)
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

/* =========================================================================
 * SELECT(a0, a1, ctl) — same as Sage: returns a1 if ctl else a0
 * ========================================================================= */
static void SELECT(fp_t a0, fp_t a1, bool ctl, fp_t out)
{
    if (ctl) fp_copy(out, a1);
    else     fp_copy(out, a0);
}

/* =========================================================================
 * GPOW helpers — one function per (i, elen) pair that actually appears.
 *
 * General GPOW rule (from Sage):
 *   e &= (1<<elen)-1
 *   ri = i % w;  i_row = i // w
 *   if ri != 0: e <<= ri; elen += ri
 *   t = gw[i_row][e & mask]
 *   while elen > w: e >>= w; i_row++; t *= gw[i_row][e & mask]
 *
 * w=6, mask=0x3F.
 *
 * Instances needed (from Sage tree):
 *   GPOW(28, *, 4)   i=28 ri=4 -> row=4, e<<=4, elen=8 -> 1 table entry, 0 M
 *                    after shift: gw[4][e & 0x3F] only (elen=8 <= 6? No, 8>6)
 *                    row=4: consume 6 bits  (elen=8 -> 8-6=2 > 0 -> next)
 *                    row=5: consume remaining 2 bits  -> 1 M total
 *   GPOW(24, *, 9)   i=24 ri=0 -> row=4, elen=9 -> rows 4,5,6 -> 2 M
 *   GPOW(19, *, 9)   i=19 ri=1 -> row=3, e<<=1, elen=10 -> rows 3,4,5 -> 2 M
 *   GPOW(10, *, 18)  i=10 ri=4 -> row=1, e<<=4, elen=22 -> rows 1,2,3,4 -> 3 M
 *   GPOW(0,  *, 17)  i=0  ri=0 -> row=0, elen=17 -> rows 0,1,2 -> 2 M
 *   GPOW(17, *, 9)   i=17 ri=5 -> row=2, e<<=5, elen=14 -> rows 2,3 -> 1 M
 *   GPOW(26, *, 5)   i=26 ri=2 -> row=4, e<<=2, elen=7 -> rows 4,5 -> 1 M
 * ========================================================================= */

/*
 * GPOW(28, e, 4):
 *   ri=4, e<<=4, elen=8
 *   row=4: gw[4][e & 0x3F]; e>>=6; elen=2
 *   row=5: gw[5][e & 0x3F]; done
 *   -> 1 M
 */
static void GPOW_i28_e4(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 4) - 1;
    e <<= 4;                            /* ri=4 */
    fp_copy(t, gw[4][e & 0x3F]); e >>= 6;
    fp_mul(t, t, gw[5][e & 0x3F]); //countM++;
}

/*
 * GPOW(24, e, 9):
 *   ri=0, elen=9
 *   row=4: gw[4][e & 0x3F]; e>>=6; elen=3
 *   row=5: gw[5][e & 0x3F]; e>>=6; elen=-3 -> done (but we still hit row 5)
 *   Wait: elen=9, first chunk uses 6 bits -> elen-=6 = 3 > 0 -> row 5
 *         row 5: elen-=6 = -3 <= 0 -> done
 *   -> 1 M
 */
static void GPOW_i24_e9(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 9) - 1;
    /* ri=0, i_row=4 */
    fp_copy(t, gw[4][e & 0x3F]); e >>= 6;
    fp_mul(t, t, gw[5][e & 0x3F]); //countM++;
}

/*
 * GPOW(19, e, 9):
 *   ri=1, e<<=1, elen=10
 *   row=3: gw[3][e & 0x3F]; e>>=6; elen=4
 *   row=4: gw[4][e & 0x3F]; e>>=6; elen=-2 -> done
 *   -> 1 M
 */
static void GPOW_i19_e9(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 9) - 1;
    e <<= 1;                            /* ri=1 */
    fp_copy(t, gw[3][e & 0x3F]); e >>= 6;
    fp_mul(t, t, gw[4][e & 0x3F]); //countM++;
}

/*
 * GPOW(10, e, 18):
 *   ri=4, e<<=4, elen=22
 *   row=1: gw[1][e & 0x3F]; e>>=6; elen=16
 *   row=2: gw[2][e & 0x3F]; e>>=6; elen=10
 *   row=3: gw[3][e & 0x3F]; e>>=6; elen=4
 *   row=4: gw[4][e & 0x3F]; e>>=6; elen=-2 -> done
 *   -> 3 M
 */
static void GPOW_i10_e18(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 18) - 1;
    e <<= 4;                            /* ri=4 */
    fp_copy(t, gw[1][e & 0x3F]); e >>= 6;
    fp_mul(t, t, gw[2][e & 0x3F]); //countM++; 
e >>= 6;
    fp_mul(t, t, gw[3][e & 0x3F]); //countM++; 
e >>= 6;
    fp_mul(t, t, gw[4][e & 0x3F]); //countM++;
}

/*
 * GPOW(0, e, 17):
 *   ri=0, elen=17
 *   row=0: gw[0][e & 0x3F]; e>>=6; elen=11
 *   row=1: gw[1][e & 0x3F]; e>>=6; elen=5
 *   row=2: gw[2][e & 0x3F]; e>>=6; elen=-1 -> done
 *   -> 2 M
 */
static void GPOW_i0_e17(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 17) - 1;
    fp_copy(t, gw[0][e & 0x3F]); e >>= 6;
    fp_mul(t, t, gw[1][e & 0x3F]); //countM++; 
e >>= 6;
    fp_mul(t, t, gw[2][e & 0x3F]); //countM++;
}

/*
 * GPOW(17, e, 9):
 *   ri=5, e<<=5, elen=14
 *   row=2: gw[2][e & 0x3F]; e>>=6; elen=8
 *   row=3: gw[3][e & 0x3F]; e>>=6; elen=2 -> done (elen<=0? 2-6=-4 -> done after row3)
 *   Wait: first step elen-=6=8>0 -> row3; elen-=6=-4<=0 -> done
 *   -> 1 M
 */
//static void GPOW_i17_e9(fp_t t, uint64_t e, fp_t gw[nw][we])
//{
//    e &= ((uint64_t)1 << 9) - 1;
//    e <<= 5;                            /* ri=5 */
//    fp_copy(t, gw[2][e & 0x3F]); e >>= 6;
//    fp_mul(t, t, gw[3][e & 0x3F]); countM++;
//}
static void GPOW_i17_e9(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 9) - 1;

    e <<= 5;                            /* ri = 5 */

    fp_copy(t, gw[2][e & 0x3F]);
    e >>= 6;

    fp_mul(t, t, gw[3][e & 0x3F]); 
    //countM++;
    e >>= 6;

    fp_mul(t, t, gw[4][e & 0x3F]); 
    //countM++;
}
/*
 * GPOW(26, e, 5):
 *   ri=2, e<<=2, elen=7
 *   row=4: gw[4][e & 0x3F]; e>>=6; elen=1 -> done (1-6<0)
 *   Wait: first chunk elen-=6=1>0 -> row5; elen-=6=-5<=0 -> done
 *   row=4: gw[4]... ; row=5: gw[5]...
 *   -> 1 M
 */
static void GPOW_i26_e5(fp_t t, uint64_t e, fp_t gw[nw][we])
{
    e &= ((uint64_t)1 << 5) - 1;
    e <<= 2;                            /* ri=2 */
    fp_copy(t, gw[4][e & 0x3F]); e >>= 6;
    fp_mul(t, t, gw[5][e & 0x3F]); //countM++;
}

/* =========================================================================
 * DLPpow2ext — direct translation of sqrt_n37_w6.sage DLPpow2ext.
 * No recursion, no stack.  All variables local (fp_t on RELIC heap).
 * out_t receives t_A on return.
 * ========================================================================= */
void DLPpow2ext(fp_t u_A, fp_t out_t,
                fp_t gw[nw][we], int rlll[4096], fp_t rll[we], fp_t fll[we], fp_t gpp[n])
{
    const uint64_t one_k = 1;

    /* ── Declare all fp_t temporaries ─────────────────────────────────── */
    fp_t h0_A, hlp_A, f_A, f_A_sq;
    fp_t hlp_PA, h0_PA;
    fp_t h0_PAH;
    fp_t h1_PAH, u2_PAH;
    fp_t h1_PA, hlp1_PA, u2_PA;
    fp_t h0_PAL;
    fp_t h1_PAL, u2_PAL;
    fp_t hlp1_A;
    fp_t u_X0, h0_X0;
    fp_t h0_X0H;
    fp_t h1_X0H, u2_X0H;
    fp_t f_X0, f_X0_sq;
    fp_t u_X1, h0_X1;
    fp_t f_X1, f_X1_sq;
    fp_t u_X2;
    fp_t d1_X2;
    fp_t t_X1, t_X0, t_A;

    fp_null(h0_A);    fp_new(h0_A);
    fp_null(hlp_A);   fp_new(hlp_A);
    fp_null(f_A);     fp_new(f_A);
    fp_null(f_A_sq);  fp_new(f_A_sq);
    fp_null(hlp_PA);  fp_new(hlp_PA);
    fp_null(h0_PA);   fp_new(h0_PA);
    fp_null(h0_PAH);  fp_new(h0_PAH);
    fp_null(h1_PAH);  fp_new(h1_PAH);
    fp_null(u2_PAH);  fp_new(u2_PAH);
    fp_null(h1_PA);   fp_new(h1_PA);
    fp_null(hlp1_PA); fp_new(hlp1_PA);
    fp_null(u2_PA);   fp_new(u2_PA);
    fp_null(h0_PAL);  fp_new(h0_PAL);
    fp_null(h1_PAL);  fp_new(h1_PAL);
    fp_null(u2_PAL);  fp_new(u2_PAL);
    fp_null(hlp1_A);  fp_new(hlp1_A);
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
    fp_null(d1_X2);   fp_new(d1_X2);
    fp_null(t_X1);    fp_new(t_X1);
    fp_null(t_X0);    fp_new(t_X0);
    fp_null(t_A);     fp_new(t_A);

    int _ii, _tmp;

    /* =========================================================================
     * TOP: lb=37, a=18, b=19, nlb1=10, do_hlp=True
     * Square 19; capture hlp_A at j=10.
     *
     * Sage:
     *   hlp_A = None
     *   h0_A  = u_A
     *   for _j in range(19):
     *       if _j == 10: hlp_A = h0_A
     *       h0_A *= h0_A
     *   cost["S"] += 19
     * ========================================================================= */
    fp_copy(h0_A, u_A);
    for (int _j = 0; _j < 19; _j++) {
        if (_j == 10) fp_copy(hlp_A, h0_A);
        fp_sqr(h0_A, h0_A);
    }
    //countS += 19;
//    fp_print(h0_A);
    /* =========================================================================
     * PA block: DLPpow2(h0_A, i=19)
     * lb=18, a=9, b=9, nlb1=5, do_hlp=True
     * Square 9; capture hlp_PA at j=5.
     *
     * Sage:
     *   hlp_PA = None
     *   h0_PA  = h0_A
     *   for _j in range(9):
     *       if _j == 5: hlp_PA = h0_PA
     *       h0_PA *= h0_PA
     *   cost["S"] += 9
     * ========================================================================= */
    fp_copy(h0_PA, h0_A);
    for (int _j = 0; _j < 9; _j++) {
        if (_j == 5) fp_copy(hlp_PA, h0_PA);
        fp_sqr(h0_PA, h0_PA);
    }
    //countS += 9;

    /* =========================================================================
     * PAH block: DLPpow2(h0_PA, i=28)
     * lb=9, a=4, b=5, do_hlp=False (b=5 <= w=6)
     * Square 5 (no hlp capture).
     *
     * Sage:
     *   h0_PAH = h0_PA
     *   for _j in range(5): h0_PAH *= h0_PAH
     *   cost["S"] += 5
     * ========================================================================= */
    fp_copy(h0_PAH, h0_PA);
    for (int _j = 0; _j < 5; _j++) fp_sqr(h0_PAH, h0_PAH);
    //countS += 5;

    /* leaf-H: i=33, lb=4, shift = w-lb = 6-4 = 2
     *   for _ii in range(2^w): if rll[_ii] == h0_PAH: _tmp = _ii
     *   c_PAH = _tmp >> 2
     */
	bn_t tmp_bn;
	bn_null(tmp_bn);bn_new(tmp_bn);
	dig_t d;
    fp_prime_back(tmp_bn, h0_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xfff;
    _tmp=rlll[d];
//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_PAH) == RLC_EQ) _tmp = _ii;
    uint64_t c_PAH = (uint64_t)_tmp >> 2;
//    printf("\n%lx\n",c_PAH);
    /* GPOW(28, 2^4 - c_PAH, 4)   [1 M]
     * h1_PAH = SELECT(h1_PAH, gpp[32], c_PAH == 0)
     * u2_PAH = h0_PA * h1_PAH
     */
    GPOW_i28_e4(h1_PAH, (one_k << 4) - c_PAH, gw);
    SELECT(h1_PAH, gpp[32], c_PAH == 0, h1_PAH);
    //countM++;
    fp_mul(u2_PAH, h0_PA, h1_PAH);

    /* leaf-L: i=32, lb=5, shift = w-lb = 6-5 = 1
     *   for _ii in range(2^w): if rll[_ii] == u2_PAH: _tmp = _ii
     *   d_PAH = _tmp >> 1
     *   c_PA  = c_PAH + ((d_PAH - 1) % 2^5) * 2^4
     */
    fp_prime_back(tmp_bn, u2_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0xfff;
    _tmp=rlll[d];
//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_PAH) == RLC_EQ) _tmp = _ii;
    uint64_t d_PAH = (uint64_t)_tmp >> 1;
    uint64_t c_PA  = c_PAH + ((d_PAH - 1) % 32) * 16;
//    printf("\n%lx\n",c_PA);
    /* =========================================================================
     * Correct PA at i=19, a=9.
     * hlp_PA MUST be updated before it is used as PAL's helper.
     *
     * Sage:
     *   h1_PA   = GPOW(19, 2^9 - c_PA, 9)
     *   h1_PA   = SELECT(h1_PA, gpp[28], c_PA == 0)
     *   hlp1_PA = GPOW(24, 2^9 - c_PA, 9)    # i + nlb1 = 19 + 5 = 24
     *   hlp1_PA = SELECT(hlp1_PA, gpp[33], c_PA == 0)
     *   hlp_PA *= hlp1_PA                     # <- update hlp_PA
     *   cost["M"] += 1
     *   cost["M"] += 1
     *   u2_PA = h0_A * h1_PA                  # corrected input for PAL
     * ========================================================================= */
    GPOW_i19_e9(h1_PA,   (one_k << 9) - c_PA, gw);
    SELECT(h1_PA,   gpp[28], c_PA == 0, h1_PA);
    GPOW_i24_e9(hlp1_PA, (one_k << 9) - c_PA, gw);
    SELECT(hlp1_PA, gpp[33], c_PA == 0, hlp1_PA);
    fp_mul(hlp_PA, hlp_PA, hlp1_PA); //countM++;
    //countM++;
    fp_mul(u2_PA, h0_A, h1_PA);

    /* =========================================================================
     * PAL block: DLPpow2(u2_PA, i=28, helper=updated_hlp_PA)
     * Helper consumed -> h0 = updated_hlp_PA, no squarings.
     *
     * Sage:
     *   h0_PAL = hlp_PA            # <- must be the UPDATED value
     *
     *   leaf-H: i=33, lb=4, shift=2
     *   for _ii in range(2^w): if rll[_ii] == h0_PAL: _tmp = _ii
     *   c_PAL = _tmp >> 2
     *   h1_PAL = GPOW(28, 2^4 - c_PAL, 4)
     *   h1_PAL = SELECT(h1_PAL, gpp[32], c_PAL == 0)
     *   cost["M"] += 1
     *   u2_PAL = u2_PA * h1_PAL
     *
     *   leaf-L: i=32, lb=5, shift=1
     *   for _ii in range(2^w): if rll[_ii] == u2_PAL: _tmp = _ii
     *   d_PAL = _tmp >> 1
     *   d_PA  = c_PAL + ((d_PAL - 1) % 2^5) * 2^4
     *
     *   c_A = c_PA + ((d_PA - 1) % 2^9) * 2^9
     * ========================================================================= */
    fp_copy(h0_PAL, hlp_PA);           /* helper consumed: h0 = updated hlp_PA */
    fp_prime_back(tmp_bn, h0_PAL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xfff;
    _tmp=rlll[d];

//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_PAL) == RLC_EQ) _tmp = _ii;
    uint64_t c_PAL = (uint64_t)_tmp >> 2;
//    printf("\n%lx\n",c_PAL);
    GPOW_i28_e4(h1_PAL, (one_k << 4) - c_PAL, gw);
    SELECT(h1_PAL, gpp[32], c_PAL == 0, h1_PAL);
    //countM++;
    fp_mul(u2_PAL, u2_PA, h1_PAL);
	
    fp_prime_back(tmp_bn, u2_PAL);
    bn_get_dig(&d, tmp_bn);
    d=d&0xfff;
    _tmp=rlll[d];
//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_PAL) == RLC_EQ) _tmp = _ii;
    uint64_t d_PAL = (uint64_t)_tmp >> 1;
    uint64_t d_PA  = c_PAL + ((d_PAL - 1) % 32) * 16;

    /* Combine -> c_A is the 18-bit DLP result */
    uint64_t c_A = c_PA + ((d_PA - 1) % 512) * 512;
//    printf("\n%lx\n",c_A);
    /* =========================================================================
     * f_A: top-level DLPpow2ext half-exponent correction.
     * Update hlp_A BEFORE using it as EXT2 depth-0 helper.
     *
     * Sage:
     *   f_A = GPOW(0, ((2^18 - c_A + 1) >> 1), 17)
     *   f_A = SELECT(f_A, gpp[17], c_A == 0)
     *   hlp1_A = GPOW(10, 2^18 - c_A, 18)    # i_A + nlb1_A = 0 + 10 = 10
     *   hlp1_A = SELECT(hlp1_A, gpp[28], c_A == 0)
     *   hlp_A *= hlp1_A                       # <- update hlp_A
     *   cost["M"] += 1
     *   cost["M"] += 1
     *   cost["S"] += 1                        # accounts for f_A^2 in u_X0
     * ========================================================================= */
    GPOW_i0_e17(f_A,    ((one_k << 18) - c_A + 1) >> 1, gw);
    SELECT(f_A,    gpp[17], c_A == 0, f_A);
    GPOW_i10_e18(hlp1_A, (one_k << 18) - c_A, gw);
    SELECT(hlp1_A, gpp[28], c_A == 0, hlp1_A);
    fp_mul(hlp_A, hlp_A, hlp1_A); //countM++;
    //countM++;
    //countS++;   /* f_A^2 inside u_X0 */

    /* =========================================================================
     * EXT2 depth 0: i_X0=18, lb=19, a=9, b=10
     * helper = updated_hlp_A (consumed) -> h0_X0 = updated_hlp_A, no squarings.
     * c_X0 = DLPpow2(h0_X0, i=28): same shape as PAH (sq 5, two rll lookups).
     *
     * Sage:
     *   u_X0  = u_A * f_A^2
     *   h0_X0 = hlp_A              # updated hlp_A consumed; no squarings
     *
     *   h0_X0H = h0_X0
     *   for _j in range(5): h0_X0H *= h0_X0H
     *   cost["S"] += 5
     *
     *   leaf-H: i=33, lb=4, shift=2
     *   c_X0H = _tmp >> 2
     *   GPOW(28, 2^4 - c_X0H, 4)   [1 M]
     *   leaf-L: i=32, lb=5, shift=1
     *   c_X0 = c_X0H + ((d_X0H - 1) % 2^5) * 2^4
     *
     *   f_X0 = GPOW(17, 2^9 - c_X0, 9)   [1 M]
     *   f_X0 = SELECT(f_X0, gpp[26], c_X0 == 0)
     *   cost["S"] += 1                    # f_X0^2 inside u_X1
     * ========================================================================= */
    fp_sqr(f_A_sq, f_A);
    fp_mul(u_X0, u_A, f_A_sq);
    fp_copy(h0_X0, hlp_A);             /* updated hlp_A consumed, no squarings */

    fp_copy(h0_X0H, h0_X0);
    for (int _j = 0; _j < 5; _j++) fp_sqr(h0_X0H, h0_X0H);
    //countS += 5;
	
    fp_prime_back(tmp_bn, h0_X0H);
    bn_get_dig(&d, tmp_bn);
    d=d&0xfff;
    _tmp=rlll[d];
//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_X0H) == RLC_EQ) _tmp = _ii;
    uint64_t c_X0H = (uint64_t)_tmp >> 2;
    
//    printf("\n%lx\n",c_X0H);

    GPOW_i28_e4(h1_X0H, (one_k << 4) - c_X0H, gw);
    SELECT(h1_X0H, gpp[32], c_X0H == 0, h1_X0H);
    //countM++;
    fp_mul(u2_X0H, h0_X0, h1_X0H);
//    fp_print(u2_X0H);
    fp_prime_back(tmp_bn, u2_X0H);
    bn_get_dig(&d, tmp_bn);
    d=d&0xfff;
    _tmp=rlll[d];
//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u2_X0H) == RLC_EQ) _tmp = _ii;
    uint64_t d_X0H = (uint64_t)_tmp >> 1;
    uint64_t c_X0  = c_X0H + ((d_X0H - 1) % 32) * 16;
//    printf("\n%lx\n",c_X0);
    /* f_X0: hlp_X0=None (no hlp update) */
    GPOW_i17_e9(f_X0, (one_k << 9) - c_X0, gw);
//    fp_print(f_X0);
    SELECT(f_X0, gpp[26], c_X0 == 0, f_X0);
    //countM++;
    //countS++;   /* f_X0^2 inside u_X1 */
//    fp_print(f_X0);
    /* =========================================================================
     * EXT2 depth 1: i_X1=27, lb=10, a=5, b=5
     * helper=None; do_hlp=False (b=5 <= w=6) -> square 5, no hlp capture.
     * c_X1 = DLPpow2(h0_X1, i=32): LEAF lb=5, shift = w-lb = 6-5 = 1
     *
     * Sage:
     *   u_X1  = u_X0 * f_X0^2
     *   h0_X1 = u_X1
     *   for _j in range(5): h0_X1 *= h0_X1
     *   cost["S"] += 5
     *
     *   leaf at i=32, lb=5, shift=1:
     *   for _ii in range(2^w): if rll[_ii] == h0_X1: _tmp = _ii
     *   c_X1 = _tmp >> 1
     *
     *   f_X1 = GPOW(26, 2^5 - c_X1, 5)   [1 M]
     *   f_X1 = SELECT(f_X1, gpp[31], c_X1 == 0)
     *   cost["S"] += 1                    # f_X1^2 inside u_X2
     * ========================================================================= */
    fp_sqr(f_X0_sq, f_X0);
    fp_mul(u_X1, u_X0, f_X0_sq);
    fp_copy(h0_X1, u_X1);
    for (int _j = 0; _j < 5; _j++) fp_sqr(h0_X1, h0_X1);
    //countS += 5;
//    fp_print(h0_X1);
    fp_prime_back(tmp_bn, h0_X1);
    bn_get_dig(&d, tmp_bn);
    d=d&0xfff;
    _tmp=rlll[d];
//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], h0_X1) == RLC_EQ) _tmp = _ii;
    uint64_t c_X1 = (uint64_t)_tmp >> 1;
//    printf("\n%lx\n",c_X1);
    /* f_X1: no hlp (hlp_X1=None) */
    GPOW_i26_e5(f_X1, (one_k << 5) - c_X1, gw);
    SELECT(f_X1, gpp[31], c_X1 == 0, f_X1);
    //countM++;
    //countS++;   /* f_X1^2 inside u_X2 */

    /* =========================================================================
     * EXT2 BASE: i_X2=32, lb=5 <= w=6
     * u_X2 = u_X1 * f_X1^2
     * rll lookup -> c1_X2 (shift >> 1);  fll[c1_X2 << 1] -> d1_X2
     *
     * Sage:
     *   u_X2 = u_X1 * f_X1^2
     *
     *   for _ii in range(2^w): if rll[_ii] == u_X2: _tmp = _ii
     *   c1_X2 = _tmp >> 1
     *   d1_X2 = fll[c1_X2 << 1]
     * ========================================================================= */
    fp_sqr(f_X1_sq, f_X1);
    fp_mul(u_X2, u_X1, f_X1_sq);
	
    fp_prime_back(tmp_bn, u_X2);
    bn_get_dig(&d, tmp_bn);
    d=d&0xfff;
    _tmp=rlll[d];
//    _tmp = 0;
//    for (_ii = 0; _ii < we; _ii++)
//        if (fp_cmp(rll[_ii], u_X2) == RLC_EQ) _tmp = _ii;
    uint64_t c1_X2 = (uint64_t)_tmp >> 1;
    fp_copy(d1_X2, fll[c1_X2 << 1]);

    /* =========================================================================
     * Unwind EXT2 chain
     *
     * Sage:
     *   # depth 1 -> base:
     *   t_X1 = f_X1 * d1_X2;       cost["M"] += 1
     *   e_X1 = c_X1 + ((c1_X2 - 1) % 2^5)  * 2^5      b=5, a=5
     *
     *   # depth 0 -> depth 1:
     *   t_X0 = f_X0 * t_X1;        cost["M"] += 1
     *   e_X0 = c_X0 + ((e_X1 - 1) % 2^10) * 2^9        b=10, a=9
     *
     *   # top-level -> depth 0:
     *   t_A     = f_A * t_X0;      cost["M"] += 1
     *   e_final = c_A + ((e_X0 - 1) % 2^19) * 2^18     b=19, a=18
     * ========================================================================= */
    fp_mul(t_X1, f_X1, d1_X2); //countM++;
    //uint64_t e_X1 = c_X1 + ((c1_X2 - 1) % 32)   * 32;

    fp_mul(t_X0, f_X0, t_X1);  //countM++;
    //uint64_t e_X0 = c_X0 + ((e_X1 - 1) % 1024) * 512;

    fp_mul(out_t, f_A, t_X0);    //countM++;
    //uint64_t e_final = c_A + ((e_X0 - 1) % ((uint64_t)1 << 19)) * ((uint64_t)1 << 18);
    //(void)e_final;  /* not used by sqrt_ext caller */

    //fp_copy(out_t, t_A);

    /* ── free all ─────────────────────────────────────────────────────── */
    fp_free(h0_A);    fp_free(hlp_A);   fp_free(f_A);    fp_free(f_A_sq);
    fp_free(hlp_PA);  fp_free(h0_PA);
    fp_free(h0_PAH);
    fp_free(h1_PAH);  fp_free(u2_PAH);
    fp_free(h1_PA);   fp_free(hlp1_PA); fp_free(u2_PA);
    fp_free(h0_PAL);  fp_free(h1_PAL);  fp_free(u2_PAL);
    fp_free(hlp1_A);
    fp_free(u_X0);    fp_free(h0_X0);
    fp_free(h0_X0H);  fp_free(h1_X0H);  fp_free(u2_X0H);
    fp_free(f_X0);    fp_free(f_X0_sq);
    fp_free(u_X1);    fp_free(h0_X1);
    fp_free(f_X1);    fp_free(f_X1_sq);
    fp_free(u_X2);    fp_free(d1_X2);
    fp_free(t_X1);    fp_free(t_X0);    fp_free(t_A);
}

/* =========================================================================
 * sqrt_ext — same as Sage sqrt_ext
 *
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
              fp_t gw[nw][we], int rlll[4096], fp_t rll[we], fp_t fll[we], fp_t gpp[n])
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

/* =========================================================================
 * main
 *
 * p = 0x17452A017CBDD682A502A1E13A9D671D27958EECD2C33C3A36C2ADE221D9B956
 *       BEA4F49B2B5EE7D7D72AD20065DCB2B8E9CA0015B5152C00000811E000000001
 * n = 37, z = 11
 * m = (p-1) / 2^37
 * g = z^m
 * h = g^(2^(n-w)) = g^(2^31)
 * h1 = sqrt(h);  hh = h1^{-1}
 * e_exp = (m-1)/2  used in sqrt_ext
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
	int rlll[4096]={0};

rlll[ 0x1 ]= 0 ;
rlll[ 0x2a6 ]= 1 ;
rlll[ 0x7e4 ]= 2 ;
rlll[ 0x6d ]= 3 ;
rlll[ 0x8c5 ]= 4 ;
rlll[ 0x3e4 ]= 5 ;
rlll[ 0xc24 ]= 6 ;
rlll[ 0x424 ]= 7 ;
rlll[ 0x41d ]= 8 ;
rlll[ 0x18d ]= 9 ;
rlll[ 0x689 ]= 10 ;
rlll[ 0x5eb ]= 11 ;
rlll[ 0x322 ]= 12 ;
rlll[ 0xed1 ]= 13 ;
rlll[ 0x677 ]= 14 ;
rlll[ 0xeb0 ]= 15 ;
rlll[ 0x8e3 ]= 16 ;
rlll[ 0xf02 ]= 17 ;
rlll[ 0xa88 ]= 18 ;
rlll[ 0xfc7 ]= 19 ;
rlll[ 0xe43 ]= 20 ;
rlll[ 0xa43 ]= 21 ;
rlll[ 0x85b ]= 22 ;
rlll[ 0xf1a ]= 23 ;
rlll[ 0x5ec ]= 24 ;
rlll[ 0x4b2 ]= 25 ;
rlll[ 0x3c6 ]= 26 ;
rlll[ 0xdc5 ]= 27 ;
rlll[ 0xdb6 ]= 28 ;
rlll[ 0x810 ]= 29 ;
rlll[ 0x719 ]= 30 ;
rlll[ 0xb62 ]= 31 ;
rlll[ 0x0 ]= 32 ;
rlll[ 0xd5b ]= 33 ;
rlll[ 0x81d ]= 34 ;
rlll[ 0xf94 ]= 35 ;
rlll[ 0x73c ]= 36 ;
rlll[ 0xc1d ]= 37 ;
rlll[ 0x3dd ]= 38 ;
rlll[ 0xbdd ]= 39 ;
rlll[ 0xbe4 ]= 40 ;
rlll[ 0xe74 ]= 41 ;
rlll[ 0x978 ]= 42 ;
rlll[ 0xa16 ]= 43 ;
rlll[ 0xcdf ]= 44 ;
rlll[ 0x130 ]= 45 ;
rlll[ 0x98a ]= 46 ;
rlll[ 0x151 ]= 47 ;
rlll[ 0x71e ]= 48 ;
rlll[ 0xff ]= 49 ;
rlll[ 0x579 ]= 50 ;
rlll[ 0x3a ]= 51 ;
rlll[ 0x1be ]= 52 ;
rlll[ 0x5be ]= 53 ;
rlll[ 0x7a6 ]= 54 ;
rlll[ 0xe7 ]= 55 ;
rlll[ 0xa15 ]= 56 ;
rlll[ 0xb4f ]= 57 ;
rlll[ 0xc3b ]= 58 ;
rlll[ 0x23c ]= 59 ;
rlll[ 0x24b ]= 60 ;
rlll[ 0x7f1 ]= 61 ;
rlll[ 0x8e8 ]= 62 ;
rlll[ 0x49f ]= 63 ;

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

    /* Use a known square for testing: b = 4 */
    fp_rand(b);
    while (fp_is_sqr(b) != 1) fp_rand(b);

    /* z = 11,  g = z^m where m = (p-1)/2^37 */
    bn_read_str(tmp, "b", 1, 16);
    fp_prime_conv(z, tmp);

    /*
     * m = (p-1) / 2^37
     * p-1 = 0x17452A017CBDD682A502A1E13A9D671D27958EECD2C33C3A36C2ADE221D9B956
     *         BEA4F49B2B5EE7D7D72AD20065DCB2B8E9CA0015B5152C00000811DFFFFFFFF80
     *                                                               last byte /2^7
     * Numerically:
     *   m = (p-1) >> 37
     *
     * In hex (128-bit hex string for m, computed offline):
     *   m = 0xBA29500BE5EEB41528151...  (full value stored as hex string)
     *
     * Sage reference:
     *   m = (p - 1) // 2^37
     * We use RELIC bn arithmetic to compute it precisely.
     */
    {
        bn_t p_bn, two37;
        bn_null(p_bn); bn_new(p_bn);
        bn_null(two37); bn_new(two37);

        /* read p */
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

    /*
     * e_exp = (m-1)/2  (used in sqrt_ext as x^e_exp = x^((m-1)/2))
     */
    {
        bn_t one;
        bn_null(one); bn_new(one);
        bn_set_dig(one, 1);
        bn_sub(e, m, one);
        bn_rsh(e, e, 1);   /* e = (m-1)/2 */
        bn_free(one);
    }

    /* h = g^(2^(n-w)) = g^(2^31) */
    {
        bn_t exp_h;
        bn_null(exp_h); bn_new(exp_h);
        bn_set_dig(exp_h, 1);
        bn_lsh(exp_h, exp_h, 31);   /* 2^31 */
        fp_exp(h, g, exp_h);
        bn_free(exp_h);
    }

    /* hh = h1^{-1} where h1 = sqrt(h) */
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
