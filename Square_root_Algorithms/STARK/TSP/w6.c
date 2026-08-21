

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <immintrin.h>


#define LIMB_BITS  60
#define LIMB_MASK  0x0FFFFFFFFFFFFFFFull

typedef struct { uint64_t low; uint64_t high; } uint120_t;

static inline uint120_t make120(uint64_t high, uint64_t low) {
    return (uint120_t){ .low = low & LIMB_MASK, .high = high & LIMB_MASK };
}
static inline uint120_t add120(uint120_t a, uint120_t b) {
    uint64_t raw   = a.low + b.low;
    uint64_t carry = raw >> LIMB_BITS;
    return make120((a.high + b.high + carry) & LIMB_MASK, raw & LIMB_MASK);
}
static inline uint120_t sub120(uint120_t a, uint120_t b) {
    uint64_t raw    = (1ull << LIMB_BITS) + a.low - b.low;
    uint64_t borrow = 1 - (raw >> LIMB_BITS);
    return make120((a.high - b.high - borrow) & LIMB_MASK, raw & LIMB_MASK);
}
static inline uint120_t rshift120(uint120_t a, int n) {
    if (n <= 0)   return a;
    if (n >= 120) return make120(0, 0);
    if (n < LIMB_BITS) {
        uint64_t nl = ((a.low >> n) | (a.high << (LIMB_BITS - n))) & LIMB_MASK;
        uint64_t nh = (a.high >> n) & LIMB_MASK;
        return make120(nh, nl);
    }
    return make120(0, (a.high >> (n - LIMB_BITS)) & LIMB_MASK);
}


#define POW96   make120(1ull << 36, 0)

#define MASK48  0x0000FFFFFFFFFFFFull


#define CHUNK96(e120, k) \
    ( ((k) < 10) \
      ? (uint8_t)((e120).low  >> ((k) * 6)) & 0x3F \
      : (uint8_t)((e120).high >> (((k) - 10) * 6)) & 0x3F )


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


#define W   6
#define WE  64     /* 2^6 */
#define N   192
#define NW  32     /* ceil(192/6) */

int countM = 0, countS = 0;


static void SELECT(fp_t a0, fp_t a1, bool ctl, fp_t out) {
    if (ctl) fp_copy(out, a1); else fp_copy(out, a0);
}
static int rll_lookup(fp_t x, fp_t rll[WE]) {
    for (int k = 0; k < WE; k++)
        if (fp_cmp(rll[k], x) == RLC_EQ) return k;
    return 0;
}


static void GPOW_180_6(fp_t t, uint64_t e, fp_t gw[NW][WE])
{
    e &= (1u << 6) - 1;
    fp_copy(t, gw[30][e & 0x3F]);
}


static void GPOW_174_12(fp_t t, uint64_t e, fp_t gw[NW][WE])
{
    e &= (1u << 12) - 1;
    fp_copy(t, gw[29][e & 0x3F]); e >>= 6;
    fp_mul(t, t, gw[30][e & 0x3F]); 
    //countM++;
}


static void GPOW_168_12(fp_t t, uint64_t e, fp_t gw[NW][WE])
{
    e &= (1u << 12) - 1;
    fp_copy(t, gw[28][e & 0x3F]); e >>= 6;
    fp_mul(t, t, gw[29][e & 0x3F]); 
    //countM++;
}


static void GPOW_156_24(fp_t t, uint64_t e, fp_t gw[NW][WE])
{
    e &= (1ull << 24) - 1;
    fp_copy(t, gw[26][e & 0x3F]); e >>= 6;
    fp_mul(t, t, gw[27][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[28][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[29][e & 0x3F]); 
    //countM++;
}


static void GPOW_144_24(fp_t t, uint64_t e, fp_t gw[NW][WE])
{
    e &= (1ull << 24) - 1;
    fp_copy(t, gw[24][e & 0x3F]); e >>= 6;
    fp_mul(t, t, gw[25][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[26][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[27][e & 0x3F]); 
    //countM++;
}


static void GPOW_120_48(fp_t t, uint64_t e, fp_t gw[NW][WE])
{
    e &= MASK48;
    fp_copy(t, gw[20][e & 0x3F]); e >>= 6;
    fp_mul(t, t, gw[21][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[22][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[23][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[24][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[25][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[26][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[27][e & 0x3F]); 
    //countM++;
}


static void GPOW_96_48(fp_t t, uint64_t e, fp_t gw[NW][WE])
{
    e &= MASK48;
    fp_copy(t, gw[16][e & 0x3F]); e >>= 6;
    fp_mul(t, t, gw[17][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[18][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[19][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[20][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[21][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[22][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[23][e & 0x3F]); 
    //countM++;
}


static void GPOW_0_95(fp_t t, uint120_t e, fp_t gw[NW][WE])
{
    
    e.high &= ~(1ull << 35);
    fp_copy(t, gw[ 0][CHUNK96(e,  0)]);
    fp_mul(t, t, gw[ 1][CHUNK96(e,  1)]); 
    //countM++;
    fp_mul(t, t, gw[ 2][CHUNK96(e,  2)]); 
    //countM++;
    fp_mul(t, t, gw[ 3][CHUNK96(e,  3)]); 
    //countM++;
    fp_mul(t, t, gw[ 4][CHUNK96(e,  4)]); 
    //countM++;
    fp_mul(t, t, gw[ 5][CHUNK96(e,  5)]); 
    //countM++;
    fp_mul(t, t, gw[ 6][CHUNK96(e,  6)]); 
    //countM++;
    fp_mul(t, t, gw[ 7][CHUNK96(e,  7)]); 
    //countM++;
    fp_mul(t, t, gw[ 8][CHUNK96(e,  8)]); 
    //countM++;
    fp_mul(t, t, gw[ 9][CHUNK96(e,  9)]); 
    //countM++;
    fp_mul(t, t, gw[10][CHUNK96(e, 10)]); 
    //countM++;
    fp_mul(t, t, gw[11][CHUNK96(e, 11)]); 
    //countM++;
    fp_mul(t, t, gw[12][CHUNK96(e, 12)]); 
    //countM++;
    fp_mul(t, t, gw[13][CHUNK96(e, 13)]); 
    //countM++;
    fp_mul(t, t, gw[14][CHUNK96(e, 14)]); 
    //countM++;
    fp_mul(t, t, gw[15][CHUNK96(e, 15)]); 
    //countM++;
}


static void GPOW_48_96(fp_t t, uint120_t e, fp_t gw[NW][WE])
{
   
    e.high &= (1ull << 36) - 1;
    fp_copy(t, gw[ 8][CHUNK96(e,  0)]);
    fp_mul(t, t, gw[ 9][CHUNK96(e,  1)]); 
    //countM++;
    fp_mul(t, t, gw[10][CHUNK96(e,  2)]); 
    //countM++;
    fp_mul(t, t, gw[11][CHUNK96(e,  3)]); 
    //countM++;
    fp_mul(t, t, gw[12][CHUNK96(e,  4)]); 
    //countM++;
    fp_mul(t, t, gw[13][CHUNK96(e,  5)]); 
    //countM++;
    fp_mul(t, t, gw[14][CHUNK96(e,  6)]); 
    //countM++;
    fp_mul(t, t, gw[15][CHUNK96(e,  7)]); 
    //countM++;
    fp_mul(t, t, gw[16][CHUNK96(e,  8)]); 
    //countM++;
    fp_mul(t, t, gw[17][CHUNK96(e,  9)]); 
    //countM++;
    fp_mul(t, t, gw[18][CHUNK96(e, 10)]); 
    //countM++;
    fp_mul(t, t, gw[19][CHUNK96(e, 11)]); 
    //countM++;
    fp_mul(t, t, gw[20][CHUNK96(e, 12)]); 
    //countM++;
    fp_mul(t, t, gw[21][CHUNK96(e, 13)]); 
    //countM++;
    fp_mul(t, t, gw[22][CHUNK96(e, 14)]); 
    //countM++;
    fp_mul(t, t, gw[23][CHUNK96(e, 15)]); 
    //countM++;
}


static void GPOW_95_48(fp_t t, uint64_t e, fp_t gw[NW][WE])
{
    e &= MASK48; e <<= 5;
    fp_copy(t, gw[15][e & 0x3F]); e >>= 6;
    fp_mul(t, t, gw[16][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[17][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[18][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[19][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[20][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[21][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[22][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[23][e & 0x3F]); 
    //countM++;
}


static void GPOW_143_24(fp_t t, uint64_t e, fp_t gw[NW][WE])
{
    e &= (1ull << 24) - 1; e <<= 5;
    fp_copy(t, gw[23][e & 0x3F]); e >>= 6;
    fp_mul(t, t, gw[24][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[25][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[26][e & 0x3F]); 
    //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[27][e & 0x3F]); 
    //countM++;
}


static void GPOW_167_12(fp_t t, uint64_t e, fp_t gw[NW][WE])
{
    e &= (1u << 12) - 1; e <<= 5;
    fp_copy(t, gw[27][e & 0x3F]); e >>= 6;
    fp_mul(t, t, gw[28][e & 0x3F]); //countM++; 
    e >>= 6;
    fp_mul(t, t, gw[29][e & 0x3F]); //countM++;
}

static void GPOW_179_6(fp_t t, uint64_t e, fp_t gw[NW][WE])
{
    e &= (1u << 6) - 1; e <<= 5;
    fp_copy(t, gw[29][e & 0x3F]); e >>= 6;
    fp_mul(t, t, gw[30][e & 0x3F]); //countM++;
}

void precomputation(fp_t g, fp_t h, fp_t hh,
                    fp_t gw[NW][WE], fp_t rll[WE], fp_t fll[WE], fp_t gpp[N])
{
    bn_t temp, one;
    bn_null(temp); bn_new(temp);
    bn_null(one);  bn_new(one);
    bn_set_dig(one, 1);

    for (int v = 0; v < WE; v++) {
        bn_set_dig(temp, v);
        fp_exp(rll[v], h,  temp);
        fp_exp(fll[v], hh, temp);
    }
    for (int i = 0; i < NW; i++) {
        for (int j = 0; j < WE; j++) {
            bn_lsh(temp, one, i * W);
            bn_mul_dig(temp, temp, j);
            fp_exp(gw[i][j], g, temp);
        }
    }
    fp_copy(gpp[0], g);
    for (int i = 1; i < N; i++)
        fp_sqr(gpp[i], gpp[i - 1]);

    bn_free(temp);
    bn_free(one);
}


void solve_dlp_pow2_flat(fp_t u_A, fp_t out_d,
                         fp_t gw[NW][WE],int rlll[2048], fp_t rll[WE], fp_t fll[WE],
                         fp_t gpp[N])
{
    
    fp_t h0_A, h0_PA, h0_PAH, h0_PAHH, hlp_A, hlp_PA, hlp_PAH, hlp_PAHH;
    fp_t h1_PAHHH, u2_PAHHH, h1_PAHH, hlp1_PAHH, u2_PAHH;
    fp_t h0_PAHHL, h1_PAHHL, u2_PAHHL;
    fp_t h1_PAH, hlp1_PAH, u2_PAH;
    fp_t h0_PAHL, h0_PAHLH, h1_PAHLH, u2_PAHLH;
    fp_t h1_PAHL, u2_PAHL;
    fp_t h0_PAHLL, h1_PAHLL, u2_PAHLL;
    fp_t h1_PA, hlp1_PA, u2_PA;
    fp_t h0_PAL, hlp_PALH, h0_PALH, h0_PALHH, h1_PALHH, u2_PALHH;
    fp_t h1_PALH, hlp1_PALH, u2_PALH;
    fp_t h0_PALHL, h1_PALHL, u2_PALHL;
    fp_t h1_PAL, u2_PAL;
    fp_t hlp_PALL, h0_PALL, h0_PALLH, h1_PALLH, u2_PALLH;
    fp_t h1_PALL, hlp1_PALL, u2_PALL;
    fp_t h0_PALLL, h1_PALLL, u2_PALLL;
    fp_t f_A, hlp1_A, u_X0;
    fp_t h0_X0, hlp_X0P, h0_X0P, hlp_X0PH, h0_X0PH, h0_X0PHH;
    fp_t h1_X0PHH, u2_X0PHH, h1_X0PH, hlp1_X0PH, u2_X0PH;
    fp_t h0_X0PHL, h1_X0PHL, u2_X0PHL;
    fp_t h1_X0P, hlp1_X0P, u2_X0P;
    fp_t h0_X0PL, h0_X0PLH, h1_X0PLH, u2_X0PLH;
    fp_t h1_X0PL, u2_X0PL;
    fp_t h0_X0PLL, h1_X0PLL, u2_X0PLL;
    fp_t f_X0, u_X1;
    fp_t hlp_X1, h0_X1, hlp_X1H, h0_X1H, h0_X1HH;
    fp_t h1_X1HH, u2_X1HH, h1_X1H, hlp1_X1H, u2_X1H;
    fp_t h0_X1HL, h1_X1HL, u2_X1HL;
    fp_t f_X1, hlp1_X1, u_X2;
    fp_t h0_X2, h0_X2H, h1_X2H, u2_X2H;
    fp_t f_X2, u_X3, h0_X3, f_X3, u_X4;
    fp_t t_X3, t_X2, t_X1, t_X0, t_A;

#define FP_ALLOC(v) fp_null(v); fp_new(v)
    FP_ALLOC(h0_A); FP_ALLOC(h0_PA); FP_ALLOC(h0_PAH); FP_ALLOC(h0_PAHH);
    FP_ALLOC(hlp_A); FP_ALLOC(hlp_PA); FP_ALLOC(hlp_PAH); FP_ALLOC(hlp_PAHH);
    FP_ALLOC(h1_PAHHH); FP_ALLOC(u2_PAHHH); FP_ALLOC(h1_PAHH); FP_ALLOC(hlp1_PAHH); FP_ALLOC(u2_PAHH);
    FP_ALLOC(h0_PAHHL); FP_ALLOC(h1_PAHHL); FP_ALLOC(u2_PAHHL);
    FP_ALLOC(h1_PAH); FP_ALLOC(hlp1_PAH); FP_ALLOC(u2_PAH);
    FP_ALLOC(h0_PAHL); FP_ALLOC(h0_PAHLH); FP_ALLOC(h1_PAHLH); FP_ALLOC(u2_PAHLH);
    FP_ALLOC(h1_PAHL); FP_ALLOC(u2_PAHL);
    FP_ALLOC(h0_PAHLL); FP_ALLOC(h1_PAHLL); FP_ALLOC(u2_PAHLL);
    FP_ALLOC(h1_PA); FP_ALLOC(hlp1_PA); FP_ALLOC(u2_PA);
    FP_ALLOC(h0_PAL); FP_ALLOC(hlp_PALH); FP_ALLOC(h0_PALH); FP_ALLOC(h0_PALHH);
    FP_ALLOC(h1_PALHH); FP_ALLOC(u2_PALHH);
    FP_ALLOC(h1_PALH); FP_ALLOC(hlp1_PALH); FP_ALLOC(u2_PALH);
    FP_ALLOC(h0_PALHL); FP_ALLOC(h1_PALHL); FP_ALLOC(u2_PALHL);
    FP_ALLOC(h1_PAL); FP_ALLOC(u2_PAL);
    FP_ALLOC(hlp_PALL); FP_ALLOC(h0_PALL); FP_ALLOC(h0_PALLH);
    FP_ALLOC(h1_PALLH); FP_ALLOC(u2_PALLH);
    FP_ALLOC(h1_PALL); FP_ALLOC(hlp1_PALL); FP_ALLOC(u2_PALL);
    FP_ALLOC(h0_PALLL); FP_ALLOC(h1_PALLL); FP_ALLOC(u2_PALLL);
    FP_ALLOC(f_A); FP_ALLOC(hlp1_A); FP_ALLOC(u_X0);
    FP_ALLOC(h0_X0); FP_ALLOC(hlp_X0P); FP_ALLOC(h0_X0P); FP_ALLOC(hlp_X0PH);
    FP_ALLOC(h0_X0PH); FP_ALLOC(h0_X0PHH);
    FP_ALLOC(h1_X0PHH); FP_ALLOC(u2_X0PHH); FP_ALLOC(h1_X0PH); FP_ALLOC(hlp1_X0PH); FP_ALLOC(u2_X0PH);
    FP_ALLOC(h0_X0PHL); FP_ALLOC(h1_X0PHL); FP_ALLOC(u2_X0PHL);
    FP_ALLOC(h1_X0P); FP_ALLOC(hlp1_X0P); FP_ALLOC(u2_X0P);
    FP_ALLOC(h0_X0PL); FP_ALLOC(h0_X0PLH); FP_ALLOC(h1_X0PLH); FP_ALLOC(u2_X0PLH);
    FP_ALLOC(h1_X0PL); FP_ALLOC(u2_X0PL);
    FP_ALLOC(h0_X0PLL); FP_ALLOC(h1_X0PLL); FP_ALLOC(u2_X0PLL);
    FP_ALLOC(f_X0); FP_ALLOC(u_X1);
    FP_ALLOC(hlp_X1); FP_ALLOC(h0_X1); FP_ALLOC(hlp_X1H); FP_ALLOC(h0_X1H); FP_ALLOC(h0_X1HH);
    FP_ALLOC(h1_X1HH); FP_ALLOC(u2_X1HH); FP_ALLOC(h1_X1H); FP_ALLOC(hlp1_X1H); FP_ALLOC(u2_X1H);
    FP_ALLOC(h0_X1HL); FP_ALLOC(h1_X1HL); FP_ALLOC(u2_X1HL);
    FP_ALLOC(f_X1); FP_ALLOC(hlp1_X1); FP_ALLOC(u_X2);
    FP_ALLOC(h0_X2); FP_ALLOC(h0_X2H); FP_ALLOC(h1_X2H); FP_ALLOC(u2_X2H);
    FP_ALLOC(f_X2); FP_ALLOC(u_X3); FP_ALLOC(h0_X3); FP_ALLOC(f_X3); FP_ALLOC(u_X4);
    FP_ALLOC(t_X3); FP_ALLOC(t_X2); FP_ALLOC(t_X1); FP_ALLOC(t_X0); FP_ALLOC(t_A);

   
    fp_copy(h0_A, u_A);
    for (int j = 0; j < 96; j++) {
        if (j == 48) fp_copy(hlp_A, h0_A);
        fp_sqr(h0_A, h0_A); //countS++;
    }

   
    fp_copy(h0_PA, h0_A);
    for (int j = 0; j < 48; j++) {
        if (j == 24) fp_copy(hlp_PA, h0_PA);
        fp_sqr(h0_PA, h0_PA); //countS++;
    }

  
    fp_copy(h0_PAH, h0_PA);
    for (int j = 0; j < 24; j++) {
        if (j == 12) fp_copy(hlp_PAH, h0_PAH);
        fp_sqr(h0_PAH, h0_PAH); //countS++;
    }


    fp_copy(h0_PAHH, h0_PAH);
    for (int j = 0; j < 12; j++) {
        if (j == 6) fp_copy(hlp_PAHH, h0_PAHH);
        fp_sqr(h0_PAHH, h0_PAHH); //countS++;
    }

    {
 
    fp_t h0_PAHHH; fp_null(h0_PAHHH); fp_new(h0_PAHHH);
    fp_copy(h0_PAHHH, h0_PAHH);
    for (int j = 0; j < 6; j++) { fp_sqr(h0_PAHHH, h0_PAHHH); //countS++; 
    }
    bn_t tmp_bn;
    bn_null(tmp_bn);bn_new(tmp_bn);
    dig_t d;
    int _tmp;
    fp_prime_back(tmp_bn, h0_PAHHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t c_PAHHH =_tmp;
  
    GPOW_180_6(h1_PAHHH, (1u << 6) - c_PAHHH, gw);
    SELECT(h1_PAHHH, gpp[186], c_PAHHH == 0, h1_PAHHH);
    //countM++;
    fp_mul(u2_PAHHH, h0_PAHH, h1_PAHHH);    /* u=h0_PAHH */
    fp_prime_back(tmp_bn, u2_PAHHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t d_PAHHH  = _tmp;

    uint64_t c_PAHH_H = c_PAHHH + ((d_PAHHH - 1) & 0x3Full) * 64;   /* 12-bit */

    fp_free(h0_PAHHH);

  
    GPOW_168_12(h1_PAHH,   (1u << 12) - c_PAHH_H, gw);
    SELECT(h1_PAHH,   gpp[180], c_PAHH_H == 0, h1_PAHH);
    GPOW_174_12(hlp1_PAHH, (1u << 12) - c_PAHH_H, gw);
    SELECT(hlp1_PAHH, gpp[186], c_PAHH_H == 0, hlp1_PAHH);
    fp_mul(hlp_PAHH, hlp_PAHH, hlp1_PAHH); //countM++; 
    fp_mul(u2_PAHH,  h0_PAH,   h1_PAHH);   //countM++; 


    fp_prime_back(tmp_bn, hlp_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t c_PAHHL_H = _tmp;
   
    GPOW_180_6(h1_PAHHL, (1u << 6) - c_PAHHL_H, gw);
    SELECT(h1_PAHHL, gpp[186], c_PAHHL_H == 0, h1_PAHHL);
    //countM++;
    fp_mul(u2_PAHHL, u2_PAHH, h1_PAHHL);    
    fp_prime_back(tmp_bn, u2_PAHHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t d_PAHHL   = _tmp;

    uint64_t c_PAHH_L  = c_PAHHL_H + ((d_PAHHL - 1) & 0x3Full) * 64;
    uint64_t c_PAHH    = c_PAHH_H  + ((c_PAHH_L - 1) & 0xFFFull) * 4096;  /* 24-bit */

  
    GPOW_144_24(h1_PAH,   (1ull << 24) - c_PAHH, gw);
    SELECT(h1_PAH,   gpp[168], c_PAHH == 0, h1_PAH);
    GPOW_156_24(hlp1_PAH, (1ull << 24) - c_PAHH, gw);
    SELECT(hlp1_PAH, gpp[180], c_PAHH == 0, hlp1_PAH);
    fp_mul(hlp_PAH, hlp_PAH, hlp1_PAH); //countM++;  /* hlp *= */
    fp_mul(u2_PAH,  h0_PA,   h1_PAH);   //countM++;  /* u=h0_PA */


    fp_copy(h0_PAHL, hlp_PAH);


    fp_copy(h0_PAHLH, h0_PAHL);
    for (int j = 0; j < 6; j++) { fp_sqr(h0_PAHLH, h0_PAHLH); //countS++; 
    }
    fp_prime_back(tmp_bn, h0_PAHLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t c_PAHLH_H = _tmp;

    GPOW_180_6(h1_PAHLH, (1u << 6) - c_PAHLH_H, gw);
    SELECT(h1_PAHLH, gpp[186], c_PAHLH_H == 0, h1_PAHLH);
    //countM++;
    fp_mul(u2_PAHLH, h0_PAHL, h1_PAHLH);    /* u=h0_PAHL */
    fp_prime_back(tmp_bn, u2_PAHLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t d_PAHLH  = _tmp;

    uint64_t c_PAHLH  = c_PAHLH_H + ((d_PAHLH - 1) & 0x3Full) * 64;

    GPOW_168_12(h1_PAHL, (1u << 12) - c_PAHLH, gw);
    SELECT(h1_PAHL, gpp[180], c_PAHLH == 0, h1_PAHL);
    fp_mul(u2_PAHL, u2_PAH, h1_PAHL); //countM++;


    fp_copy(h0_PAHLL, u2_PAHL);
    for (int j = 0; j < 6; j++) { fp_sqr(h0_PAHLL, h0_PAHLL); //countS++; 
    }
    fp_prime_back(tmp_bn, h0_PAHLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t c_PAHLL_H = _tmp;

    GPOW_180_6(h1_PAHLL, (1u << 6) - c_PAHLL_H, gw);
    SELECT(h1_PAHLL, gpp[186], c_PAHLL_H == 0, h1_PAHLL);
    //countM++;
    fp_mul(u2_PAHLL, u2_PAHL, h1_PAHLL); 
    fp_prime_back(tmp_bn, u2_PAHLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t d_PAHLL = _tmp;

    uint64_t c_PAHLL = c_PAHLL_H + ((d_PAHLL - 1) & 0x3Full) * 64;
    uint64_t c_PAHL  = c_PAHLH   + ((c_PAHLL - 1) & 0xFFFull) * 4096;  /* 24-bit */

    uint64_t c_PAH_48 = c_PAHH + ((c_PAHL - 1) & 0xFFFFFFull) * (1ull << 24); /* 48-bit */


    GPOW_96_48(h1_PA,   ((1ull << 48) - c_PAH_48) & MASK48, gw);
    SELECT(h1_PA,   gpp[144], c_PAH_48 == 0, h1_PA);
    GPOW_120_48(hlp1_PA, ((1ull << 48) - c_PAH_48) & MASK48, gw);
    SELECT(hlp1_PA, gpp[168], c_PAH_48 == 0, hlp1_PA);
    fp_mul(hlp_PA, hlp_PA, hlp1_PA); //countM++;
    fp_mul(u2_PA,  h0_A,   h1_PA);   //countM++;    /* u=h0_A */


    fp_copy(h0_PAL, hlp_PA);

    fp_copy(h0_PALH, h0_PAL);
    fp_set_dig(hlp_PALH, 1);
    for (int j = 0; j < 12; j++) {
        if (j == 6) fp_copy(hlp_PALH, h0_PALH);
        fp_sqr(h0_PALH, h0_PALH); //countS++;
    }

    fp_copy(h0_PALHH, h0_PALH);
    for (int j = 0; j < 6; j++) { fp_sqr(h0_PALHH, h0_PALHH); //countS++; 
    }
    fp_prime_back(tmp_bn, h0_PALHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t c_PALHH_H = _tmp;

    GPOW_180_6(h1_PALHH, (1u << 6) - c_PALHH_H, gw);
    SELECT(h1_PALHH, gpp[186], c_PALHH_H == 0, h1_PALHH);
    //countM++;
    fp_mul(u2_PALHH, h0_PALH, h1_PALHH);
    fp_prime_back(tmp_bn, u2_PALHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t d_PALHH = _tmp;

    uint64_t c_PALHH = c_PALHH_H + ((d_PALHH - 1) & 0x3Full) * 64;

    GPOW_168_12(h1_PALH,   (1u << 12) - c_PALHH, gw);
    SELECT(h1_PALH,   gpp[180], c_PALHH == 0, h1_PALH);
    GPOW_174_12(hlp1_PALH, (1u << 12) - c_PALHH, gw);
    SELECT(hlp1_PALH, gpp[186], c_PALHH == 0, hlp1_PALH);
    fp_mul(hlp_PALH, hlp_PALH, hlp1_PALH); //countM++;
    fp_mul(u2_PALH,  h0_PAL,   h1_PALH);   //countM++;    /* u=h0_PAL */


    fp_prime_back(tmp_bn, hlp_PALH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t c_PALHL_H = _tmp;
   
    GPOW_180_6(h1_PALHL, (1u << 6) - c_PALHL_H, gw);
    SELECT(h1_PALHL, gpp[186], c_PALHL_H == 0, h1_PALHL);
    //countM++;
    fp_mul(u2_PALHL, u2_PALH, h1_PALHL);    /* u=u2_PALH */
    fp_prime_back(tmp_bn, u2_PALHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t d_PALHL  = _tmp;
    
    uint64_t c_PALHL  = c_PALHL_H + ((d_PALHL - 1) & 0x3Full) * 64;
    uint64_t c_PALH_24 = c_PALHH   + ((c_PALHL - 1) & 0xFFFull) * 4096;  /* 24-bit */

    
    GPOW_144_24(h1_PAL, (1ull << 24) - c_PALH_24, gw);
    SELECT(h1_PAL, gpp[168], c_PALH_24 == 0, h1_PAL);
    fp_mul(u2_PAL, u2_PA, h1_PAL); //countM++;    /* u=u2_PA */

    
    fp_copy(h0_PALL, u2_PAL);
    fp_set_dig(hlp_PALL, 1);
    for (int j = 0; j < 12; j++) {
        if (j == 6) fp_copy(hlp_PALL, h0_PALL);
        fp_sqr(h0_PALL, h0_PALL); //countS++;
    }
   
    fp_copy(h0_PALLH, h0_PALL);
    for (int j = 0; j < 6; j++) { fp_sqr(h0_PALLH, h0_PALLH); //countS++; 
    }
    fp_prime_back(tmp_bn, h0_PALLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t c_PALLH_H = _tmp;
    
    GPOW_180_6(h1_PALLH, (1u << 6) - c_PALLH_H, gw);
    SELECT(h1_PALLH, gpp[186], c_PALLH_H == 0, h1_PALLH);
    //countM++;
    fp_mul(u2_PALLH, h0_PALL, h1_PALLH);
    fp_prime_back(tmp_bn, u2_PALLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t d_PALLH = _tmp;
    
    uint64_t c_PALLH = c_PALLH_H + ((d_PALLH - 1) & 0x3Full) * 64;

    
    GPOW_168_12(h1_PALL,   (1u << 12) - c_PALLH, gw);
    SELECT(h1_PALL,   gpp[180], c_PALLH == 0, h1_PALL);
    GPOW_174_12(hlp1_PALL, (1u << 12) - c_PALLH, gw);
    SELECT(hlp1_PALL, gpp[186], c_PALLH == 0, hlp1_PALL);
    fp_mul(hlp_PALL, hlp_PALL, hlp1_PALL); //countM++;
    fp_mul(u2_PALL,  u2_PAL,   h1_PALL);   //countM++;    /* u=u2_PAL */

    
    fp_prime_back(tmp_bn, hlp_PALL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t c_PALLL_H = _tmp;
    
    GPOW_180_6(h1_PALLL, (1u << 6) - c_PALLL_H, gw);
    SELECT(h1_PALLL, gpp[186], c_PALLL_H == 0, h1_PALLL);
    //countM++;
    fp_mul(u2_PALLL, u2_PALL, h1_PALLL);
    fp_prime_back(tmp_bn, u2_PALLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t d_PALLL   = _tmp;
    
    uint64_t c_PALLL   = c_PALLL_H  + ((d_PALLL - 1) & 0x3Full) * 64;
    uint64_t c_PALL_24 = c_PALLH    + ((c_PALLL - 1) & 0xFFFull) * 4096;

    uint64_t d_PAL_48 = c_PALH_24 + ((c_PALL_24 - 1) & 0xFFFFFFull) * (1ull << 24); 

    
    uint64_t d_PAL_m1 = (d_PAL_48 - 1) & MASK48;
    uint120_t c_A;
    c_A.low  = (c_PAH_48 | ((d_PAL_m1 & 0xFFFull) << 48)) & LIMB_MASK;
    c_A.high = (d_PAL_m1 >> 12) & LIMB_MASK;

    
    uint120_t exp_fA = rshift120(add120(sub120(POW96, c_A), make120(0, 1)), 1);
    GPOW_0_95(f_A, exp_fA, gw);
    bool cA_zero = (c_A.low == 0 && c_A.high == 0);
    SELECT(f_A, gpp[95], cA_zero, f_A);

    uint120_t exp_hlp1A = sub120(POW96, c_A);
    GPOW_48_96(hlp1_A, exp_hlp1A, gw);
    SELECT(hlp1_A, gpp[144], cA_zero, hlp1_A);

    fp_mul(hlp_A, hlp_A, hlp1_A); //countM++;  
    //countM++;                                  
    fp_sqr(u_X0, f_A); //countS++;          
    fp_mul(u_X0, u_X0, u_A);                

   
    fp_copy(h0_X0, hlp_A);

    
    fp_copy(h0_X0P, h0_X0);
    fp_set_dig(hlp_X0P, 1);
    for (int j = 0; j < 24; j++) {
        if (j == 12) fp_copy(hlp_X0P, h0_X0P);
        fp_sqr(h0_X0P, h0_X0P); //countS++;
    }
    
    fp_copy(h0_X0PH, h0_X0P);
    fp_set_dig(hlp_X0PH, 1);
    for (int j = 0; j < 12; j++) {
        if (j == 6) fp_copy(hlp_X0PH, h0_X0PH);
        fp_sqr(h0_X0PH, h0_X0PH); //countS++;
    }
   
    fp_copy(h0_X0PHH, h0_X0PH);
    for (int j = 0; j < 6; j++) { fp_sqr(h0_X0PHH, h0_X0PHH); //countS++; 
    }
    fp_prime_back(tmp_bn, h0_X0PHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t c_X0PHH_H = _tmp;

    GPOW_180_6(h1_X0PHH, (1u << 6) - c_X0PHH_H, gw);
    SELECT(h1_X0PHH, gpp[186], c_X0PHH_H == 0, h1_X0PHH);
    //countM++;
    fp_mul(u2_X0PHH, h0_X0PH, h1_X0PHH);
    fp_prime_back(tmp_bn, u2_X0PHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t d_X0PHH = _tmp;
 
    uint64_t c_X0PHH = c_X0PHH_H + ((d_X0PHH - 1) & 0x3Full) * 64;


    GPOW_168_12(h1_X0PH,   (1u << 12) - c_X0PHH, gw);
    SELECT(h1_X0PH,   gpp[180], c_X0PHH == 0, h1_X0PH);
    GPOW_174_12(hlp1_X0PH, (1u << 12) - c_X0PHH, gw);
    SELECT(hlp1_X0PH, gpp[186], c_X0PHH == 0, hlp1_X0PH);
    fp_mul(hlp_X0PH, hlp_X0PH, hlp1_X0PH); //countM++;
    fp_mul(u2_X0PH,  h0_X0P,   h1_X0PH);   //countM++;

  
    fp_prime_back(tmp_bn, hlp_X0PH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t c_X0PHL_H = _tmp;

    GPOW_180_6(h1_X0PHL, (1u << 6) - c_X0PHL_H, gw);
    SELECT(h1_X0PHL, gpp[186], c_X0PHL_H == 0, h1_X0PHL);
    //countM++;
    fp_mul(u2_X0PHL, u2_X0PH, h1_X0PHL);
    fp_prime_back(tmp_bn, u2_X0PHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t d_X0PHL  = _tmp;

    uint64_t c_X0PHL  = c_X0PHL_H + ((d_X0PHL - 1) & 0x3Full) * 64;
    uint64_t c_X0PH_24 = c_X0PHH + ((c_X0PHL - 1) & 0xFFFull) * 4096;


    GPOW_144_24(h1_X0P,   (1ull << 24) - c_X0PH_24, gw);
    SELECT(h1_X0P,   gpp[168], c_X0PH_24 == 0, h1_X0P);
    GPOW_156_24(hlp1_X0P, (1ull << 24) - c_X0PH_24, gw);
    SELECT(hlp1_X0P, gpp[180], c_X0PH_24 == 0, hlp1_X0P);
    fp_mul(hlp_X0P, hlp_X0P, hlp1_X0P); //countM++;
    fp_mul(u2_X0P,  h0_X0,   h1_X0P);   //countM++;    /* u=h0_X0 */
    fp_copy(h0_X0PL, hlp_X0P);
    fp_copy(h0_X0PLH, h0_X0PL);
    for (int j = 0; j < 6; j++) { fp_sqr(h0_X0PLH, h0_X0PLH); //countS++; 
    }
    fp_prime_back(tmp_bn, h0_X0PLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t c_X0PLH_H = _tmp;

    GPOW_180_6(h1_X0PLH, (1u << 6) - c_X0PLH_H, gw);
    SELECT(h1_X0PLH, gpp[186], c_X0PLH_H == 0, h1_X0PLH);
    //countM++;
    fp_mul(u2_X0PLH, h0_X0PL, h1_X0PLH);    /* u=h0_X0PL */
    fp_prime_back(tmp_bn, u2_X0PLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t d_X0PLH = _tmp;

    uint64_t c_X0PLH = c_X0PLH_H + ((d_X0PLH - 1) & 0x3Full) * 64;


    GPOW_168_12(h1_X0PL, (1u << 12) - c_X0PLH, gw);
    SELECT(h1_X0PL, gpp[180], c_X0PLH == 0, h1_X0PL);
    fp_mul(u2_X0PL, u2_X0P, h1_X0PL); //countM++;    /* u=u2_X0P */


    fp_copy(h0_X0PLL, u2_X0PL);
    for (int j = 0; j < 6; j++) { fp_sqr(h0_X0PLL, h0_X0PLL); //countS++; 
    }
    fp_prime_back(tmp_bn, h0_X0PLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t c_X0PLL_H = _tmp;

    GPOW_180_6(h1_X0PLL, (1u << 6) - c_X0PLL_H, gw);
    SELECT(h1_X0PLL, gpp[186], c_X0PLL_H == 0, h1_X0PLL);
    //countM++;
    fp_mul(u2_X0PLL, u2_X0PL, h1_X0PLL);
    fp_prime_back(tmp_bn, u2_X0PLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t d_X0PLL  = _tmp;

    uint64_t c_X0PLL  = c_X0PLL_H  + ((d_X0PLL - 1) & 0x3Full) * 64;
    uint64_t c_X0PL_24 = c_X0PLH   + ((c_X0PLL - 1) & 0xFFFull) * 4096;

    uint64_t c_X0_48 = c_X0PH_24 + ((c_X0PL_24 - 1) & 0xFFFFFFull) * (1ull << 24); /* 48-bit */


    GPOW_95_48(f_X0, ((1ull << 48) - c_X0_48) & MASK48, gw);
    SELECT(f_X0, gpp[143], c_X0_48 == 0, f_X0);
    //countM++;
    fp_sqr(u_X1, f_X0); //countS++;           
    fp_mul(u_X1, u_X0, u_X1);               


    fp_copy(h0_X1, u_X1);
    fp_set_dig(hlp_X1, 1);
    for (int j = 0; j < 24; j++) {
        if (j == 12) fp_copy(hlp_X1, h0_X1);
        fp_sqr(h0_X1, h0_X1); //countS++;
    }
  
    fp_copy(h0_X1H, h0_X1);
    fp_set_dig(hlp_X1H, 1);
    for (int j = 0; j < 12; j++) {
        if (j == 6) fp_copy(hlp_X1H, h0_X1H);
        fp_sqr(h0_X1H, h0_X1H); //countS++;
    }
    
    fp_copy(h0_X1HH, h0_X1H);
    for (int j = 0; j < 6; j++) { fp_sqr(h0_X1HH, h0_X1HH); //countS++; 
    }
    fp_prime_back(tmp_bn, h0_X1HH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t c_X1HH_H = _tmp;
    GPOW_180_6(h1_X1HH, (1u << 6) - c_X1HH_H, gw);
    SELECT(h1_X1HH, gpp[186], c_X1HH_H == 0, h1_X1HH);
    //countM++;
    fp_mul(u2_X1HH, h0_X1H, h1_X1HH);
    fp_prime_back(tmp_bn, u2_X1HH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t d_X1HH = _tmp;
    uint64_t c_X1HH = c_X1HH_H + ((d_X1HH - 1) & 0x3Full) * 64;

    
    GPOW_168_12(h1_X1H,   (1u << 12) - c_X1HH, gw);
    SELECT(h1_X1H,   gpp[180], c_X1HH == 0, h1_X1H);
    GPOW_174_12(hlp1_X1H, (1u << 12) - c_X1HH, gw);
    SELECT(hlp1_X1H, gpp[186], c_X1HH == 0, hlp1_X1H);
    fp_mul(hlp_X1H, hlp_X1H, hlp1_X1H); //countM++;
    fp_mul(u2_X1H,  h0_X1,   h1_X1H);   //countM++;

    
    fp_copy(h0_X1HL, hlp_X1H);
    fp_prime_back(tmp_bn, h0_X1HL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t c_X1HL_H = _tmp;
    
    GPOW_180_6(h1_X1HL, (1u << 6) - c_X1HL_H, gw);
    SELECT(h1_X1HL, gpp[186], c_X1HL_H == 0, h1_X1HL);
    //countM++;
    fp_mul(u2_X1HL, u2_X1H, h1_X1HL);
    fp_prime_back(tmp_bn, u2_X1HL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t d_X1HL   = _tmp;
    
    uint64_t c_X1HL   = c_X1HL_H + ((d_X1HL - 1) & 0x3Full) * 64;
    uint64_t c_X1H_24 = c_X1HH   + ((c_X1HL - 1) & 0xFFFull) * 4096;

    
    GPOW_143_24(f_X1,    (1ull << 24) - c_X1H_24, gw);
    SELECT(f_X1,    gpp[167], c_X1H_24 == 0, f_X1);
    GPOW_156_24(hlp1_X1, (1ull << 24) - c_X1H_24, gw);
    SELECT(hlp1_X1, gpp[180], c_X1H_24 == 0, hlp1_X1);
    fp_mul(hlp_X1, hlp_X1, hlp1_X1); //countM++; 
    //countM++;                                    
    fp_sqr(u_X2, f_X1); //countS++;               
    fp_mul(u_X2, u_X1, u_X2);                    

    
    fp_copy(h0_X2, hlp_X1);

    fp_copy(h0_X2H, h0_X2);
    for (int j = 0; j < 6; j++) { fp_sqr(h0_X2H, h0_X2H); //countS++; 
    }
    fp_prime_back(tmp_bn, h0_X2H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t c_X2H_H = _tmp;
    GPOW_180_6(h1_X2H, (1u << 6) - c_X2H_H, gw);
    SELECT(h1_X2H, gpp[186], c_X2H_H == 0, h1_X2H);
    //countM++;
    fp_mul(u2_X2H, h0_X2, h1_X2H); 
    fp_prime_back(tmp_bn, u2_X2H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t d_X2H = _tmp;
    uint64_t c_X2  = c_X2H_H + ((d_X2H - 1) & 0x3Full) * 64;    /* 12-bit */


    GPOW_167_12(f_X2, (1u << 12) - c_X2, gw);
    SELECT(f_X2, gpp[179], c_X2 == 0, f_X2);
    //countM++;
    fp_sqr(u_X3, f_X2); //countS++;   
    fp_mul(u_X3, u_X2, u_X3);      

   
    fp_copy(h0_X3, u_X3);
    for (int j = 0; j < 6; j++) { fp_sqr(h0_X3, h0_X3); //countS++; 
    }
    fp_prime_back(tmp_bn, h0_X3);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t c_X3 = _tmp;
    

   
    GPOW_179_6(f_X3, (1u << 6) - c_X3, gw);
    SELECT(f_X3, gpp[185], c_X3 == 0, f_X3);
    //countM++;
    fp_sqr(u_X4, f_X3); //countS++;
    fp_mul(u_X4, u_X3, u_X4);      

    
     fp_prime_back(tmp_bn, u_X4);
    bn_get_dig(&d, tmp_bn);
    d=d&0x7ff;
    _tmp=rlll[d];
    uint64_t c1 = _tmp;
   
    fp_copy(t_X3, fll[c1]);   

  
    fp_mul(t_X3, f_X3, t_X3); //countM++;
 

    fp_mul(t_X2, f_X2, t_X3); //countM++;
   

    fp_mul(t_X1, f_X1, t_X2); //countM++;
  

    fp_mul(t_X0, f_X0, t_X1); //countM++;


    fp_mul(t_A, f_A, t_X0); //countM++;


    } 
    bn_free(tmp_bn);
    fp_copy(out_d, t_A);

#define FP_FREE(v) fp_free(v)
    FP_FREE(h0_A); FP_FREE(h0_PA); FP_FREE(h0_PAH); FP_FREE(h0_PAHH);
    FP_FREE(hlp_A); FP_FREE(hlp_PA); FP_FREE(hlp_PAH); FP_FREE(hlp_PAHH);
    FP_FREE(h1_PAHHH); FP_FREE(u2_PAHHH); FP_FREE(h1_PAHH); FP_FREE(hlp1_PAHH); FP_FREE(u2_PAHH);
    FP_FREE(h0_PAHHL); FP_FREE(h1_PAHHL); FP_FREE(u2_PAHHL);
    FP_FREE(h1_PAH); FP_FREE(hlp1_PAH); FP_FREE(u2_PAH);
    FP_FREE(h0_PAHL); FP_FREE(h0_PAHLH); FP_FREE(h1_PAHLH); FP_FREE(u2_PAHLH);
    FP_FREE(h1_PAHL); FP_FREE(u2_PAHL);
    FP_FREE(h0_PAHLL); FP_FREE(h1_PAHLL); FP_FREE(u2_PAHLL);
    FP_FREE(h1_PA); FP_FREE(hlp1_PA); FP_FREE(u2_PA);
    FP_FREE(h0_PAL); FP_FREE(hlp_PALH); FP_FREE(h0_PALH); FP_FREE(h0_PALHH);
    FP_FREE(h1_PALHH); FP_FREE(u2_PALHH);
    FP_FREE(h1_PALH); FP_FREE(hlp1_PALH); FP_FREE(u2_PALH);
    FP_FREE(h0_PALHL); FP_FREE(h1_PALHL); FP_FREE(u2_PALHL);
    FP_FREE(h1_PAL); FP_FREE(u2_PAL);
    FP_FREE(hlp_PALL); FP_FREE(h0_PALL); FP_FREE(h0_PALLH);
    FP_FREE(h1_PALLH); FP_FREE(u2_PALLH);
    FP_FREE(h1_PALL); FP_FREE(hlp1_PALL); FP_FREE(u2_PALL);
    FP_FREE(h0_PALLL); FP_FREE(h1_PALLL); FP_FREE(u2_PALLL);
    FP_FREE(f_A); FP_FREE(hlp1_A); FP_FREE(u_X0);
    FP_FREE(h0_X0); FP_FREE(hlp_X0P); FP_FREE(h0_X0P); FP_FREE(hlp_X0PH);
    FP_FREE(h0_X0PH); FP_FREE(h0_X0PHH);
    FP_FREE(h1_X0PHH); FP_FREE(u2_X0PHH); FP_FREE(h1_X0PH); FP_FREE(hlp1_X0PH); FP_FREE(u2_X0PH);
    FP_FREE(h0_X0PHL); FP_FREE(h1_X0PHL); FP_FREE(u2_X0PHL);
    FP_FREE(h1_X0P); FP_FREE(hlp1_X0P); FP_FREE(u2_X0P);
    FP_FREE(h0_X0PL); FP_FREE(h0_X0PLH); FP_FREE(h1_X0PLH); FP_FREE(u2_X0PLH);
    FP_FREE(h1_X0PL); FP_FREE(u2_X0PL);
    FP_FREE(h0_X0PLL); FP_FREE(h1_X0PLL); FP_FREE(u2_X0PLL);
    FP_FREE(f_X0); FP_FREE(u_X1);
    FP_FREE(hlp_X1); FP_FREE(h0_X1); FP_FREE(hlp_X1H); FP_FREE(h0_X1H); FP_FREE(h0_X1HH);
    FP_FREE(h1_X1HH); FP_FREE(u2_X1HH); FP_FREE(h1_X1H); FP_FREE(hlp1_X1H); FP_FREE(u2_X1H);
    FP_FREE(h0_X1HL); FP_FREE(h1_X1HL); FP_FREE(u2_X1HL);
    FP_FREE(f_X1); FP_FREE(hlp1_X1); FP_FREE(u_X2);
    FP_FREE(h0_X2); FP_FREE(h0_X2H); FP_FREE(h1_X2H); FP_FREE(u2_X2H);
    FP_FREE(f_X2); FP_FREE(u_X3); FP_FREE(h0_X3); FP_FREE(f_X3); FP_FREE(u_X4);
    FP_FREE(t_X3); FP_FREE(t_X2); FP_FREE(t_X1); FP_FREE(t_X0); FP_FREE(t_A);
}

/* ── SQRT ────────────────────────────────────────────────────────────────── */
void SQRT(fp_t x, fp_t y, bn_t e,
          fp_t gw[NW][WE], int rlll[2048], fp_t rll[WE], fp_t fll[WE], fp_t gpp[N])
{
    fp_t u, v, w_, d;
    fp_null(u); fp_null(v); fp_null(w_); fp_null(d);
    fp_new(u); fp_new(v); fp_new(w_); fp_new(d);

    fp_exp(v, x, e);
    fp_mul(w_, x, v); //countM++;
    fp_mul(u,  w_, v); //countM++;
    solve_dlp_pow2_flat(u, d, gw, rlll, rll, fll, gpp);
    fp_mul(y, w_, d); //countM++;
    //fp_print(x);
    //fp_sqr(y, y);
    //fp_print(y);

    fp_free(u); fp_free(v); fp_free(w_); fp_free(d);
}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(void)
{
    if (core_init() != RLC_OK) { core_clean(); return 1; }
    if (fp_param_set_any_pmers() != RLC_OK) {
        printf("Curve initialization failed\n");
        core_clean(); return 1;
    }
    printf("Using Primes for bls24-509, n=192, w=6\n");

    fp_t gw[NW][WE], rll[WE], fll[WE], gpp[N], g, z, h, hh, b, y;
    int rlll[2048]={0};
    rlll[ 0x1 ]= 0 ;
    rlll[ 0x765 ]= 1 ;
    rlll[ 0xab ]= 2 ;
    rlll[ 0x4a ]= 3 ;
    rlll[ 0x539 ]= 4 ;
    rlll[ 0xfe ]= 5 ;
    rlll[ 0x6d7 ]= 6 ;
    rlll[ 0x31e ]= 7 ;
    rlll[ 0x6bb ]= 8 ;
    rlll[ 0x74d ]= 9 ;
    rlll[ 0x70c ]= 10 ;
    rlll[ 0x732 ]= 11 ;
    rlll[ 0x752 ]= 12 ;
    rlll[ 0x63d ]= 13 ;
    rlll[ 0x471 ]= 14 ;
    rlll[ 0x534 ]= 15 ;
    rlll[ 0x7e3 ]= 16 ;
    rlll[ 0x388 ]= 17 ;
    rlll[ 0x655 ]= 18 ;
    rlll[ 0x63 ]= 19 ;
    rlll[ 0xd1 ]= 20 ;
    rlll[ 0x46d ]= 21 ;
    rlll[ 0x675 ]= 22 ;
    rlll[ 0x95 ]= 23 ;
    rlll[ 0x5cd ]= 24 ;
    rlll[ 0x2a ]= 25 ;
    rlll[ 0x715 ]= 26 ;
    rlll[ 0x548 ]= 27 ;
    rlll[ 0x58c ]= 28 ;
    rlll[ 0x5f5 ]= 29 ;
    rlll[ 0xc0 ]= 30 ;
    rlll[ 0x129 ]= 31 ;
    rlll[ 0x0 ]= 32 ;
    rlll[ 0x9c ]= 33 ;
    rlll[ 0x756 ]= 34 ;
    rlll[ 0x7b7 ]= 35 ;
    rlll[ 0x2c8 ]= 36 ;
    rlll[ 0x703 ]= 37 ;
    rlll[ 0x12a ]= 38 ;
    rlll[ 0x4e3 ]= 39 ;
    rlll[ 0x146 ]= 40 ;
    rlll[ 0xb4 ]= 41 ;
    rlll[ 0xf5 ]= 42 ;
    rlll[ 0xcf ]= 43 ;
    rlll[ 0xaf ]= 44 ;
    rlll[ 0x1c4 ]= 45 ;
    rlll[ 0x390 ]= 46 ;
    rlll[ 0x2cd ]= 47 ;
    rlll[ 0x1e ]= 48 ;
    rlll[ 0x479 ]= 49 ;
    rlll[ 0x1ac ]= 50 ;
    rlll[ 0x79e ]= 51 ;
    rlll[ 0x730 ]= 52 ;
    rlll[ 0x394 ]= 53 ;
    rlll[ 0x18c ]= 54 ;
    rlll[ 0x76c ]= 55 ;
    rlll[ 0x234 ]= 56 ;
    rlll[ 0x7d7 ]= 57 ;
    rlll[ 0xec ]= 58 ;
    rlll[ 0x2b9 ]= 59 ;
    rlll[ 0x275 ]= 60 ;
    rlll[ 0x20c ]= 61 ;
    rlll[ 0x741 ]= 62 ;
    rlll[ 0x6d8 ]= 63 ;

    for (int i = 0; i < NW; i++)
        for (int j = 0; j < WE; j++) { fp_null(gw[i][j]); fp_new(gw[i][j]); }
    for (int i = 0; i < WE; i++) { fp_null(rll[i]); fp_new(rll[i]); }
    for (int i = 0; i < WE; i++) { fp_null(fll[i]); fp_new(fll[i]); }
    for (int i = 0; i < N;  i++) { fp_null(gpp[i]); fp_new(gpp[i]); }
    fp_null(b); fp_new(b); fp_null(y); fp_new(y);
    fp_null(g); fp_new(g); fp_null(z); fp_new(z);
    fp_null(h); fp_new(h); fp_null(hh); fp_new(hh);

    fp_rand(b); while (!fp_is_sqr(b)) fp_rand(b);

    bn_t tmp, m, e;
    bn_null(e); bn_new(e);
    bn_null(m); bn_new(m);
    bn_null(tmp); bn_new(tmp);

    /* z = 3 */
    bn_read_str(tmp, "3", 1, 16);
    fp_prime_conv(z, tmp);

    /* g = z^m where m = (p-1)/2^192 */
    bn_read_str(m, "800000000000011", 32, 16);
    fp_exp(g, z, m);

    /* e = (m-1)/2 for sqrt step: x^((m-1)/2) */
    bn_read_str(e, "400000000000008", 32, 16);

    /* h = g^(2^(n-w)) = g^(2^186)
     * 2^186 = 0x40000000000000000000000000000000000000000000000  (47 hex digits) */
    bn_t a1; bn_null(a1); bn_new(a1);
    bn_read_str(a1, "40000000000000000000000000000000000000000000000", 47, 16);
    fp_exp(h, g, a1);

    /* hh = h1^(-1) where h1 = sqrt(h), so hh = (sqrt(h))^(-1) */
    fp_srt(hh, h);
    fp_inv(hh, hh);

    precomputation(g, h, hh, gw, rll, fll, gpp);

    MEASURE(SQRT(b, y, e, gw,rlll, rll, fll, gpp);)

    printf("RDTSC_clk_min    = %f\n", RDTSC_clk_min);
    printf("RDTSC_clk_median = %f\n", RDTSC_clk_median);
    printf("RDTSC_clk_max    = %f\n", RDTSC_clk_max);
    printf("mult_count = %d\n", countM);
    printf("sqr_count  = %d\n", countS);

    for (int i = 0; i < NW; i++) for (int j = 0; j < WE; j++) fp_free(gw[i][j]);
    for (int i = 0; i < WE; i++) fp_free(rll[i]);
    for (int i = 0; i < WE; i++) fp_free(fll[i]);
    for (int i = 0; i < N;  i++) fp_free(gpp[i]);
    fp_free(b); fp_free(y); fp_free(g); fp_free(z); fp_free(h); fp_free(hh);
    bn_free(e); bn_free(m); bn_free(a1); bn_free(tmp);
    core_clean();
    return 0;
}
