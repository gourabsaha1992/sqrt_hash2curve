

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <immintrin.h>

/* ── 120-bit integer ─────────────────────────────────────────────────────── */
#define LIMB_BITS 60
#define LIMB_MASK 0x0FFFFFFFFFFFFFFFull
typedef struct { uint64_t low; uint64_t high; } uint120_t;
static inline uint120_t make120(uint64_t h,uint64_t l)
{ return (uint120_t){.low=l&LIMB_MASK,.high=h&LIMB_MASK}; }
static inline uint120_t add120(uint120_t a,uint120_t b){
    uint64_t r=a.low+b.low,c=r>>LIMB_BITS;
    return make120((a.high+b.high+c)&LIMB_MASK,r&LIMB_MASK);}
static inline uint120_t sub120(uint120_t a,uint120_t b){
    uint64_t r=(1ull<<LIMB_BITS)+a.low-b.low,bw=1-(r>>LIMB_BITS);
    return make120((a.high-b.high-bw)&LIMB_MASK,r&LIMB_MASK);}
static inline uint120_t rshift120(uint120_t a,int n){
    if(n<=0)return a; if(n>=120)return make120(0,0);
    if(n<LIMB_BITS)return make120((a.high>>n)&LIMB_MASK,
        ((a.low>>n)|(a.high<<(LIMB_BITS-n)))&LIMB_MASK);
    return make120(0,(a.high>>(n-LIMB_BITS))&LIMB_MASK);}
#define POW96  make120(1ull<<36,0)
#define MASK48 0x0000FFFFFFFFFFFFull
/* CHUNK96 w=4: k<15 from low, k>=15 from high, no straddle */
#define CHUNK96(e,k) (((k)<15)?(uint8_t)((e).low>>((k)*4))&0xF \
                               :(uint8_t)((e).high>>(((k)-15)*4))&0xF)

/* ── Platform clock ──────────────────────────────────────────────────────── */
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

#define W 4
#define WE 16
#define N 192
#define NW 48
int countM=0,countS=0;

static void SELECT(fp_t a0,fp_t a1,bool ctl,fp_t out){
    if(ctl)fp_copy(out,a1);else fp_copy(out,a0);}
static int rll_lookup(fp_t x,fp_t rll[WE]){
    for(int k=0;k<WE;k++)if(fp_cmp(rll[k],x)==RLC_EQ)return k;
    return 0;}

/* ── GPOW functions ──────────────────────────────────────────────────────── */
static void GPOW_186_3(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=(1u<<3)-1;e<<=2;
    fp_copy(t,gw[46][e&0xF]);e>>=4;fp_mul(t,t,gw[47][e&0xF]);
    //countM++;
    }
static void GPOW_180_6(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=(1u<<6)-1;
    fp_copy(t,gw[45][e&0xF]);e>>=4;fp_mul(t,t,gw[46][e&0xF]);
    //countM++;
    }
static void GPOW_174_12(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=(1u<<12)-1;e<<=2;
    fp_copy(t,gw[43][e&0xF]);e>>=4;fp_mul(t,t,gw[44][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[45][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[46][e&0xF]);
    //countM++;
    }
static void GPOW_168_12(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=(1u<<12)-1;
    fp_copy(t,gw[42][e&0xF]);e>>=4;fp_mul(t,t,gw[43][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[44][e&0xF]);
    //countM++;
    }
static void GPOW_156_24(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=(1u<<24)-1;
    fp_copy(t,gw[39][e&0xF]);e>>=4;fp_mul(t,t,gw[40][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[41][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[42][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[43][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[44][e&0xF]);
    //countM++;
    }
static void GPOW_144_24(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=(1u<<24)-1;
    fp_copy(t,gw[36][e&0xF]);e>>=4;fp_mul(t,t,gw[37][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[38][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[39][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[40][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[41][e&0xF]);
    //countM++;
    }
static void GPOW_120_48(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=MASK48;
    fp_copy(t,gw[30][e&0xF]);
    e>>=4;fp_mul(t,t,gw[31][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[32][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[33][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[34][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[35][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[36][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[37][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[38][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[39][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[40][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[41][e&0xF]);
    //countM++;
    }
static void GPOW_96_48(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=MASK48;
    fp_copy(t,gw[24][e&0xF]);
    e>>=4;fp_mul(t,t,gw[25][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[26][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[27][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[28][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[29][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[30][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[31][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[32][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[33][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[34][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[35][e&0xF]);
    //countM++;
    }
static void GPOW_0_95(fp_t t,uint120_t e,fp_t gw[NW][WE]){
    e.high&=~(1ull<<35);
    fp_copy(t,gw[0][CHUNK96(e,0)]);
    fp_mul(t,t,gw[1][CHUNK96(e,1)]);
    //countM++;
    fp_mul(t,t,gw[2][CHUNK96(e,2)]);
    //countM++;
    fp_mul(t,t,gw[3][CHUNK96(e,3)]);
    //countM++;
    fp_mul(t,t,gw[4][CHUNK96(e,4)]);
    //countM++;
    fp_mul(t,t,gw[5][CHUNK96(e,5)]);
    //countM++;
    fp_mul(t,t,gw[6][CHUNK96(e,6)]);
    //countM++;
    fp_mul(t,t,gw[7][CHUNK96(e,7)]);
    //countM++;
    fp_mul(t,t,gw[8][CHUNK96(e,8)]);
    //countM++;
    fp_mul(t,t,gw[9][CHUNK96(e,9)]);
    //countM++;
    fp_mul(t,t,gw[10][CHUNK96(e,10)]);
    //countM++;
    fp_mul(t,t,gw[11][CHUNK96(e,11)]);
    //countM++;
    fp_mul(t,t,gw[12][CHUNK96(e,12)]);
    //countM++;
    fp_mul(t,t,gw[13][CHUNK96(e,13)]);
    //countM++;
    fp_mul(t,t,gw[14][CHUNK96(e,14)]);
    //countM++;
    fp_mul(t,t,gw[15][CHUNK96(e,15)]);
    //countM++;
    fp_mul(t,t,gw[16][CHUNK96(e,16)]);
    //countM++;
    fp_mul(t,t,gw[17][CHUNK96(e,17)]);
    //countM++;
    fp_mul(t,t,gw[18][CHUNK96(e,18)]);
    //countM++;
    fp_mul(t,t,gw[19][CHUNK96(e,19)]);
    //countM++;
    fp_mul(t,t,gw[20][CHUNK96(e,20)]);
    //countM++;
    fp_mul(t,t,gw[21][CHUNK96(e,21)]);
    //countM++;
    fp_mul(t,t,gw[22][CHUNK96(e,22)]);
    //countM++;
    fp_mul(t,t,gw[23][CHUNK96(e,23)]);
    //countM++;
    }
static void GPOW_48_96(fp_t t,uint120_t e,fp_t gw[NW][WE]){
    e.high&=(1ull<<36)-1;
    fp_copy(t,gw[12][CHUNK96(e,0)]);
    fp_mul(t,t,gw[13][CHUNK96(e,1)]);
    //countM++;
    fp_mul(t,t,gw[14][CHUNK96(e,2)]);
    //countM++;
    fp_mul(t,t,gw[15][CHUNK96(e,3)]);
    //countM++;
    fp_mul(t,t,gw[16][CHUNK96(e,4)]);
    //countM++;
    fp_mul(t,t,gw[17][CHUNK96(e,5)]);
    //countM++;
    fp_mul(t,t,gw[18][CHUNK96(e,6)]);
    //countM++;
    fp_mul(t,t,gw[19][CHUNK96(e,7)]);
    //countM++;
    fp_mul(t,t,gw[20][CHUNK96(e,8)]);
    //countM++;
    fp_mul(t,t,gw[21][CHUNK96(e,9)]);
    //countM++;
    fp_mul(t,t,gw[22][CHUNK96(e,10)]);
    //countM++;
    fp_mul(t,t,gw[23][CHUNK96(e,11)]);
    //countM++;
    fp_mul(t,t,gw[24][CHUNK96(e,12)]);
    //countM++;
    fp_mul(t,t,gw[25][CHUNK96(e,13)]);
    //countM++;
    fp_mul(t,t,gw[26][CHUNK96(e,14)]);
    //countM++;
    fp_mul(t,t,gw[27][CHUNK96(e,15)]);
    //countM++;
    fp_mul(t,t,gw[28][CHUNK96(e,16)]);
    //countM++;
    fp_mul(t,t,gw[29][CHUNK96(e,17)]);
    //countM++;
    fp_mul(t,t,gw[30][CHUNK96(e,18)]);
    //countM++;
    fp_mul(t,t,gw[31][CHUNK96(e,19)]);
    //countM++;
    fp_mul(t,t,gw[32][CHUNK96(e,20)]);
    //countM++;
    fp_mul(t,t,gw[33][CHUNK96(e,21)]);
    //countM++;
    fp_mul(t,t,gw[34][CHUNK96(e,22)]);
    //countM++;
    fp_mul(t,t,gw[35][CHUNK96(e,23)]);
    //countM++;
    }
static void GPOW_95_48(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=MASK48;e<<=3;
    fp_copy(t,gw[23][e&0xF]);
    e>>=4;fp_mul(t,t,gw[24][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[25][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[26][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[27][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[28][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[29][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[30][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[31][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[32][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[33][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[34][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[35][e&0xF]);
    //countM++;
    }
static void GPOW_143_24(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=(1u<<24)-1;e<<=3;
    fp_copy(t,gw[35][e&0xF]);
    e>>=4;fp_mul(t,t,gw[36][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[37][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[38][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[39][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[40][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[41][e&0xF]);
    //countM++;
    }
static void GPOW_167_12(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=(1u<<12)-1;e<<=3;
    fp_copy(t,gw[41][e&0xF]);
    e>>=4;fp_mul(t,t,gw[42][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[43][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[44][e&0xF]);
    //countM++;
    }
static void GPOW_179_6(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=(1u<<6)-1;e<<=3;
    fp_copy(t,gw[44][e&0xF]);
    e>>=4;fp_mul(t,t,gw[45][e&0xF]);
    //countM++;
    e>>=4;fp_mul(t,t,gw[46][e&0xF]);
    //countM++;
    }
static void GPOW_185_3(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=(1u<<3)-1;e<<=1;fp_copy(t,gw[46][e&0xF]);}

/* ── Precomputation ──────────────────────────────────────────────────────── */
void precomputation(fp_t g,fp_t h,fp_t hh,
                    fp_t gw[NW][WE],fp_t rll[WE],fp_t fll[WE],fp_t gpp[N]){
    bn_t temp,one;bn_null(temp);bn_new(temp);bn_null(one);bn_new(one);bn_set_dig(one,1);
    for(int v=0;v<WE;v++){bn_set_dig(temp,v);fp_exp(rll[v],h,temp);fp_exp(fll[v],hh,temp);}
    for(int i=0;i<NW;i++)for(int j=0;j<WE;j++){
        bn_lsh(temp,one,i*W);bn_mul_dig(temp,temp,j);fp_exp(gw[i][j],g,temp);}
    fp_copy(gpp[0],g);
    for(int i=1;i<N;i++)fp_sqr(gpp[i],gpp[i-1]);
    bn_free(temp);bn_free(one);}

/* ── solve_dlp_pow2_flat ─────────────────────────────────────────────────── */
void solve_dlp_pow2_flat(fp_t u_A,fp_t out_d,
                         fp_t gw[NW][WE], int rlll[32],fp_t rll[WE],fp_t fll[WE],fp_t gpp[N])
{
#define FA(v) fp_null(v);fp_new(v)
    fp_t h0_A,h0_PA,h0_PAH,h0_PAHH,hlp_A,hlp_PA,hlp_PAH,hlp_PAHH;
    FA(h0_A);FA(h0_PA);FA(h0_PAH);FA(h0_PAHH);FA(hlp_A);FA(hlp_PA);FA(hlp_PAH);FA(hlp_PAHH);
    /* DLP186 sub-block temporaries — c_A tree */
    fp_t h0_PAHHH,sq_PAHHHH,h1_PAHHHH,u2_PAHHHH,sq_PAHHHL,h1_PAHHHL,u2_PAHHHL,u2_PAHHH;
    FA(h0_PAHHH);FA(sq_PAHHHH);FA(h1_PAHHHH);FA(u2_PAHHHH);FA(sq_PAHHHL);FA(h1_PAHHHL);FA(u2_PAHHHL);
    fp_t sq_PAHHLH,h1_PAHHLH,u2_PAHHLH,sq_PAHHLL,h1_PAHHLL,u2_PAHHLL;
    FA(sq_PAHHLH);FA(h1_PAHHLH);FA(u2_PAHHLH);FA(sq_PAHHLL);FA(h1_PAHHLL);FA(u2_PAHHLL);
    fp_t h1_PAHH,hlp1_PAHH,u2_PAHH;FA(h1_PAHH);FA(hlp1_PAHH);FA(u2_PAHH);
    fp_t h0_PAHLH,sq_PAHLHH,h1_PAHLHH,u2_PAHLHH,sq_PAHLHL,h1_PAHLHL,u2_PAHLHL;
    FA(h0_PAHLH);FA(sq_PAHLHH);FA(h1_PAHLHH);FA(u2_PAHLHH);FA(sq_PAHLHL);FA(h1_PAHLHL);FA(u2_PAHLHL);
    fp_t h1_PAHL,u2_PAHL;FA(h1_PAHL);FA(u2_PAHL);
    fp_t h0_PAHLL,sq_PAHLLH,h1_PAHLLH,u2_PAHLLH,sq_PAHLLL,h1_PAHLLL,u2_PAHLLL;
    FA(h0_PAHLL);FA(sq_PAHLLH);FA(h1_PAHLLH);FA(u2_PAHLLH);FA(sq_PAHLLL);FA(h1_PAHLLL);FA(u2_PAHLLL);
    fp_t h1_PAH,hlp1_PAH,u2_PAH;FA(h1_PAH);FA(hlp1_PAH);FA(u2_PAH);
    fp_t h0_PALH,hlp_PALH;FA(h0_PALH);FA(hlp_PALH);
    fp_t h0_PALHH,sq_PALHHH,h1_PALHHH,u2_PALHHH,sq_PALHHL,h1_PALHHL,u2_PALHHL;
    FA(h0_PALHH);FA(sq_PALHHH);FA(h1_PALHHH);FA(u2_PALHHH);FA(sq_PALHHL);FA(h1_PALHHL);FA(u2_PALHHL);
    fp_t h1_PALH,hlp1_PALH,u2_PALH;FA(h1_PALH);FA(hlp1_PALH);FA(u2_PALH);
    fp_t sq_PALHLH,h1_PALHLH,u2_PALHLH,sq_PALHLL,h1_PALHLL,u2_PALHLL;
    FA(sq_PALHLH);FA(h1_PALHLH);FA(u2_PALHLH);FA(sq_PALHLL);FA(h1_PALHLL);FA(u2_PALHLL);
    fp_t h1_PAL,u2_PAL;FA(h1_PAL);FA(u2_PAL);
    fp_t h0_PALL,hlp_PALL;FA(h0_PALL);FA(hlp_PALL);
    fp_t h0_PALLH,sq_PALLHH,h1_PALLHH,u2_PALLHH,sq_PALLHL,h1_PALLHL,u2_PALLHL;
    FA(h0_PALLH);FA(sq_PALLHH);FA(h1_PALLHH);FA(u2_PALLHH);FA(sq_PALLHL);FA(h1_PALLHL);FA(u2_PALLHL);
    fp_t h1_PALL,hlp1_PALL,u2_PALL;FA(h1_PALL);FA(hlp1_PALL);FA(u2_PALL);
    fp_t sq_PALLLH,h1_PALLLH,u2_PALLLH,sq_PALLLL,h1_PALLLL,u2_PALLLL;
    FA(sq_PALLLH);FA(h1_PALLLH);FA(u2_PALLLH);FA(sq_PALLLL);FA(h1_PALLLL);FA(u2_PALLLL);
    fp_t h1_PA,hlp1_PA,u2_PA;FA(h1_PA);FA(hlp1_PA);FA(u2_PA);
    fp_t f_A,hlp1_A,u_X0;FA(f_A);FA(hlp1_A);FA(u_X0);
    /* EXT0 */
    fp_t h0_X0,h0_X0P,hlp_X0P,h0_X0PH,hlp_X0PH;
    FA(h0_X0);FA(h0_X0P);FA(hlp_X0P);FA(h0_X0PH);FA(hlp_X0PH);
    fp_t h0_X0PHH,sq_X0PHHH,h1_X0PHHH,u2_X0PHHH,sq_X0PHHL,h1_X0PHHL,u2_X0PHHL;
    FA(h0_X0PHH);FA(sq_X0PHHH);FA(h1_X0PHHH);FA(u2_X0PHHH);FA(sq_X0PHHL);FA(h1_X0PHHL);FA(u2_X0PHHL);
    fp_t h1_X0PH,hlp1_X0PH,u2_X0PH;FA(h1_X0PH);FA(hlp1_X0PH);FA(u2_X0PH);
    fp_t sq_X0PHLH,h1_X0PHLH,u2_X0PHLH,sq_X0PHLL,h1_X0PHLL,u2_X0PHLL;
    FA(sq_X0PHLH);FA(h1_X0PHLH);FA(u2_X0PHLH);FA(sq_X0PHLL);FA(h1_X0PHLL);FA(u2_X0PHLL);
    fp_t h1_X0P,hlp1_X0P,u2_X0P;FA(h1_X0P);FA(hlp1_X0P);FA(u2_X0P);
    fp_t h0_X0PLH,sq_X0PLHH,h1_X0PLHH,u2_X0PLHH,sq_X0PLHL,h1_X0PLHL,u2_X0PLHL;
    FA(h0_X0PLH);FA(sq_X0PLHH);FA(h1_X0PLHH);FA(u2_X0PLHH);FA(sq_X0PLHL);FA(h1_X0PLHL);FA(u2_X0PLHL);
    fp_t h1_X0PL,u2_X0PL;FA(h1_X0PL);FA(u2_X0PL);
    fp_t h0_X0PLL,sq_X0PLLH,h1_X0PLLH,u2_X0PLLH,sq_X0PLLL,h1_X0PLLL,u2_X0PLLL;
    FA(h0_X0PLL);FA(sq_X0PLLH);FA(h1_X0PLLH);FA(u2_X0PLLH);FA(sq_X0PLLL);FA(h1_X0PLLL);FA(u2_X0PLLL);
    fp_t f_X0,u_X1;FA(f_X0);FA(u_X1);
    /* EXT1 */
    fp_t h0_X1,hlp_X1,h0_X1H,hlp_X1H;FA(h0_X1);FA(hlp_X1);FA(h0_X1H);FA(hlp_X1H);
    fp_t h0_X1HH,sq_X1HHH,h1_X1HHH,u2_X1HHH,sq_X1HHL,h1_X1HHL,u2_X1HHL;
    FA(h0_X1HH);FA(sq_X1HHH);FA(h1_X1HHH);FA(u2_X1HHH);FA(sq_X1HHL);FA(h1_X1HHL);FA(u2_X1HHL);
    fp_t h1_X1H,hlp1_X1H,u2_X1H;FA(h1_X1H);FA(hlp1_X1H);FA(u2_X1H);
    fp_t sq_X1HLH,h1_X1HLH,u2_X1HLH,sq_X1HLL,h1_X1HLL,u2_X1HLL;
    FA(sq_X1HLH);FA(h1_X1HLH);FA(u2_X1HLH);FA(sq_X1HLL);FA(h1_X1HLL);FA(u2_X1HLL);
    fp_t f_X1,hlp1_X1,u_X2;FA(f_X1);FA(hlp1_X1);FA(u_X2);
    /* EXT2 */
    fp_t h0_X2,h0_X2H,sq_X2HH,h1_X2HH,u2_X2HH,sq_X2HL,h1_X2HL,u2_X2HL;
    FA(h0_X2);FA(h0_X2H);FA(sq_X2HH);FA(h1_X2HH);FA(u2_X2HH);FA(sq_X2HL);FA(h1_X2HL);FA(u2_X2HL);
    fp_t f_X2,u_X3;FA(f_X2);FA(u_X3);
  
    fp_t h0_X3,sq_X3H,h1_X3H,u2_X3H;
    FA(h0_X3);FA(sq_X3H);FA(h1_X3H);FA(u2_X3H);
    fp_t f_X3,u_X4;FA(f_X3);FA(u_X4);

    fp_t h0_X4,f_X4,u_X5,t_X4,t_X3,t_X2,t_X1,t_X0,t_A;
    FA(h0_X4);FA(f_X4);FA(u_X5);FA(t_X4);FA(t_X3);FA(t_X2);FA(t_X1);FA(t_X0);FA(t_A);
#undef FA


    fp_copy(h0_A,u_A);
    for(int j=0;j<96;j++){if(j==48)fp_copy(hlp_A,h0_A);fp_sqr(h0_A,h0_A);
    //countS++;
    }

    
    fp_copy(h0_PA,h0_A);
    for(int j=0;j<48;j++){if(j==24)fp_copy(hlp_PA,h0_PA);fp_sqr(h0_PA,h0_PA);
    //countS++;
    }

    
    fp_copy(h0_PAH,h0_PA);
    for(int j=0;j<24;j++){if(j==12)fp_copy(hlp_PAH,h0_PAH);fp_sqr(h0_PAH,h0_PAH);
    //countS++;
    }

   
    fp_copy(h0_PAHH,h0_PAH);
    for(int j=0;j<12;j++){if(j==6)fp_copy(hlp_PAHH,h0_PAHH);fp_sqr(h0_PAHH,h0_PAHH);
    //countS++;
    }

    {
    
    fp_copy(h0_PAHHH,h0_PAHH);
    for(int j=0;j<6;j++){fp_sqr(h0_PAHHH,h0_PAHHH);
    //countS++;
    }

    fp_copy(sq_PAHHHH,h0_PAHHH);
    fp_sqr(sq_PAHHHH,sq_PAHHHH);
    //countS++;
    fp_sqr(sq_PAHHHH,sq_PAHHHH);
    //countS++;
    fp_sqr(sq_PAHHHH,sq_PAHHHH);
    //countS++;
    bn_t tmp_bn;
    bn_null(tmp_bn);bn_new(tmp_bn);
    dig_t d;
    int _tmp;
    fp_prime_back(tmp_bn, sq_PAHHHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PAHHHH_H=_tmp>>1;
 
    GPOW_186_3(h1_PAHHHH,(1u<<3)-c_PAHHHH_H,gw);SELECT(h1_PAHHHH,gpp[189],c_PAHHHH_H==0,h1_PAHHHH);
    //countM++;
    fp_mul(u2_PAHHHH,h0_PAHHH,h1_PAHHHH);
    fp_prime_back(tmp_bn, u2_PAHHHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PAHHHH_L=_tmp>>1;
 
    uint64_t c_PAHHHH=c_PAHHHH_H+((c_PAHHHH_L-1)&7u)*8;

    GPOW_180_6(h1_PAHHHH,(1u<<6)-c_PAHHHH,gw);SELECT(h1_PAHHHH,gpp[186],c_PAHHHH==0,h1_PAHHHH);
    //countM++;
    fp_mul(u2_PAHHH,h0_PAHH,h1_PAHHHH);
 
    fp_copy(sq_PAHHHL,u2_PAHHH);
    fp_sqr(sq_PAHHHL,sq_PAHHHL);
    //countS++;
    fp_sqr(sq_PAHHHL,sq_PAHHHL);
    //countS++;
    fp_sqr(sq_PAHHHL,sq_PAHHHL);
    //countS++;
    fp_prime_back(tmp_bn, sq_PAHHHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PAHHHL_H=_tmp>>1;

    GPOW_186_3(h1_PAHHHL,(1u<<3)-c_PAHHHL_H,gw);SELECT(h1_PAHHHL,gpp[189],c_PAHHHL_H==0,h1_PAHHHL);
    //countM++;
    fp_mul(u2_PAHHHL,u2_PAHHH,h1_PAHHHL);
    fp_prime_back(tmp_bn, u2_PAHHHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PAHHHL_L=_tmp>>1;
    
    uint64_t c_PAHHHL=c_PAHHHL_H+((c_PAHHHL_L-1)&7u)*8;
    uint64_t c_PAHHH=c_PAHHHH+((c_PAHHHL-1)&63u)*64;  

    
    GPOW_168_12(h1_PAHH,(1u<<12)-c_PAHHH,gw);SELECT(h1_PAHH,gpp[180],c_PAHHH==0,h1_PAHH);
    GPOW_174_12(hlp1_PAHH,(1u<<12)-c_PAHHH,gw);SELECT(hlp1_PAHH,gpp[186],c_PAHHH==0,hlp1_PAHH);
    fp_mul(hlp_PAHH,hlp_PAHH,hlp1_PAHH);
    //countM++;
    fp_mul(u2_PAHH,h0_PAH,h1_PAHH);
    //countM++;

   
    fp_copy(sq_PAHHLH,hlp_PAHH);
    fp_sqr(sq_PAHHLH,sq_PAHHLH);
    //countS++;
    fp_sqr(sq_PAHHLH,sq_PAHHLH);
    //countS++;
    fp_sqr(sq_PAHHLH,sq_PAHHLH);
    //countS++;
    fp_prime_back(tmp_bn, sq_PAHHLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PAHHLH_H=_tmp>>1;
   
    GPOW_186_3(h1_PAHHLH,(1u<<3)-c_PAHHLH_H,gw);SELECT(h1_PAHHLH,gpp[189],c_PAHHLH_H==0,h1_PAHHLH);
    //countM++;
    fp_mul(u2_PAHHLH,hlp_PAHH,h1_PAHHLH);
    fp_prime_back(tmp_bn, u2_PAHHLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PAHHLH_L=_tmp>>1;
    
    uint64_t c_PAHHLH=c_PAHHLH_H+((c_PAHHLH_L-1)&7u)*8;
    
    GPOW_180_6(h1_PAHHLH,(1u<<6)-c_PAHHLH,gw);SELECT(h1_PAHHLH,gpp[186],c_PAHHLH==0,h1_PAHHLH);
    //countM++;
    fp_mul(u2_PAHHLH,u2_PAHH,h1_PAHHLH);  
    
    fp_copy(sq_PAHHLL,u2_PAHHLH);
    fp_sqr(sq_PAHHLL,sq_PAHHLL);
    //countS++;
    fp_sqr(sq_PAHHLL,sq_PAHHLL);
    //countS++;
    fp_sqr(sq_PAHHLL,sq_PAHHLL);
    //countS++;
    fp_prime_back(tmp_bn, sq_PAHHLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PAHHLL_H=_tmp>>1;
    
    GPOW_186_3(h1_PAHHLL,(1u<<3)-c_PAHHLL_H,gw);SELECT(h1_PAHHLL,gpp[189],c_PAHHLL_H==0,h1_PAHHLL);
    //countM++;
    fp_mul(u2_PAHHLL,u2_PAHHLH,h1_PAHHLL);
    fp_prime_back(tmp_bn, u2_PAHHLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PAHHLL_L=_tmp>>1;
   
    uint64_t c_PAHHLL=c_PAHHLL_H+((c_PAHHLL_L-1)&7u)*8;
    uint64_t c_PAHHL=c_PAHHLH+((c_PAHHLL-1)&63u)*64;
    uint64_t c_PAHH=c_PAHHH+((c_PAHHL-1)&0xFFFull)*4096;  

    
    GPOW_144_24(h1_PAH,(1ull<<24)-c_PAHH,gw);SELECT(h1_PAH,gpp[168],c_PAHH==0,h1_PAH);
    GPOW_156_24(hlp1_PAH,(1ull<<24)-c_PAHH,gw);SELECT(hlp1_PAH,gpp[180],c_PAHH==0,hlp1_PAH);
    fp_mul(hlp_PAH,hlp_PAH,hlp1_PAH);
    //countM++;
    fp_mul(u2_PAH,h0_PA,h1_PAH);
    //countM++;

    
    fp_copy(h0_PAHLH,hlp_PAH);
    for(int j=0;j<6;j++){fp_sqr(h0_PAHLH,h0_PAHLH);
    //countS++;
    }
    
    fp_copy(sq_PAHLHH,h0_PAHLH);
    fp_sqr(sq_PAHLHH,sq_PAHLHH);
    //countS++;
    fp_sqr(sq_PAHLHH,sq_PAHLHH);
    //countS++;
    fp_sqr(sq_PAHLHH,sq_PAHLHH);
    //countS++;
    fp_prime_back(tmp_bn, sq_PAHLHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PAHLHH_H=_tmp>>1;
    
    GPOW_186_3(h1_PAHLHH,(1u<<3)-c_PAHLHH_H,gw);SELECT(h1_PAHLHH,gpp[189],c_PAHLHH_H==0,h1_PAHLHH);
    //countM++;
    fp_mul(u2_PAHLHH,h0_PAHLH,h1_PAHLHH);
    fp_prime_back(tmp_bn, u2_PAHLHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PAHLHH_L=_tmp>>1;
    
    uint64_t c_PAHLHH=c_PAHLHH_H+((c_PAHLHH_L-1)&7u)*8;
    
    GPOW_180_6(h1_PAHLHH,(1u<<6)-c_PAHLHH,gw);SELECT(h1_PAHLHH,gpp[186],c_PAHLHH==0,h1_PAHLHH);
    //countM++;
    fp_mul(u2_PAHLHH,hlp_PAH,h1_PAHLHH);   
    
    fp_copy(sq_PAHLHL,u2_PAHLHH);
    fp_sqr(sq_PAHLHL,sq_PAHLHL);
    //countS++;
    fp_sqr(sq_PAHLHL,sq_PAHLHL);
    //countS++;
    fp_sqr(sq_PAHLHL,sq_PAHLHL);
    //countS++;
    fp_prime_back(tmp_bn, sq_PAHLHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PAHLHL_H=_tmp>>1;
    
    GPOW_186_3(h1_PAHLHL,(1u<<3)-c_PAHLHL_H,gw);SELECT(h1_PAHLHL,gpp[189],c_PAHLHL_H==0,h1_PAHLHL);
    //countM++;
    fp_mul(u2_PAHLHL,u2_PAHLHH,h1_PAHLHL);
    fp_prime_back(tmp_bn, u2_PAHLHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PAHLHL_L=_tmp>>1;
    
    uint64_t c_PAHLHL=c_PAHLHL_H+((c_PAHLHL_L-1)&7u)*8;
    uint64_t c_PAHLH=c_PAHLHH+((c_PAHLHL-1)&63u)*64;

    
    GPOW_168_12(h1_PAHL,(1u<<12)-c_PAHLH,gw);SELECT(h1_PAHL,gpp[180],c_PAHLH==0,h1_PAHL);
    //countM++;
    fp_mul(u2_PAHL,u2_PAH,h1_PAHL);

   
    fp_copy(h0_PAHLL,u2_PAHL);
    for(int j=0;j<6;j++){fp_sqr(h0_PAHLL,h0_PAHLL);
    //countS++;
    }
    
    fp_copy(sq_PAHLLH,h0_PAHLL);
    fp_sqr(sq_PAHLLH,sq_PAHLLH);
    //countS++;
    fp_sqr(sq_PAHLLH,sq_PAHLLH);
    //countS++;
    fp_sqr(sq_PAHLLH,sq_PAHLLH);
    //countS++;
    fp_prime_back(tmp_bn, sq_PAHLLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PAHLLH_H=_tmp>>1;
    
    GPOW_186_3(h1_PAHLLH,(1u<<3)-c_PAHLLH_H,gw);SELECT(h1_PAHLLH,gpp[189],c_PAHLLH_H==0,h1_PAHLLH);
    //countM++;
    fp_mul(u2_PAHLLH,h0_PAHLL,h1_PAHLLH);
    fp_prime_back(tmp_bn, u2_PAHLLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PAHLLH_L=_tmp>>1;
    
    uint64_t c_PAHLLH=c_PAHLLH_H+((c_PAHLLH_L-1)&7u)*8;
    
    GPOW_180_6(h1_PAHLLH,(1u<<6)-c_PAHLLH,gw);SELECT(h1_PAHLLH,gpp[186],c_PAHLLH==0,h1_PAHLLH);
    //countM++;
    fp_mul(u2_PAHLLH,u2_PAHL,h1_PAHLLH);
    
    fp_copy(sq_PAHLLL,u2_PAHLLH);
    fp_sqr(sq_PAHLLL,sq_PAHLLL);
    //countS++;
    fp_sqr(sq_PAHLLL,sq_PAHLLL);
    //countS++;
    fp_sqr(sq_PAHLLL,sq_PAHLLL);
    //countS++;
    fp_prime_back(tmp_bn, sq_PAHLLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PAHLLL_H=_tmp>>1;
    
    GPOW_186_3(h1_PAHLLL,(1u<<3)-c_PAHLLL_H,gw);SELECT(h1_PAHLLL,gpp[189],c_PAHLLL_H==0,h1_PAHLLL);
    //countM++;
    fp_mul(u2_PAHLLL,u2_PAHLLH,h1_PAHLLL);
    fp_prime_back(tmp_bn, u2_PAHLLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PAHLLL_L=_tmp>>1;
   
    uint64_t c_PAHLLL=c_PAHLLL_H+((c_PAHLLL_L-1)&7u)*8;
    uint64_t c_PAHLL=c_PAHLLH+((c_PAHLLL-1)&63u)*64;
    uint64_t c_PAHL=c_PAHLH+((c_PAHLL-1)&0xFFFull)*4096;  /* 24-bit */
    uint64_t c_PAH=c_PAHH+((c_PAHL-1)&0xFFFFFFull)*(1ull<<24);  /* 48-bit */

    
    GPOW_96_48(h1_PA,((1ull<<48)-c_PAH)&MASK48,gw);SELECT(h1_PA,gpp[144],c_PAH==0,h1_PA);
    GPOW_120_48(hlp1_PA,((1ull<<48)-c_PAH)&MASK48,gw);SELECT(hlp1_PA,gpp[168],c_PAH==0,hlp1_PA);
    fp_mul(hlp_PA,hlp_PA,hlp1_PA);
    //countM++;
    fp_mul(u2_PA,h0_A,h1_PA);
    //countM++;

    
    fp_copy(h0_PALH,hlp_PA);
    for(int j=0;j<12;j++){if(j==6)fp_copy(hlp_PALH,h0_PALH);fp_sqr(h0_PALH,h0_PALH);
    //countS++;
    }
    
    fp_copy(h0_PALHH,h0_PALH);
    for(int j=0;j<6;j++){fp_sqr(h0_PALHH,h0_PALHH);
    //countS++;
    }
    fp_copy(sq_PALHHH,h0_PALHH);
    fp_sqr(sq_PALHHH,sq_PALHHH);
    //countS++;
    fp_sqr(sq_PALHHH,sq_PALHHH);
    //countS++;
    fp_sqr(sq_PALHHH,sq_PALHHH);
    //countS++;
    fp_prime_back(tmp_bn, sq_PALHHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PALHHH_H=_tmp>>1;
    
    GPOW_186_3(h1_PALHHH,(1u<<3)-c_PALHHH_H,gw);SELECT(h1_PALHHH,gpp[189],c_PALHHH_H==0,h1_PALHHH);
    //countM++;
    fp_mul(u2_PALHHH,h0_PALHH,h1_PALHHH);
    fp_prime_back(tmp_bn, u2_PALHHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PALHHH_L=_tmp>>1;
    
    uint64_t c_PALHHH=c_PALHHH_H+((c_PALHHH_L-1)&7u)*8;
    GPOW_180_6(h1_PALHHH,(1u<<6)-c_PALHHH,gw);SELECT(h1_PALHHH,gpp[186],c_PALHHH==0,h1_PALHHH);
    //countM++;
    fp_mul(u2_PALHHH,h0_PALH,h1_PALHHH);
    fp_copy(sq_PALHHL,u2_PALHHH);
    fp_sqr(sq_PALHHL,sq_PALHHL);
    //countS++;
    fp_sqr(sq_PALHHL,sq_PALHHL);
    //countS++;
    fp_sqr(sq_PALHHL,sq_PALHHL);
    //countS++;
    fp_prime_back(tmp_bn, sq_PALHHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PALHHL_H=_tmp>>1;
    
    GPOW_186_3(h1_PALHHL,(1u<<3)-c_PALHHL_H,gw);SELECT(h1_PALHHL,gpp[189],c_PALHHL_H==0,h1_PALHHL);
    //countM++;
    fp_mul(u2_PALHHL,u2_PALHHH,h1_PALHHL);
    fp_prime_back(tmp_bn, u2_PALHHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PALHHL_L=_tmp>>1;
    
    uint64_t c_PALHHL=c_PALHHL_H+((c_PALHHL_L-1)&7u)*8;
    uint64_t c_PALHH=c_PALHHH+((c_PALHHL-1)&63u)*64;
    
    GPOW_168_12(h1_PALH,(1u<<12)-c_PALHH,gw);SELECT(h1_PALH,gpp[180],c_PALHH==0,h1_PALH);
    GPOW_174_12(hlp1_PALH,(1u<<12)-c_PALHH,gw);SELECT(hlp1_PALH,gpp[186],c_PALHH==0,hlp1_PALH);
    fp_mul(hlp_PALH,hlp_PALH,hlp1_PALH);
    //countM++;
    fp_mul(u2_PALH,hlp_PA,h1_PALH);
    //countM++;

    
    fp_copy(sq_PALHLH,hlp_PALH);
    fp_sqr(sq_PALHLH,sq_PALHLH);
    //countS++;
    fp_sqr(sq_PALHLH,sq_PALHLH);
    //countS++;
    fp_sqr(sq_PALHLH,sq_PALHLH);
    //countS++;
    fp_prime_back(tmp_bn, sq_PALHLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PALHLH_H=_tmp>>1;
    
    GPOW_186_3(h1_PALHLH,(1u<<3)-c_PALHLH_H,gw);SELECT(h1_PALHLH,gpp[189],c_PALHLH_H==0,h1_PALHLH);
    //countM++;
    fp_mul(u2_PALHLH,hlp_PALH,h1_PALHLH);
    fp_prime_back(tmp_bn, u2_PALHLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PALHLH_L=_tmp>>1;
    
    uint64_t c_PALHLH=c_PALHLH_H+((c_PALHLH_L-1)&7u)*8;
    GPOW_180_6(h1_PALHLH,(1u<<6)-c_PALHLH,gw);SELECT(h1_PALHLH,gpp[186],c_PALHLH==0,h1_PALHLH);
    //countM++;
    fp_mul(u2_PALHLH,u2_PALH,h1_PALHLH);
    fp_copy(sq_PALHLL,u2_PALHLH);
    fp_sqr(sq_PALHLL,sq_PALHLL);
    //countS++;
    fp_sqr(sq_PALHLL,sq_PALHLL);
    //countS++;
    fp_sqr(sq_PALHLL,sq_PALHLL);
    //countS++;
    fp_prime_back(tmp_bn, sq_PALHLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PALHLL_H=_tmp>>1;
    
    GPOW_186_3(h1_PALHLL,(1u<<3)-c_PALHLL_H,gw);SELECT(h1_PALHLL,gpp[189],c_PALHLL_H==0,h1_PALHLL);
    //countM++;
    fp_mul(u2_PALHLL,u2_PALHLH,h1_PALHLL);
    fp_prime_back(tmp_bn, u2_PALHLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PALHLL_L=_tmp>>1;
    
    uint64_t c_PALHLL=c_PALHLL_H+((c_PALHLL_L-1)&7u)*8;
    uint64_t c_PALHL=c_PALHLH+((c_PALHLL-1)&63u)*64;
    uint64_t c_PALH=c_PALHH+((c_PALHL-1)&0xFFFull)*4096;  /* 24-bit */
   
    GPOW_144_24(h1_PAL,(1ull<<24)-c_PALH,gw);SELECT(h1_PAL,gpp[168],c_PALH==0,h1_PAL);
    //countM++;
    fp_mul(u2_PAL,u2_PA,h1_PAL);

    
    fp_copy(h0_PALL,u2_PAL);
    for(int j=0;j<12;j++){if(j==6)fp_copy(hlp_PALL,h0_PALL);fp_sqr(h0_PALL,h0_PALL);
    //countS++;
    }
    
    fp_copy(h0_PALLH,h0_PALL);
    for(int j=0;j<6;j++){fp_sqr(h0_PALLH,h0_PALLH);
    //countS++;
    }
    fp_copy(sq_PALLHH,h0_PALLH);
    fp_sqr(sq_PALLHH,sq_PALLHH);
    //countS++;
    fp_sqr(sq_PALLHH,sq_PALLHH);
    //countS++;
    fp_sqr(sq_PALLHH,sq_PALLHH);
    //countS++;
    fp_prime_back(tmp_bn, sq_PALLHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PALLHH_H=_tmp>>1;
   
    GPOW_186_3(h1_PALLHH,(1u<<3)-c_PALLHH_H,gw);SELECT(h1_PALLHH,gpp[189],c_PALLHH_H==0,h1_PALLHH);
    //countM++;
    fp_mul(u2_PALLHH,h0_PALLH,h1_PALLHH);
    fp_prime_back(tmp_bn, u2_PALLHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PALLHH_L=_tmp>>1;
    
    uint64_t c_PALLHH=c_PALLHH_H+((c_PALLHH_L-1)&7u)*8;
    GPOW_180_6(h1_PALLHH,(1u<<6)-c_PALLHH,gw);SELECT(h1_PALLHH,gpp[186],c_PALLHH==0,h1_PALLHH);
    //countM++;
    fp_mul(u2_PALLHH,h0_PALL,h1_PALLHH);
    fp_copy(sq_PALLHL,u2_PALLHH);
    fp_sqr(sq_PALLHL,sq_PALLHL);
    //countS++;
    fp_sqr(sq_PALLHL,sq_PALLHL);
    //countS++;
    fp_sqr(sq_PALLHL,sq_PALLHL);
    //countS++;
    fp_prime_back(tmp_bn, sq_PALLHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PALLHL_H=_tmp>>1;
  
    GPOW_186_3(h1_PALLHL,(1u<<3)-c_PALLHL_H,gw);SELECT(h1_PALLHL,gpp[189],c_PALLHL_H==0,h1_PALLHL);
    //countM++;
    fp_mul(u2_PALLHL,u2_PALLHH,h1_PALLHL);
    fp_prime_back(tmp_bn, u2_PALLHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PALLHL_L=_tmp>>1;
 
    uint64_t c_PALLHL=c_PALLHL_H+((c_PALLHL_L-1)&7u)*8;
    uint64_t c_PALLH=c_PALLHH+((c_PALLHL-1)&63u)*64;
   
    GPOW_168_12(h1_PALL,(1u<<12)-c_PALLH,gw);SELECT(h1_PALL,gpp[180],c_PALLH==0,h1_PALL);
    GPOW_174_12(hlp1_PALL,(1u<<12)-c_PALLH,gw);SELECT(hlp1_PALL,gpp[186],c_PALLH==0,hlp1_PALL);
    fp_mul(hlp_PALL,hlp_PALL,hlp1_PALL);
    //countM++;
    fp_mul(u2_PALL,u2_PAL,h1_PALL);
    //countM++;

  
    fp_copy(sq_PALLLH,hlp_PALL);
    fp_sqr(sq_PALLLH,sq_PALLLH);
    //countS++;
    fp_sqr(sq_PALLLH,sq_PALLLH);
    //countS++;
    fp_sqr(sq_PALLLH,sq_PALLLH);
    //countS++;
    fp_prime_back(tmp_bn, sq_PALLLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PALLLH_H=_tmp>>1;
 
    GPOW_186_3(h1_PALLLH,(1u<<3)-c_PALLLH_H,gw);SELECT(h1_PALLLH,gpp[189],c_PALLLH_H==0,h1_PALLLH);
    //countM++;
    fp_mul(u2_PALLLH,hlp_PALL,h1_PALLLH);
    fp_prime_back(tmp_bn, u2_PALLLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PALLLH_L=_tmp>>1;
 
    uint64_t c_PALLLH=c_PALLLH_H+((c_PALLLH_L-1)&7u)*8;
    GPOW_180_6(h1_PALLLH,(1u<<6)-c_PALLLH,gw);SELECT(h1_PALLLH,gpp[186],c_PALLLH==0,h1_PALLLH);
    //countM++;
    fp_mul(u2_PALLLH,u2_PALL,h1_PALLLH);
    fp_copy(sq_PALLLL,u2_PALLLH);
    fp_sqr(sq_PALLLL,sq_PALLLL);
    //countS++;
    fp_sqr(sq_PALLLL,sq_PALLLL);
    //countS++;
    fp_sqr(sq_PALLLL,sq_PALLLL);
    //countS++;
    fp_prime_back(tmp_bn, sq_PALLLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PALLLL_H=_tmp>>1;
   
    GPOW_186_3(h1_PALLLL,(1u<<3)-c_PALLLL_H,gw);SELECT(h1_PALLLL,gpp[189],c_PALLLL_H==0,h1_PALLLL);
    //countM++;
    fp_mul(u2_PALLLL,u2_PALLLH,h1_PALLLL);
    fp_prime_back(tmp_bn, u2_PALLLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_PALLLL_L=_tmp>>1;

    uint64_t c_PALLLL=c_PALLLL_H+((c_PALLLL_L-1)&7u)*8;
    uint64_t c_PALLL=c_PALLLH+((c_PALLLL-1)&63u)*64;
    uint64_t c_PALL=c_PALLH+((c_PALLL-1)&0xFFFull)*4096;  /* 24-bit */
    uint64_t d_PAL=c_PALH+((c_PALL-1)&0xFFFFFFull)*(1ull<<24);  /* 48-bit */

  
    uint64_t dp1=(d_PAL-1)&MASK48;
    uint120_t c_A;
    c_A.low=(c_PAH|((dp1&0xFFFull)<<48))&LIMB_MASK;
    c_A.high=(dp1>>12)&LIMB_MASK;

  
    uint120_t exp_fA=rshift120(add120(sub120(POW96,c_A),make120(0,1)),1);
    GPOW_0_95(f_A,exp_fA,gw);
    bool cA0=(c_A.low==0&&c_A.high==0);
    SELECT(f_A,gpp[95],cA0,f_A);
    GPOW_48_96(hlp1_A,sub120(POW96,c_A),gw);
    SELECT(hlp1_A,gpp[144],cA0,hlp1_A);
    fp_mul(hlp_A,hlp_A,hlp1_A);
    //countM++;
    //countM++;   
    fp_sqr(u_X0,f_A);
    //countS++;
    fp_mul(u_X0,u_X0,u_A);


    fp_copy(h0_X0,hlp_A);
    fp_copy(h0_X0P,h0_X0);
    for(int j=0;j<24;j++){if(j==12)fp_copy(hlp_X0P,h0_X0P);fp_sqr(h0_X0P,h0_X0P);
    //countS++;
    }
    fp_copy(h0_X0PH,h0_X0P);
    for(int j=0;j<12;j++){if(j==6)fp_copy(hlp_X0PH,h0_X0PH);fp_sqr(h0_X0PH,h0_X0PH);
    //countS++;
    }

    fp_copy(h0_X0PHH,h0_X0PH);
    for(int j=0;j<6;j++){fp_sqr(h0_X0PHH,h0_X0PHH);
    //countS++;
    }
    fp_copy(sq_X0PHHH,h0_X0PHH);
    fp_sqr(sq_X0PHHH,sq_X0PHHH);
    //countS++;
    fp_sqr(sq_X0PHHH,sq_X0PHHH);
    //countS++;
    fp_sqr(sq_X0PHHH,sq_X0PHHH);
    //countS++;
    fp_prime_back(tmp_bn, sq_X0PHHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X0PHHH_H=_tmp>>1;
    
    GPOW_186_3(h1_X0PHHH,(1u<<3)-c_X0PHHH_H,gw);SELECT(h1_X0PHHH,gpp[189],c_X0PHHH_H==0,h1_X0PHHH);
    //countM++;
    fp_mul(u2_X0PHHH,h0_X0PHH,h1_X0PHHH);
    fp_prime_back(tmp_bn, u2_X0PHHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X0PHHH_L=_tmp>>1;
    
    uint64_t c_X0PHHH=c_X0PHHH_H+((c_X0PHHH_L-1)&7u)*8;
    GPOW_180_6(h1_X0PHHH,(1u<<6)-c_X0PHHH,gw);SELECT(h1_X0PHHH,gpp[186],c_X0PHHH==0,h1_X0PHHH);
    //countM++;
    fp_mul(u2_X0PHHH,h0_X0PH,h1_X0PHHH);
    fp_copy(sq_X0PHHL,u2_X0PHHH);
    fp_sqr(sq_X0PHHL,sq_X0PHHL);
    //countS++;
    fp_sqr(sq_X0PHHL,sq_X0PHHL);
    //countS++;
    fp_sqr(sq_X0PHHL,sq_X0PHHL);
    //countS++;
    fp_prime_back(tmp_bn, sq_X0PHHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X0PHHL_H=_tmp>>1;
    
    GPOW_186_3(h1_X0PHHL,(1u<<3)-c_X0PHHL_H,gw);SELECT(h1_X0PHHL,gpp[189],c_X0PHHL_H==0,h1_X0PHHL);
    //countM++;
    fp_mul(u2_X0PHHL,u2_X0PHHH,h1_X0PHHL);
    fp_prime_back(tmp_bn, u2_X0PHHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X0PHHL_L=_tmp>>1;
    
    uint64_t c_X0PHHL=c_X0PHHL_H+((c_X0PHHL_L-1)&7u)*8;
    uint64_t c_X0PHH=c_X0PHHH+((c_X0PHHL-1)&63u)*64;
    GPOW_168_12(h1_X0PH,(1u<<12)-c_X0PHH,gw);SELECT(h1_X0PH,gpp[180],c_X0PHH==0,h1_X0PH);
    GPOW_174_12(hlp1_X0PH,(1u<<12)-c_X0PHH,gw);SELECT(hlp1_X0PH,gpp[186],c_X0PHH==0,hlp1_X0PH);
    fp_mul(hlp_X0PH,hlp_X0PH,hlp1_X0PH);
    //countM++;
    fp_mul(u2_X0PH,h0_X0P,h1_X0PH);
    //countM++;
    
    fp_copy(sq_X0PHLH,hlp_X0PH);
    fp_sqr(sq_X0PHLH,sq_X0PHLH);
    //countS++;
    fp_sqr(sq_X0PHLH,sq_X0PHLH);
    //countS++;
    fp_sqr(sq_X0PHLH,sq_X0PHLH);
    //countS++;
    fp_prime_back(tmp_bn, sq_X0PHLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X0PHLH_H=_tmp>>1;
    
    GPOW_186_3(h1_X0PHLH,(1u<<3)-c_X0PHLH_H,gw);SELECT(h1_X0PHLH,gpp[189],c_X0PHLH_H==0,h1_X0PHLH);
    //countM++;
    fp_mul(u2_X0PHLH,hlp_X0PH,h1_X0PHLH);
    fp_prime_back(tmp_bn, u2_X0PHLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X0PHLH_L=_tmp>>1;
    
    uint64_t c_X0PHLH=c_X0PHLH_H+((c_X0PHLH_L-1)&7u)*8;
    GPOW_180_6(h1_X0PHLH,(1u<<6)-c_X0PHLH,gw);SELECT(h1_X0PHLH,gpp[186],c_X0PHLH==0,h1_X0PHLH);
    //countM++;
    fp_mul(u2_X0PHLH,u2_X0PH,h1_X0PHLH);
    fp_copy(sq_X0PHLL,u2_X0PHLH);
    fp_sqr(sq_X0PHLL,sq_X0PHLL);
    //countS++;
    fp_sqr(sq_X0PHLL,sq_X0PHLL);
    //countS++;
    fp_sqr(sq_X0PHLL,sq_X0PHLL);
    //countS++;
    fp_prime_back(tmp_bn, sq_X0PHLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X0PHLL_H=_tmp>>1;
    
    GPOW_186_3(h1_X0PHLL,(1u<<3)-c_X0PHLL_H,gw);SELECT(h1_X0PHLL,gpp[189],c_X0PHLL_H==0,h1_X0PHLL);
    //countM++;
    fp_mul(u2_X0PHLL,u2_X0PHLH,h1_X0PHLL);
    fp_prime_back(tmp_bn, u2_X0PHLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X0PHLL_L=_tmp>>1;
    
    uint64_t c_X0PHLL=c_X0PHLL_H+((c_X0PHLL_L-1)&7u)*8;
    uint64_t c_X0PHL=c_X0PHLH+((c_X0PHLL-1)&63u)*64;
    uint64_t c_X0PH=c_X0PHH+((c_X0PHL-1)&0xFFFull)*4096;
    GPOW_144_24(h1_X0P,(1ull<<24)-c_X0PH,gw);SELECT(h1_X0P,gpp[168],c_X0PH==0,h1_X0P);
    GPOW_156_24(hlp1_X0P,(1ull<<24)-c_X0PH,gw);SELECT(hlp1_X0P,gpp[180],c_X0PH==0,hlp1_X0P);
    fp_mul(hlp_X0P,hlp_X0P,hlp1_X0P);
    //countM++;
    fp_mul(u2_X0P,h0_X0,h1_X0P);
    //countM++;
    
    fp_copy(h0_X0PLH,hlp_X0P);
    for(int j=0;j<6;j++){fp_sqr(h0_X0PLH,h0_X0PLH);
    //countS++;
    }
    fp_copy(sq_X0PLHH,h0_X0PLH);
    fp_sqr(sq_X0PLHH,sq_X0PLHH);
    //countS++;
    fp_sqr(sq_X0PLHH,sq_X0PLHH);
    //countS++;
    fp_sqr(sq_X0PLHH,sq_X0PLHH);
    //countS++;
    fp_prime_back(tmp_bn, sq_X0PLHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X0PLHH_H=_tmp>>1;
  
    GPOW_186_3(h1_X0PLHH,(1u<<3)-c_X0PLHH_H,gw);SELECT(h1_X0PLHH,gpp[189],c_X0PLHH_H==0,h1_X0PLHH);
    //countM++;
    fp_mul(u2_X0PLHH,h0_X0PLH,h1_X0PLHH);
    fp_prime_back(tmp_bn, u2_X0PLHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X0PLHH_L=_tmp>>1;
   
    uint64_t c_X0PLHH=c_X0PLHH_H+((c_X0PLHH_L-1)&7u)*8;
    GPOW_180_6(h1_X0PLHH,(1u<<6)-c_X0PLHH,gw);SELECT(h1_X0PLHH,gpp[186],c_X0PLHH==0,h1_X0PLHH);
    //countM++;
    fp_mul(u2_X0PLHH,hlp_X0P,h1_X0PLHH);
    fp_copy(sq_X0PLHL,u2_X0PLHH);
    fp_sqr(sq_X0PLHL,sq_X0PLHL);
    //countS++;
    fp_sqr(sq_X0PLHL,sq_X0PLHL);
    //countS++;
    fp_sqr(sq_X0PLHL,sq_X0PLHL);
    //countS++;
    fp_prime_back(tmp_bn, sq_X0PLHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X0PLHL_H=_tmp>>1;
    
    GPOW_186_3(h1_X0PLHL,(1u<<3)-c_X0PLHL_H,gw);SELECT(h1_X0PLHL,gpp[189],c_X0PLHL_H==0,h1_X0PLHL);
    //countM++;
    fp_mul(u2_X0PLHL,u2_X0PLHH,h1_X0PLHL);
    fp_prime_back(tmp_bn, u2_X0PLHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X0PLHL_L=_tmp>>1;
    
    uint64_t c_X0PLHL=c_X0PLHL_H+((c_X0PLHL_L-1)&7u)*8;
    uint64_t c_X0PLH=c_X0PLHH+((c_X0PLHL-1)&63u)*64;
    GPOW_168_12(h1_X0PL,(1u<<12)-c_X0PLH,gw);SELECT(h1_X0PL,gpp[180],c_X0PLH==0,h1_X0PL);
    //countM++;
    fp_mul(u2_X0PL,u2_X0P,h1_X0PL);
   
    fp_copy(h0_X0PLL,u2_X0PL);
    for(int j=0;j<6;j++){fp_sqr(h0_X0PLL,h0_X0PLL);
    //countS++;
    }
    fp_copy(sq_X0PLLH,h0_X0PLL);
    fp_sqr(sq_X0PLLH,sq_X0PLLH);
    //countS++;
    fp_sqr(sq_X0PLLH,sq_X0PLLH);
    //countS++;
    fp_sqr(sq_X0PLLH,sq_X0PLLH);
    //countS++;
    fp_prime_back(tmp_bn, sq_X0PLLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X0PLLH_H=_tmp>>1;
    
    GPOW_186_3(h1_X0PLLH,(1u<<3)-c_X0PLLH_H,gw);SELECT(h1_X0PLLH,gpp[189],c_X0PLLH_H==0,h1_X0PLLH);
    //countM++;
    fp_mul(u2_X0PLLH,h0_X0PLL,h1_X0PLLH);
    fp_prime_back(tmp_bn, u2_X0PLLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X0PLLH_L=_tmp>>1;
    
    uint64_t c_X0PLLH=c_X0PLLH_H+((c_X0PLLH_L-1)&7u)*8;
    GPOW_180_6(h1_X0PLLH,(1u<<6)-c_X0PLLH,gw);SELECT(h1_X0PLLH,gpp[186],c_X0PLLH==0,h1_X0PLLH);
    //countM++;
    fp_mul(u2_X0PLLH,u2_X0PL,h1_X0PLLH);
    fp_copy(sq_X0PLLL,u2_X0PLLH);
    fp_sqr(sq_X0PLLL,sq_X0PLLL);
    //countS++;
    fp_sqr(sq_X0PLLL,sq_X0PLLL);
    //countS++;
    fp_sqr(sq_X0PLLL,sq_X0PLLL);
    //countS++;
    fp_prime_back(tmp_bn, sq_X0PLLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X0PLLL_H=_tmp>>1;
    
    GPOW_186_3(h1_X0PLLL,(1u<<3)-c_X0PLLL_H,gw);SELECT(h1_X0PLLL,gpp[189],c_X0PLLL_H==0,h1_X0PLLL);
    //countM++;
    fp_mul(u2_X0PLLL,u2_X0PLLH,h1_X0PLLL);
    fp_prime_back(tmp_bn, u2_X0PLLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X0PLLL_L=_tmp>>1;
    
    uint64_t c_X0PLLL=c_X0PLLL_H+((c_X0PLLL_L-1)&7u)*8;
    uint64_t c_X0PLL=c_X0PLLH+((c_X0PLLL-1)&63u)*64;
    uint64_t c_X0PL=c_X0PLH+((c_X0PLL-1)&0xFFFull)*4096;
    uint64_t c_X0=c_X0PH+((c_X0PL-1)&0xFFFFFFull)*(1ull<<24);  /* 48-bit */
    GPOW_95_48(f_X0,((1ull<<48)-c_X0)&MASK48,gw);SELECT(f_X0,gpp[143],c_X0==0,f_X0);
    //countM++;
    fp_sqr(u_X1,f_X0);
    //countS++;
    fp_mul(u_X1,u_X0,u_X1);

    
    fp_copy(h0_X1,u_X1);
    for(int j=0;j<24;j++){if(j==12)fp_copy(hlp_X1,h0_X1);fp_sqr(h0_X1,h0_X1);
    //countS++;
    }
    fp_copy(h0_X1H,h0_X1);
    for(int j=0;j<12;j++){if(j==6)fp_copy(hlp_X1H,h0_X1H);fp_sqr(h0_X1H,h0_X1H);
    //countS++;
    }
    
    fp_copy(h0_X1HH,h0_X1H);
    for(int j=0;j<6;j++){fp_sqr(h0_X1HH,h0_X1HH);
    //countS++;
    }
    fp_copy(sq_X1HHH,h0_X1HH);
    fp_sqr(sq_X1HHH,sq_X1HHH);
    //countS++;
    fp_sqr(sq_X1HHH,sq_X1HHH);
    //countS++;
    fp_sqr(sq_X1HHH,sq_X1HHH);
    //countS++;
    fp_prime_back(tmp_bn, sq_X1HHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X1HHH_H=_tmp>>1;
    
    GPOW_186_3(h1_X1HHH,(1u<<3)-c_X1HHH_H,gw);SELECT(h1_X1HHH,gpp[189],c_X1HHH_H==0,h1_X1HHH);
    //countM++;
    fp_mul(u2_X1HHH,h0_X1HH,h1_X1HHH);
    fp_prime_back(tmp_bn, u2_X1HHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X1HHH_L=_tmp>>1;
    
    uint64_t c_X1HHH=c_X1HHH_H+((c_X1HHH_L-1)&7u)*8;
    GPOW_180_6(h1_X1HHH,(1u<<6)-c_X1HHH,gw);SELECT(h1_X1HHH,gpp[186],c_X1HHH==0,h1_X1HHH);
    //countM++;
    fp_mul(u2_X1HHH,h0_X1H,h1_X1HHH);
    fp_copy(sq_X1HHL,u2_X1HHH);
    fp_sqr(sq_X1HHL,sq_X1HHL);
    //countS++;
    fp_sqr(sq_X1HHL,sq_X1HHL);
    //countS++;
    fp_sqr(sq_X1HHL,sq_X1HHL);
    //countS++;
    fp_prime_back(tmp_bn, sq_X1HHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X1HHL_H=_tmp>>1;
  
    GPOW_186_3(h1_X1HHL,(1u<<3)-c_X1HHL_H,gw);SELECT(h1_X1HHL,gpp[189],c_X1HHL_H==0,h1_X1HHL);
    //countM++;
    fp_mul(u2_X1HHL,u2_X1HHH,h1_X1HHL);
    fp_prime_back(tmp_bn, u2_X1HHL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X1HHL_L=_tmp>>1;
    
    uint64_t c_X1HHL=c_X1HHL_H+((c_X1HHL_L-1)&7u)*8;
    uint64_t c_X1HH=c_X1HHH+((c_X1HHL-1)&63u)*64;
    GPOW_168_12(h1_X1H,(1u<<12)-c_X1HH,gw);SELECT(h1_X1H,gpp[180],c_X1HH==0,h1_X1H);
    GPOW_174_12(hlp1_X1H,(1u<<12)-c_X1HH,gw);SELECT(hlp1_X1H,gpp[186],c_X1HH==0,hlp1_X1H);
    fp_mul(hlp_X1H,hlp_X1H,hlp1_X1H);
    //countM++;
    fp_mul(u2_X1H,h0_X1,h1_X1H);
    //countM++;
   
    fp_copy(sq_X1HLH,hlp_X1H);
    fp_sqr(sq_X1HLH,sq_X1HLH);
    //countS++;
    fp_sqr(sq_X1HLH,sq_X1HLH);
    //countS++;
    fp_sqr(sq_X1HLH,sq_X1HLH);
    //countS++;
    fp_prime_back(tmp_bn, sq_X1HLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X1HLH_H=_tmp>>1;
    
    GPOW_186_3(h1_X1HLH,(1u<<3)-c_X1HLH_H,gw);SELECT(h1_X1HLH,gpp[189],c_X1HLH_H==0,h1_X1HLH);
    //countM++;
    fp_mul(u2_X1HLH,hlp_X1H,h1_X1HLH);
    fp_prime_back(tmp_bn, u2_X1HLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X1HLH_L=_tmp>>1;
   
    uint64_t c_X1HLH=c_X1HLH_H+((c_X1HLH_L-1)&7u)*8;
    GPOW_180_6(h1_X1HLH,(1u<<6)-c_X1HLH,gw);SELECT(h1_X1HLH,gpp[186],c_X1HLH==0,h1_X1HLH);
    //countM++;
    fp_mul(u2_X1HLH,u2_X1H,h1_X1HLH);
    fp_copy(sq_X1HLL,u2_X1HLH);
    fp_sqr(sq_X1HLL,sq_X1HLL);
    //countS++;
    fp_sqr(sq_X1HLL,sq_X1HLL);
    //countS++;
    fp_sqr(sq_X1HLL,sq_X1HLL);
    //countS++;
    fp_prime_back(tmp_bn, sq_X1HLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X1HLL_H=_tmp>>1;
   
    GPOW_186_3(h1_X1HLL,(1u<<3)-c_X1HLL_H,gw);SELECT(h1_X1HLL,gpp[189],c_X1HLL_H==0,h1_X1HLL);
    //countM++;
    fp_mul(u2_X1HLL,u2_X1HLH,h1_X1HLL);
    fp_prime_back(tmp_bn, u2_X1HLL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X1HLL_L=_tmp>>1;
   
    uint64_t c_X1HLL=c_X1HLL_H+((c_X1HLL_L-1)&7u)*8;
    uint64_t c_X1HL=c_X1HLH+((c_X1HLL-1)&63u)*64;
    uint64_t c_X1H=c_X1HH+((c_X1HL-1)&0xFFFull)*4096;  /* 24-bit */
    GPOW_143_24(f_X1,(1ull<<24)-c_X1H,gw);SELECT(f_X1,gpp[167],c_X1H==0,f_X1);
    GPOW_156_24(hlp1_X1,(1ull<<24)-c_X1H,gw);SELECT(hlp1_X1,gpp[180],c_X1H==0,hlp1_X1);
    fp_mul(hlp_X1,hlp_X1,hlp1_X1);
    //countM++;
    //countM++;  
    fp_sqr(u_X2,f_X1);
    //countS++;
    fp_mul(u_X2,u_X1,u_X2);

   
    fp_copy(h0_X2,hlp_X1);
    fp_copy(h0_X2H,h0_X2);
    for(int j=0;j<6;j++){fp_sqr(h0_X2H,h0_X2H);
    //countS++;
    }
    fp_copy(sq_X2HH,h0_X2H);
    fp_sqr(sq_X2HH,sq_X2HH);
    //countS++;
    fp_sqr(sq_X2HH,sq_X2HH);
    //countS++;
    fp_sqr(sq_X2HH,sq_X2HH);
    //countS++;
    fp_prime_back(tmp_bn, sq_X2HH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X2HH_H=_tmp>>1;
    
    GPOW_186_3(h1_X2HH,(1u<<3)-c_X2HH_H,gw);SELECT(h1_X2HH,gpp[189],c_X2HH_H==0,h1_X2HH);
    //countM++;
    fp_mul(u2_X2HH,h0_X2H,h1_X2HH);
    fp_prime_back(tmp_bn, u2_X2HH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X2HH_L=_tmp>>1;
    
    uint64_t c_X2HH=c_X2HH_H+((c_X2HH_L-1)&7u)*8;
    GPOW_180_6(h1_X2HH,(1u<<6)-c_X2HH,gw);SELECT(h1_X2HH,gpp[186],c_X2HH==0,h1_X2HH);
    //countM++;
    fp_mul(u2_X2HH,h0_X2,h1_X2HH);
    fp_copy(sq_X2HL,u2_X2HH);
    fp_sqr(sq_X2HL,sq_X2HL);
    //countS++;
    fp_sqr(sq_X2HL,sq_X2HL);
    //countS++;
    fp_sqr(sq_X2HL,sq_X2HL);
    //countS++;
    fp_prime_back(tmp_bn, sq_X2HL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X2HL_H=_tmp>>1;
    
    GPOW_186_3(h1_X2HL,(1u<<3)-c_X2HL_H,gw);SELECT(h1_X2HL,gpp[189],c_X2HL_H==0,h1_X2HL);
    //countM++;
    fp_mul(u2_X2HL,u2_X2HH,h1_X2HL);
    fp_prime_back(tmp_bn, u2_X2HL);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X2HL_L=_tmp>>1;
    
    uint64_t c_X2HL=c_X2HL_H+((c_X2HL_L-1)&7u)*8;
    uint64_t c_X2H=c_X2HH+((c_X2HL-1)&63u)*64;  /* 12-bit */
    GPOW_167_12(f_X2,(1u<<12)-c_X2H,gw);SELECT(f_X2,gpp[179],c_X2H==0,f_X2);
    //countM++;
    fp_sqr(u_X3,f_X2);
    //countS++;
    fp_mul(u_X3,u_X2,u_X3);

   
    fp_copy(h0_X3,u_X3);
    for(int j=0;j<6;j++){fp_sqr(h0_X3,h0_X3);
    //countS++;
    }

    fp_copy(sq_X3H,h0_X3);
    fp_sqr(sq_X3H,sq_X3H);
    //countS++;
    fp_sqr(sq_X3H,sq_X3H);
    //countS++;
    fp_sqr(sq_X3H,sq_X3H);
    //countS++;
    fp_prime_back(tmp_bn, sq_X3H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X3H_H=_tmp>>1;

    GPOW_186_3(h1_X3H,(1u<<3)-c_X3H_H,gw);SELECT(h1_X3H,gpp[189],c_X3H_H==0,h1_X3H);
    //countM++;
    fp_mul(u2_X3H,h0_X3,h1_X3H);
    fp_prime_back(tmp_bn, u2_X3H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X3H_L=_tmp>>1;

    uint64_t c_X3=c_X3H_H+((c_X3H_L-1)&7u)*8; 
    GPOW_179_6(f_X3,(1u<<6)-c_X3,gw);SELECT(f_X3,gpp[185],c_X3==0,f_X3);
    //countM++;
    fp_sqr(u_X4,f_X3);
    //countS++;
    fp_mul(u_X4,u_X3,u_X4);


    fp_copy(h0_X4,u_X4);
    fp_sqr(h0_X4,h0_X4);
    //countS++;
    fp_sqr(h0_X4,h0_X4);
    //countS++;
    fp_sqr(h0_X4,h0_X4);
    //countS++;
    fp_prime_back(tmp_bn, h0_X4);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c_X4=_tmp>>1;

    GPOW_185_3(f_X4,(1u<<3)-c_X4,gw);SELECT(f_X4,gpp[188],c_X4==0,f_X4);
    //countM++;
    fp_sqr(u_X5,f_X4);
    //countS++;
    fp_mul(u_X5,u_X4,u_X5);

    fp_prime_back(tmp_bn, u_X5);
    bn_get_dig(&d, tmp_bn);
    d=d&0x1f;
    _tmp=rlll[d];
    uint64_t c1=_tmp>>1;

    fp_copy(t_X4,fll[c1<<1]);

    fp_mul(t_X4,f_X4,t_X4);
    //countM++;
  
    fp_mul(t_X3,f_X3,t_X4);
    //countM++;
 
    fp_mul(t_X2,f_X2,t_X3);
    //countM++;
    
    fp_mul(t_X1,f_X1,t_X2);
    //countM++;
    
    fp_mul(t_X0,f_X0,t_X1);
    //countM++;
   
    fp_mul(t_A,f_A,t_X0);
    //countM++;
    
    } 

    fp_copy(out_d,t_A);
    bn_free(tmp_bn);
#define FF(v) fp_free(v)
    FF(h0_A);FF(h0_PA);FF(h0_PAH);FF(h0_PAHH);FF(hlp_A);FF(hlp_PA);FF(hlp_PAH);FF(hlp_PAHH);
    FF(h0_PAHHH);FF(sq_PAHHHH);FF(h1_PAHHHH);FF(u2_PAHHHH);FF(sq_PAHHHL);FF(h1_PAHHHL);FF(u2_PAHHHL);
    FF(sq_PAHHLH);FF(h1_PAHHLH);FF(u2_PAHHLH);FF(sq_PAHHLL);FF(h1_PAHHLL);FF(u2_PAHHLL);
    FF(h1_PAHH);FF(hlp1_PAHH);FF(u2_PAHH);
    FF(h0_PAHLH);FF(sq_PAHLHH);FF(h1_PAHLHH);FF(u2_PAHLHH);FF(sq_PAHLHL);FF(h1_PAHLHL);FF(u2_PAHLHL);
    FF(h1_PAHL);FF(u2_PAHL);
    FF(h0_PAHLL);FF(sq_PAHLLH);FF(h1_PAHLLH);FF(u2_PAHLLH);FF(sq_PAHLLL);FF(h1_PAHLLL);FF(u2_PAHLLL);
    FF(h1_PAH);FF(hlp1_PAH);FF(u2_PAH);
    FF(h0_PALH);FF(hlp_PALH);
    FF(h0_PALHH);FF(sq_PALHHH);FF(h1_PALHHH);FF(u2_PALHHH);FF(sq_PALHHL);FF(h1_PALHHL);FF(u2_PALHHL);
    FF(h1_PALH);FF(hlp1_PALH);FF(u2_PALH);
    FF(sq_PALHLH);FF(h1_PALHLH);FF(u2_PALHLH);FF(sq_PALHLL);FF(h1_PALHLL);FF(u2_PALHLL);
    FF(h1_PAL);FF(u2_PAL);
    FF(h0_PALL);FF(hlp_PALL);
    FF(h0_PALLH);FF(sq_PALLHH);FF(h1_PALLHH);FF(u2_PALLHH);FF(sq_PALLHL);FF(h1_PALLHL);FF(u2_PALLHL);
    FF(h1_PALL);FF(hlp1_PALL);FF(u2_PALL);
    FF(sq_PALLLH);FF(h1_PALLLH);FF(u2_PALLLH);FF(sq_PALLLL);FF(h1_PALLLL);FF(u2_PALLLL);
    FF(h1_PA);FF(hlp1_PA);FF(u2_PA);
    FF(f_A);FF(hlp1_A);FF(u_X0);
    FF(h0_X0);FF(h0_X0P);FF(hlp_X0P);FF(h0_X0PH);FF(hlp_X0PH);
    FF(h0_X0PHH);FF(sq_X0PHHH);FF(h1_X0PHHH);FF(u2_X0PHHH);FF(sq_X0PHHL);FF(h1_X0PHHL);FF(u2_X0PHHL);
    FF(h1_X0PH);FF(hlp1_X0PH);FF(u2_X0PH);
    FF(sq_X0PHLH);FF(h1_X0PHLH);FF(u2_X0PHLH);FF(sq_X0PHLL);FF(h1_X0PHLL);FF(u2_X0PHLL);
    FF(h1_X0P);FF(hlp1_X0P);FF(u2_X0P);
    FF(h0_X0PLH);FF(sq_X0PLHH);FF(h1_X0PLHH);FF(u2_X0PLHH);FF(sq_X0PLHL);FF(h1_X0PLHL);FF(u2_X0PLHL);
    FF(h1_X0PL);FF(u2_X0PL);
    FF(h0_X0PLL);FF(sq_X0PLLH);FF(h1_X0PLLH);FF(u2_X0PLLH);FF(sq_X0PLLL);FF(h1_X0PLLL);FF(u2_X0PLLL);
    FF(f_X0);FF(u_X1);
    FF(h0_X1);FF(hlp_X1);FF(h0_X1H);FF(hlp_X1H);
    FF(h0_X1HH);FF(sq_X1HHH);FF(h1_X1HHH);FF(u2_X1HHH);FF(sq_X1HHL);FF(h1_X1HHL);FF(u2_X1HHL);
    FF(h1_X1H);FF(hlp1_X1H);FF(u2_X1H);
    FF(sq_X1HLH);FF(h1_X1HLH);FF(u2_X1HLH);FF(sq_X1HLL);FF(h1_X1HLL);FF(u2_X1HLL);
    FF(f_X1);FF(hlp1_X1);FF(u_X2);
    FF(h0_X2);FF(h0_X2H);FF(sq_X2HH);FF(h1_X2HH);FF(u2_X2HH);FF(sq_X2HL);FF(h1_X2HL);FF(u2_X2HL);
    FF(f_X2);FF(u_X3);
    FF(h0_X3);FF(sq_X3H);FF(h1_X3H);FF(u2_X3H);
    FF(f_X3);FF(u_X4);
    FF(h0_X4);FF(f_X4);FF(u_X5);FF(t_X4);FF(t_X3);FF(t_X2);FF(t_X1);FF(t_X0);FF(t_A);
#undef FF
}

/* ── SQRT ────────────────────────────────────────────────────────────────── */
void SQRT(fp_t x,fp_t y,bn_t e,
          fp_t gw[NW][WE], int rlll[32], fp_t rll[WE],fp_t fll[WE],fp_t gpp[N]){
    fp_t u,v,w_,d;fp_null(u);fp_null(v);fp_null(w_);fp_null(d);
    fp_new(u);fp_new(v);fp_new(w_);fp_new(d);
    fp_exp(v,x,e);
    fp_mul(w_,x,v);
    //countM++;
    fp_mul(u,w_,v);
    //countM++;
    solve_dlp_pow2_flat(u,d,gw,rlll,rll,fll,gpp);
    fp_mul(y,w_,d);//countM++;
    //fp_print(x);fp_sqr(y,y);fp_print(y);
    fp_free(u);fp_free(v);fp_free(w_);fp_free(d);}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(void){
    if(core_init()!=RLC_OK){core_clean();return 1;}
    if(fp_param_set_any_pmers()!=RLC_OK){printf("Curve init failed\n");core_clean();return 1;}
    printf("n=192, w=4\n");
    fp_t gw[NW][WE],rll[WE],fll[WE],gpp[N],g,z,h,hh,b,y;
    int rlll[32]={0};
    rlll[ 0x1 ]= 0 ;
    rlll[ 0x19 ]= 1 ;
    rlll[ 0x1b ]= 2 ;
    rlll[ 0x12 ]= 3 ;
    rlll[ 0x3 ]= 4 ;
    rlll[ 0x11 ]= 5 ;
    rlll[ 0xd ]= 6 ;
    rlll[ 0xc ]= 7 ;
    rlll[ 0x0 ]= 8 ;
    rlll[ 0x8 ]= 9 ;
    rlll[ 0x6 ]= 10 ;
    rlll[ 0xf ]= 11 ;
    rlll[ 0x1e ]= 12 ;
    rlll[ 0x10 ]= 13 ;
    rlll[ 0x14 ]= 14 ;
    rlll[ 0x15 ]= 15 ;
    for(int i=0;i<NW;i++)for(int j=0;j<WE;j++){fp_null(gw[i][j]);fp_new(gw[i][j]);}
    for(int i=0;i<WE;i++){fp_null(rll[i]);fp_new(rll[i]);}
    for(int i=0;i<WE;i++){fp_null(fll[i]);fp_new(fll[i]);}
    for(int i=0;i<N; i++){fp_null(gpp[i]);fp_new(gpp[i]);}
    fp_null(b);fp_new(b);fp_null(y);fp_new(y);
    fp_null(g);fp_new(g);fp_null(z);fp_new(z);fp_null(h);fp_new(h);fp_null(hh);fp_new(hh);
    fp_rand(b);while(!fp_is_sqr(b))fp_rand(b);
    bn_t tmp,m,e,a1;
    bn_null(e);bn_new(e);bn_null(m);bn_new(m);bn_null(tmp);bn_new(tmp);bn_null(a1);bn_new(a1);
    bn_read_str(tmp,"3",1,16);fp_prime_conv(z,tmp);
    bn_read_str(m,"800000000000011",32,16);fp_exp(g,z,m);//fp_print(z);fp_print(g);
    bn_read_str(e,"400000000000008",32,16);
    /* h=g^(2^188): 0x100000000000000000000000000000000000000000000000 (48 hex digits) */
    bn_read_str(a1,"100000000000000000000000000000000000000000000000",48,16);
    fp_exp(h,g,a1);fp_srt(hh,h);fp_inv(hh,hh);//fp_print(h);
    precomputation(g,h,hh,gw,rll,fll,gpp);
    MEASURE(SQRT(b,y,e,gw,rlll,rll,fll,gpp);)
    printf("RDTSC_clk_min    = %f\n",RDTSC_clk_min);
    printf("RDTSC_clk_median = %f\n",RDTSC_clk_median);
    printf("RDTSC_clk_max    = %f\n",RDTSC_clk_max);
    printf("mult_count = %d\n",countM);
    printf("sqr_count  = %d\n",countS);
    for(int i=0;i<NW;i++)for(int j=0;j<WE;j++)fp_free(gw[i][j]);
    for(int i=0;i<WE;i++)fp_free(rll[i]);
    for(int i=0;i<WE;i++)fp_free(fll[i]);
    for(int i=0;i<N; i++)fp_free(gpp[i]);
    fp_free(b);fp_free(y);fp_free(g);fp_free(z);fp_free(h);fp_free(hh);
    bn_free(e);bn_free(m);bn_free(a1);bn_free(tmp);
    core_clean();return 0;}
