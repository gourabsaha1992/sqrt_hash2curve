#include <stdio.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include<immintrin.h>
#include <relic/relic_core.h>
#include "relic/relic_md.h"

#define w 8
#define we 256
#define n 96
#define k3 1
#define nw 12
#define count 0 


int shift=0;
int i,j,ii,jj,i3;
int l[k3]= {95} ;
int kk[k3+1][k3+1]= {{0, 0}, {1, 0}} ;
int s[k3]= {0} ;
int t[k3]= {11} ;
int t_max= 11 ;
int r[k3]= {1} ;
int ep[k3+1][k3+1]= {{0, 0}, {0, 0}} ;
int rho[k3+1][k3+1]= {{0, 0}, {0, 0}} ;

/* ---------------- 120-bit fixed-width integer support ---------------- */
#define LIMB_BITS 60
#define LIMB_MASK ((1ULL << LIMB_BITS) - 1)

typedef struct { uint64_t low; uint64_t high; } uint120_t;

static inline uint120_t make120(uint64_t h, uint64_t l)
{ return (uint120_t){.low = l & LIMB_MASK, .high = h & LIMB_MASK}; }

static inline uint120_t add120(uint120_t a, uint120_t b) {
    uint64_t r = a.low + b.low, c = r >> LIMB_BITS;
    return make120((a.high + b.high + c) & LIMB_MASK, r & LIMB_MASK);
}

static inline uint120_t sub120(uint120_t a, uint120_t b) {
    uint64_t r = (1ULL << LIMB_BITS) + a.low - b.low, bw = 1 - (r >> LIMB_BITS);
    return make120((a.high - b.high - bw) & LIMB_MASK, r & LIMB_MASK);
}

static inline uint120_t rshift120(uint120_t a, int sh) {
    if (sh <= 0)  return a;
    if (sh >= 120) return make120(0, 0);
    if (sh < LIMB_BITS)
        return make120((a.high >> sh) & LIMB_MASK,
                       ((a.low >> sh) | (a.high << (LIMB_BITS - sh))) & LIMB_MASK);
    return make120(0, (a.high >> (sh - LIMB_BITS)) & LIMB_MASK);
}

static inline uint120_t lshift120(uint120_t a, int sh) {
    if (sh <= 0)  return a;
    if (sh >= 120) return make120(0, 0);
    if (sh < LIMB_BITS)
        return make120(((a.high << sh) | (a.low >> (LIMB_BITS - sh))) & LIMB_MASK,
                        (a.low << sh) & LIMB_MASK);
    return make120((a.low << (sh - LIMB_BITS)) & LIMB_MASK, 0);
}


#define MASK95_LOW  LIMB_MASK                     
#define MASK95_HIGH ((1ULL << 35) - 1)



static inline uint8_t chunk120_w8(uint120_t e, int k) {
    int bit = k * w;   
    if (bit + w <= LIMB_BITS)                 
        return (uint8_t)((e.low  >> bit) & 0xFF);
    if (bit >= LIMB_BITS)                     
        return (uint8_t)((e.high >> (bit - LIMB_BITS)) & 0xFF);
    /* straddles */
    uint64_t lo_part = e.low  >> bit;          
    uint64_t hi_part = e.high << (LIMB_BITS - bit); 
    return (uint8_t)((lo_part | hi_part) & 0xFF);
}


#define REPEAT 100000
#define REREPEAT 1
#define WARMUP (REPEAT/4)

unsigned long long RDTSC_start_clk, RDTSC_end_clk;
double RDTSC_clk[REPEAT];
double RDTSC_clk_min, RDTSC_clk_median, RDTSC_clk_max;
double min, l1;
int RDTSC_MEASURE_ITERATOR;
int RDTSC_MEASURE_REITERATOR;
int SCHED_RET_VAL;
int iii,jjj,ttt;
uint64_t d;
uint64_t c1,c,one=1;
uint64_t and_f=255;

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
{\
    for(RDTSC_MEASURE_ITERATOR=0; RDTSC_MEASURE_ITERATOR< WARMUP; RDTSC_MEASURE_ITERATOR++) \
    { \
        {x}; \
    }; \
    for (RDTSC_MEASURE_ITERATOR = 0; RDTSC_MEASURE_ITERATOR < REPEAT; RDTSC_MEASURE_ITERATOR++) \
    { \
        RDTSC_start_clk = get_Clks(); \
        {x}; \
        RDTSC_end_clk = get_Clks(); \
        RDTSC_clk[RDTSC_MEASURE_ITERATOR] = (double)(RDTSC_end_clk-RDTSC_start_clk); \
    }; \
    for (iii = 0; iii < REPEAT; iii++){ \
        min =   RDTSC_clk[iii]; \
        for (jjj = iii+1; jjj< REPEAT; jjj++){ \
            if (min > RDTSC_clk[jjj]){ \
                min = RDTSC_clk[jjj]; \
                ttt = jjj; \
            } \
        } \
        l1 = RDTSC_clk[ttt]; RDTSC_clk[ttt] = RDTSC_clk[iii]; RDTSC_clk[iii] = l1; \
    }; \
    RDTSC_clk_min = RDTSC_clk[0]; \
    RDTSC_clk_median = RDTSC_clk[REPEAT/2]; \
    RDTSC_clk_max = RDTSC_clk[REPEAT-1];\
}


void precomputation(fp_t g, fp_t tabl2[we][nw]) {
    bn_t temp, one, itemp;
    bn_null(temp); bn_new(temp);
    bn_null(one);  bn_new(one);
    bn_set_dig(one, 1);
    bn_null(itemp); bn_new(itemp);

    for (int vv = 0; vv < we; vv++) {
        for (int i = 0; i < nw; i++) {
            bn_set_dig(itemp, vv);
            bn_lsh(temp, one, i * w);
            bn_mul(temp, temp, itemp);
            fp_exp(tabl2[vv][i], g, temp);
        }
    }
    bn_free(temp); bn_free(one); bn_free(itemp);
}

uint120_t Qacc,term,shifted;

void findsqroot(fp_t u, fp_t y, fp_t tabl2[we][nw], int tab3[8192],
                uint64_t q[k3][t_max+1], bn_t e, bn_t tmp) {
    fp_t v, v1, x[k3], gamma, alpha;
    fp_t temp, temp1;
    fp_t f, a[t_max+1];

    
    uint120_t s_acc, tt_acc, qq_acc;

    fp_null(f);
    for (i = 0; i < t_max+1; i++) { fp_null(a[i]); fp_new(a[i]); }
    fp_null(temp);
    fp_null(temp1);
    fp_null(v);
    fp_null(v1);
    fp_null(gamma);
    fp_null(alpha);

    RLC_TRY {
        fp_new(v);
        fp_new(v1);
        fp_new(gamma);
        fp_new(alpha);
        fp_new(temp);
        fp_new(temp1);
        fp_new(f);
        
        
        fp_exp(v, u, e);
        fp_mul(v1, v, u);
        fp_mul(x[k3-1], v, v1);
        s_acc  = make120(0, 0);
        tt_acc = make120(0, 0);

        i3 = 0;
        fp_set_dig(gamma, 1);
        fp_copy(a[t[i3]], x[i3]);
        for (i = 1; i < t[i3]+1; i++) {
            fp_copy(a[t[i3]-i], a[t[i3]+1-i]);
            for (j = 1; j <= w; j++) {
                fp_sqr(a[t[i3]-i], a[t[i3]-i]);
            }
        }

        for (j = 0; j < t[i3]+1; j++) {
            fp_copy(f, a[j]);
            if (j != 0) {
                fp_copy(temp, tabl2[q[i3][0]][s[i3]+t[i3]-j]);
                for (i = 1; i <= r[i3]; i++) {
                    fp_sqr(temp, temp);
                }
                fp_mul(f, f, temp);
            }
            for (i = 1; i < j; i++) {
                fp_mul(f, f, tabl2[q[i3][i]][nw-(j+1-i)]);
            }
            fp_prime_back(tmp, f);
            bn_get_dig(&d, tmp);
            d = d & 0x1fff;
            if (j == 0) {
                q[i3][j] = tab3[d] >> (w - l[k3-1] + t[k3-1]*w);
            } else {
                q[i3][j] = tab3[d];
            }
        }
        Qacc = make120(0, q[i3][0]);
        for (i = 1; i < t[i3]+1; i++) {
            term = lshift120(make120(0, q[i3][i]),
                                       l[i3] - (t[i3]+1-i)*w);
            Qacc = add120(Qacc, term);
        }
        s_acc = lshift120(Qacc, n - l[i3]);
        tt_acc = add120(s_acc, tt_acc);
        for (ii = k3-1; ii >= 0; ii--) {
            shifted = rshift120(tt_acc, kk[k3][ii]);
            qq_acc = make120(shifted.high & MASK95_HIGH,
                             shifted.low  & MASK95_LOW);
        }
        for (j = 0; j < k3; j++) {
            fp_set_dig(temp, 1);
            for (jj = 0; jj < t[j]+1; jj++) {
                q[j][jj] = chunk120_w8(qq_acc, jj);
            }
            
            qq_acc = rshift120(qq_acc, (t[j]+1) * w);

            for (jj = 0; jj < t[j]+1; jj++) {
                fp_mul(temp, temp, tabl2[q[j][jj]][jj + ep[k3][j]]);
            }
            for (jj = 1; jj <= rho[k3][j]; jj++) {
                fp_sqr(temp, temp);
            }
            fp_mul(gamma, gamma, temp);
        }

        fp_mul(y, v1, gamma);
        //fp_sqr(y,y);
	//fp_print(y);
	//fp_print(u);
    }

    RLC_CATCH_ANY {
        RLC_THROW(ERR_CAUGHT);
    }
    RLC_FINALLY {
        fp_free(v);
        fp_free(v1);
        fp_free(gamma);
        fp_free(alpha);
        fp_free(temp);
        fp_free(temp1);
        fp_free(f);
        for (i = 0; i < t_max+1; i++) fp_free(a[i]);
    }
}


int main(void) {
    if (core_init() != RLC_OK) { core_clean(); return 1; }

    if (fp_param_set_any_pmers() != RLC_OK) {
        printf("Curve initialization failed\n");
        core_clean(); return 1;
    }

    printf("Using Primes of NIST-224\n");

    fp_t tabl2[we][nw], g, z, b, h, h1, h2, y;
    uint64_t q[k3][t_max+1];
    int tab3[8192] = {0};
    bn_t tmp, m, e;

    for (i = 0; i < we; i++)
        for (j = 0; j < nw; j++) { fp_null(tabl2[i][j]); fp_new(tabl2[i][j]); }

    tab3[ 0x1 ]= 0 ;
    tab3[ 0x115d ]= 1 ;
    tab3[ 0x19a6 ]= 2 ;
    tab3[ 0x4fa ]= 3 ;
    tab3[ 0x172e ]= 4 ;
    tab3[ 0x380 ]= 5 ;
    tab3[ 0x188b ]= 6 ;
    tab3[ 0x178e ]= 7 ;
    tab3[ 0x1d00 ]= 8 ;
    tab3[ 0xdd4 ]= 9 ;
    tab3[ 0xeb2 ]= 10 ;
    tab3[ 0x24f ]= 11 ;
    tab3[ 0x58f ]= 12 ;
    tab3[ 0x83f ]= 13 ;
    tab3[ 0x113b ]= 14 ;
    tab3[ 0x1d69 ]= 15 ;
    tab3[ 0x1355 ]= 16 ;
    tab3[ 0xd4c ]= 17 ;
    tab3[ 0xaa8 ]= 18 ;
    tab3[ 0x47b ]= 19 ;
    tab3[ 0x89c ]= 20 ;
    tab3[ 0x75f ]= 21 ;
    tab3[ 0x158d ]= 22 ;
    tab3[ 0x1272 ]= 23 ;
    tab3[ 0x116b ]= 24 ;
    tab3[ 0x1fe8 ]= 25 ;
    tab3[ 0x49 ]= 26 ;
    tab3[ 0x1352 ]= 27 ;
    tab3[ 0x198 ]= 28 ;
    tab3[ 0x1851 ]= 29 ;
    tab3[ 0x7 ]= 30 ;
    tab3[ 0x1bab ]= 31 ;
    tab3[ 0xef8 ]= 32 ;
    tab3[ 0x840 ]= 33 ;
    tab3[ 0x5d8 ]= 34 ;
    tab3[ 0x1678 ]= 35 ;
    tab3[ 0x1c7e ]= 36 ;
    tab3[ 0xd84 ]= 37 ;
    tab3[ 0x1e53 ]= 38 ;
    tab3[ 0x15f3 ]= 39 ;
    tab3[ 0x1ef7 ]= 40 ;
    tab3[ 0x294 ]= 41 ;
    tab3[ 0xe4 ]= 42 ;
    tab3[ 0xf35 ]= 43 ;
    tab3[ 0x199e ]= 44 ;
    tab3[ 0x1b62 ]= 45 ;
    tab3[ 0xa06 ]= 46 ;
    tab3[ 0x128d ]= 47 ;
    tab3[ 0x1749 ]= 48 ;
    tab3[ 0x1fc1 ]= 49 ;
    tab3[ 0xce6 ]= 50 ;
    tab3[ 0x148 ]= 51 ;
    tab3[ 0x197a ]= 52 ;
    tab3[ 0x1e04 ]= 53 ;
    tab3[ 0x1390 ]= 54 ;
    tab3[ 0x1fb0 ]= 55 ;
    tab3[ 0xba3 ]= 56 ;
    tab3[ 0x1092 ]= 57 ;
    tab3[ 0x3a3 ]= 58 ;
    tab3[ 0x544 ]= 59 ;
    tab3[ 0x1076 ]= 60 ;
    tab3[ 0x8c ]= 61 ;
    tab3[ 0x1b09 ]= 62 ;
    tab3[ 0x18c3 ]= 63 ;
    tab3[ 0x1e8 ]= 64 ;
    tab3[ 0x11e5 ]= 65 ;
    tab3[ 0x37 ]= 66 ;
    tab3[ 0x49e ]= 67 ;
    tab3[ 0xfff ]= 68 ;
    tab3[ 0xec8 ]= 69 ;
    tab3[ 0x1fa1 ]= 70 ;
    tab3[ 0xa30 ]= 71 ;
    tab3[ 0x1c85 ]= 72 ;
    tab3[ 0xbea ]= 73 ;
    tab3[ 0x95d ]= 74 ;
    tab3[ 0x1c18 ]= 75 ;
    tab3[ 0xca ]= 76 ;
    tab3[ 0x3f5 ]= 77 ;
    tab3[ 0x1daa ]= 78 ;
    tab3[ 0x950 ]= 79 ;
    tab3[ 0x651 ]= 80 ;
    tab3[ 0xc39 ]= 81 ;
    tab3[ 0x47c ]= 82 ;
    tab3[ 0x38e ]= 83 ;
    tab3[ 0xddf ]= 84 ;
    tab3[ 0x1e78 ]= 85 ;
    tab3[ 0xbd5 ]= 86 ;
    tab3[ 0x258 ]= 87 ;
    tab3[ 0x89f ]= 88 ;
    tab3[ 0x13cb ]= 89 ;
    tab3[ 0x101a ]= 90 ;
    tab3[ 0x678 ]= 91 ;
    tab3[ 0x1bfc ]= 92 ;
    tab3[ 0x34b ]= 93 ;
    tab3[ 0xea6 ]= 94 ;
    tab3[ 0x1e5 ]= 95 ;
    tab3[ 0x1244 ]= 96 ;
    tab3[ 0x59e ]= 97 ;
    tab3[ 0x919 ]= 98 ;
    tab3[ 0x222 ]= 99 ;
    tab3[ 0x193c ]= 100 ;
    tab3[ 0x7e6 ]= 101 ;
    tab3[ 0xa93 ]= 102 ;
    tab3[ 0x1231 ]= 103 ;
    tab3[ 0x54e ]= 104 ;
    tab3[ 0x18f ]= 105 ;
    tab3[ 0x6e2 ]= 106 ;
    tab3[ 0x291 ]= 107 ;
    tab3[ 0x15c ]= 108 ;
    tab3[ 0xcad ]= 109 ;
    tab3[ 0x19e0 ]= 110 ;
    tab3[ 0x1152 ]= 111 ;
    tab3[ 0x10d2 ]= 112 ;
    tab3[ 0x36f ]= 113 ;
    tab3[ 0x574 ]= 114 ;
    tab3[ 0x176 ]= 115 ;
    tab3[ 0x1421 ]= 116 ;
    tab3[ 0x55f ]= 117 ;
    tab3[ 0x13f1 ]= 118 ;
    tab3[ 0x4bc ]= 119 ;
    tab3[ 0x12e ]= 120 ;
    tab3[ 0x10c2 ]= 121 ;
    tab3[ 0x18d2 ]= 122 ;
    tab3[ 0x9d2 ]= 123 ;
    tab3[ 0x14b6 ]= 124 ;
    tab3[ 0x185b ]= 125 ;
    tab3[ 0x15f0 ]= 126 ;
    tab3[ 0x1044 ]= 127 ;
    tab3[ 0x0 ]= 128 ;
    tab3[ 0xea4 ]= 129 ;
    tab3[ 0x65b ]= 130 ;
    tab3[ 0x1b07 ]= 131 ;
    tab3[ 0x8d3 ]= 132 ;
    tab3[ 0x1c81 ]= 133 ;
    tab3[ 0x776 ]= 134 ;
    tab3[ 0x873 ]= 135 ;
    tab3[ 0x301 ]= 136 ;
    tab3[ 0x122d ]= 137 ;
    tab3[ 0x114f ]= 138 ;
    tab3[ 0x1db2 ]= 139 ;
    tab3[ 0x1a72 ]= 140 ;
    tab3[ 0x17c2 ]= 141 ;
    tab3[ 0xec6 ]= 142 ;
    tab3[ 0x298 ]= 143 ;
    tab3[ 0xcac ]= 144 ;
    tab3[ 0x12b5 ]= 145 ;
    tab3[ 0x1559 ]= 146 ;
    tab3[ 0x1b86 ]= 147 ;
    tab3[ 0x1765 ]= 148 ;
    tab3[ 0x18a2 ]= 149 ;
    tab3[ 0xa74 ]= 150 ;
    tab3[ 0xd8f ]= 151 ;
    tab3[ 0xe96 ]= 152 ;
    tab3[ 0x19 ]= 153 ;
    tab3[ 0x1fb8 ]= 154 ;
    tab3[ 0xcaf ]= 155 ;
    tab3[ 0x1e69 ]= 156 ;
    tab3[ 0x7b0 ]= 157 ;
    tab3[ 0x1ffa ]= 158 ;
    tab3[ 0x456 ]= 159 ;
    tab3[ 0x1109 ]= 160 ;
    tab3[ 0x17c1 ]= 161 ;
    tab3[ 0x1a29 ]= 162 ;
    tab3[ 0x989 ]= 163 ;
    tab3[ 0x383 ]= 164 ;
    tab3[ 0x127d ]= 165 ;
    tab3[ 0x1ae ]= 166 ;
    tab3[ 0xa0e ]= 167 ;
    tab3[ 0x10a ]= 168 ;
    tab3[ 0x1d6d ]= 169 ;
    tab3[ 0x1f1d ]= 170 ;
    tab3[ 0x10cc ]= 171 ;
    tab3[ 0x663 ]= 172 ;
    tab3[ 0x49f ]= 173 ;
    tab3[ 0x15fb ]= 174 ;
    tab3[ 0xd74 ]= 175 ;
    tab3[ 0x8b8 ]= 176 ;
    tab3[ 0x40 ]= 177 ;
    tab3[ 0x131b ]= 178 ;
    tab3[ 0x1eb9 ]= 179 ;
    tab3[ 0x687 ]= 180 ;
    tab3[ 0x1fd ]= 181 ;
    tab3[ 0xc71 ]= 182 ;
    tab3[ 0x51 ]= 183 ;
    tab3[ 0x145e ]= 184 ;
    tab3[ 0xf6f ]= 185 ;
    tab3[ 0x1c5e ]= 186 ;
    tab3[ 0x1abd ]= 187 ;
    tab3[ 0xf8b ]= 188 ;
    tab3[ 0x1f75 ]= 189 ;
    tab3[ 0x4f8 ]= 190 ;
    tab3[ 0x73e ]= 191 ;
    tab3[ 0x1e19 ]= 192 ;
    tab3[ 0xe1c ]= 193 ;
    tab3[ 0x1fca ]= 194 ;
    tab3[ 0x1b63 ]= 195 ;
    tab3[ 0x1002 ]= 196 ;
    tab3[ 0x1139 ]= 197 ;
    tab3[ 0x60 ]= 198 ;
    tab3[ 0x15d1 ]= 199 ;
    tab3[ 0x37c ]= 200 ;
    tab3[ 0x1417 ]= 201 ;
    tab3[ 0x16a4 ]= 202 ;
    tab3[ 0x3e9 ]= 203 ;
    tab3[ 0x1f37 ]= 204 ;
    tab3[ 0x1c0c ]= 205 ;
    tab3[ 0x257 ]= 206 ;
    tab3[ 0x16b1 ]= 207 ;
    tab3[ 0x19b0 ]= 208 ;
    tab3[ 0x13c8 ]= 209 ;
    tab3[ 0x1b85 ]= 210 ;
    tab3[ 0x1c73 ]= 211 ;
    tab3[ 0x1222 ]= 212 ;
    tab3[ 0x189 ]= 213 ;
    tab3[ 0x142c ]= 214 ;
    tab3[ 0x1da9 ]= 215 ;
    tab3[ 0x1762 ]= 216 ;
    tab3[ 0xc36 ]= 217 ;
    tab3[ 0xfe7 ]= 218 ;
    tab3[ 0x1989 ]= 219 ;
    tab3[ 0x405 ]= 220 ;
    tab3[ 0x1cb6 ]= 221 ;
    tab3[ 0x115b ]= 222 ;
    tab3[ 0x1e1c ]= 223 ;
    tab3[ 0xdbd ]= 224 ;
    tab3[ 0x1a63 ]= 225 ;
    tab3[ 0x16e8 ]= 226 ;
    tab3[ 0x1ddf ]= 227 ;
    tab3[ 0x6c5 ]= 228 ;
    tab3[ 0x181b ]= 229 ;
    tab3[ 0x156e ]= 230 ;
    tab3[ 0xdd0 ]= 231 ;
    tab3[ 0x1ab3 ]= 232 ;
    tab3[ 0x1e72 ]= 233 ;
    tab3[ 0x191f ]= 234 ;
    tab3[ 0x1d70 ]= 235 ;
    tab3[ 0x1ea5 ]= 236 ;
    tab3[ 0x1354 ]= 237 ;
    tab3[ 0x621 ]= 238 ;
    tab3[ 0xeaf ]= 239 ;
    tab3[ 0xf2f ]= 240 ;
    tab3[ 0x1c92 ]= 241 ;
    tab3[ 0x1a8d ]= 242 ;
    tab3[ 0x1e8b ]= 243 ;
    tab3[ 0xbe0 ]= 244 ;
    tab3[ 0x1aa2 ]= 245 ;
    tab3[ 0xc10 ]= 246 ;
    tab3[ 0x1b45 ]= 247 ;
    tab3[ 0x1ed3 ]= 248 ;
    tab3[ 0xf3f ]= 249 ;
    tab3[ 0x72f ]= 250 ;
    tab3[ 0x162f ]= 251 ;
    tab3[ 0xb4b ]= 252 ;
    tab3[ 0x7a6 ]= 253 ;
    tab3[ 0xa11 ]= 254 ;
    tab3[ 0xfbd ]= 255 ;

    fp_null(b);  fp_new(b);
    fp_null(y);  fp_new(y);
    fp_null(g);  fp_new(g);
    fp_null(z);  fp_new(z);
    fp_null(h);  fp_new(h);
    fp_null(h1); fp_new(h1);
    fp_null(h2); fp_new(h2);
    bn_null(tmp); bn_new(tmp);
    bn_null(m);   bn_new(m);
    bn_null(e);   bn_new(e);

    fp_rand(b);
    while (fp_is_sqr(b) != 1) fp_rand(b);
    fp_print(b);

    bn_read_str(tmp, "b", 1, 16);
    fp_prime_conv(z, tmp);

    bn_read_str(m, "ffffffffffffffffffffffffffffffff", 32, 16);
    fp_exp(g, z, m);

    bn_read_str(e, "7fffffffffffffffffffffffffffffff", 32, 16);



    precomputation(g, tabl2);

    printf("using tonelli-shank with look up table\n");
    printf("W=%d\nK=%d\n", w, k3);

    /* ll (bn_t) removed — mask is now the compile-time constant MASK95_* */
    MEASURE(findsqroot(b, y, tabl2, tab3, q, e, tmp);)

    printf("RDTSC_clk_min=%f\n",    RDTSC_clk_min);
    printf("RDTSC_clk_median=%f\n", RDTSC_clk_median);
    printf("RDTSC_clk_max=%f\n",    RDTSC_clk_max);
    printf("Shift count=%d\n",      shift);

    for (i = 0; i < we; i++)
        for (j = 0; j < nw; j++) fp_free(tabl2[i][j]);

    fp_free(b); fp_free(y); fp_free(g); fp_free(z);
    fp_free(h); fp_free(h1); fp_free(h2);
    bn_free(tmp); bn_free(e); bn_free(m);

    core_clean();
    return 0;
}
