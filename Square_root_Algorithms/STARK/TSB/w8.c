#include <stdio.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include<immintrin.h>

#include <relic/relic_core.h>
#include "relic/relic_md.h"

#define w 8
#define we 256
#define n 192
#define k1 24
#define l 23
#define rem 0
#define rem_e 1
int i,j,ii,jj,tt;
dig_t d;


void precomputation(fp_t g,fp_t h,fp_t h1,fp_t tab1[k1][we]){
    bn_t temp,one,itemp;
    bn_null(temp);
    bn_new(temp);
    bn_null(one);
    bn_new(one);
    bn_set_dig(one,1);
    bn_null(itemp);
    bn_new(itemp);
    
    for(int i=0;i<k1;i++){
            for(int j=0;j<we;j++){
                bn_lsh(temp,one,i*w);
                bn_mul_dig(temp,temp,j);
                fp_exp(tab1[i][j],g,temp);
                fp_inv(tab1[i][j],tab1[i][j]);
            }
    }
  
}


void findsqroot(fp_t u, fp_t y,fp_t tab1[k1][we],int tab2[16384],bn_t ee, bn_t tmp) {
	fp_t v,v1,v2,x[k1],temp;
        int e1[k1]={0};
	for(int i=0; i<k1;i++){
	    fp_null(x[i]);
	    
	}
	fp_null(temp);
	fp_null(v);
	fp_null(v1);
	fp_null(v2);
	RLC_TRY {
		for(int i=0; i<k1;i++){
		    fp_new(x[i]);
		    
		}
		
		fp_new(v);
		fp_new(v1);
		fp_new(v2);
	        fp_new(temp);

                
		fp_exp(v,u,ee);
		
		fp_mul(v1,v,u);
		
		fp_mul(x[0],v,v1);
		for(int i=1; i<k1;i++){
		    fp_copy(x[i],x[i-1]);
		    for(int ii=0; ii<w ; ii++){
		        fp_sqr(x[i],x[i]);
		    }
		    //fp_print(x[i]);
		}
		
		fp_prime_back(tmp, x[k1-1]);
	        
	        bn_get_dig(&d, tmp);
	        d=d&0x3fff;
	        e1[0]=tab2[d];
	        //printf("\n\n%d\n",e1[0]);
		
		
		for (int k = 1; k <= l; k++) {

                    fp_copy(temp, x[l - k]);
                    fp_mul(temp, temp, tab1[l - k][e1[0]]);
                    for (int j = 2; j <= k; j++) {
                        fp_mul(temp, temp, tab1[l + 1 - j][e1[k - j + 1]]);
                    }
                    fp_prime_back(tmp, temp);
	            bn_get_dig(&d, tmp);
	            d=d&0x3fff;
	            e1[k]=tab2[d];
	            //printf("\n%d\n",e1[k]);
                }

		
	        int d1[k1];

                for (int i = 0; i < k1; i++) {
                    d1[i] = e1[i] >> 1;
                    if (i < k1 - 1) {
                        int carry = e1[i+1] & 1;
                        d1[i] |= carry << (w - 1);
                    }
                }
                fp_copy(temp,tab1[0][d1[0]]);
	        for(int i=1;i<k1;i++){
	            fp_mul(temp,temp,tab1[i][d1[i]]);
	        }
	        
	        fp_mul(y,v1,temp);
	        //fp_sqr(y,y);
	        //fp_print(y);
	        
	}

	RLC_CATCH_ANY {
		RLC_THROW(ERR_CAUGHT);
	}
	RLC_FINALLY {
		for(int i=0; i<k1;i++){
	            fp_free(x[i]);
	        }
		fp_free(v);
		fp_free(temp);
		fp_free(v1);
		fp_free(v2);
	}

}


int squareroot(fp_t c, const fp_t a, bn_t m, fp_t t3) {
	bn_t e;
	fp_t t0, t1, t2;
	int f = 0, r = 0;

	bn_null(e);
	fp_null(t0);
	fp_null(t1);
	fp_null(t2);
	

	if (fp_is_zero(a)) {
		fp_zero(c);
		return 1;
	}

	RLC_TRY {
		bn_new(e);
		fp_new(t0);
		fp_new(t1);
		fp_new(t2);
		

		/* Make e = p. */
		e->used = RLC_FP_DIGS;
		dv_copy(e->dp, fp_prime_get(), RLC_FP_DIGS);

		
			
				/* Implement constant-time version of Tonelli-Shanks algorithm
				 * as per https://eprint.iacr.org/2020/1497.pdf */

				/* Compute progenitor as x^(p-1-2^f)/2^(f+1) where 2^f|(p-1). */

				/* Write p - 1 as (e * 2^f), odd e. */
				
				
				
				

				/* Make it e = (p - 1 - 2^f)/2^(f + 1), compute t0 = a^e. */
				bn_rsh(m, m, 1);
				fp_exp(t0, a, m);

				/* Recover 2^f-root of unity, and continue algorithm. */
				

				fp_sqr(t1, t0);
				fp_mul(t1, t1, a);
				fp_mul(c, t0, a);
				for (int j = n; j > 1; j--) {
					fp_copy(t2, t1);
					for (int i = 1; i < j - 1; i++) {
						fp_sqr(t2, t2);
					}
					fp_mul(t0, c, t3);
					fp_copy_sec(c, t0, fp_cmp_dig(t2, 1) != RLC_EQ);
					fp_sqr(t3, t3);
					fp_mul(t0, t1, t3);
					fp_copy_sec(t1, t0, fp_cmp_dig(t2, 1) != RLC_EQ);
				}
				//fp_sqr(c,c);
				//fp_print(c);

		
	}
	RLC_CATCH_ANY {
		RLC_THROW(ERR_CAUGHT);
	}
	RLC_FINALLY {
		bn_free(e);
		fp_free(t0);
		fp_free(t1);
		fp_free(t2);
	}
	return r;
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
	for (ii = 0; ii < REPEAT; ii++){ \
		min = 	RDTSC_clk[ii]; \
		for (jj = ii+1; jj< REPEAT; jj++){ \
			if (min > RDTSC_clk[jj]){ \
				min = RDTSC_clk[jj]; \
				tt = jj; \
			} \
		} \
		l1 = RDTSC_clk[tt]; RDTSC_clk[tt] = RDTSC_clk[ii]; RDTSC_clk[ii] = l1; \
	}; \
	RDTSC_clk_min = RDTSC_clk[0]; \
	RDTSC_clk_median = RDTSC_clk[REPEAT/2]; \
	RDTSC_clk_max = RDTSC_clk[REPEAT-1];\
}
















int main(void) {   
	if (core_init() != RLC_OK) {
        core_clean();
        return 1;
    }

    if (fp_param_set_any_pmers()!= RLC_OK) {
        printf("Error setting BLS24 curve parameters.\n");
        core_clean();
        return 1;
    }

    printf("Using Primes of bls12-377snark\n");

    fp_t tab1[k1][we],g,z,b,h,h1,y,t3;
    bn_t ee,m,tmp;
    dig_t q[4][4],qq[3];
    int tab2[16384]={0};

    for(int i=0;i<k1;i++){
        for(int j=0;j<we;j++){
            fp_null(tab1[i][j]);
            fp_new(tab1[i][j]);
        }
    }

    tab2[ 0x1 ]= 0 ;
tab2[ 0x3347 ]= 1 ;
tab2[ 0x320f ]= 2 ;
tab2[ 0x12c3 ]= 3 ;
tab2[ 0x3765 ]= 4 ;
tab2[ 0x3d98 ]= 5 ;
tab2[ 0x35c7 ]= 6 ;
tab2[ 0x3af9 ]= 7 ;
tab2[ 0x28ab ]= 8 ;
tab2[ 0x3079 ]= 9 ;
tab2[ 0x1a8e ]= 10 ;
tab2[ 0x1205 ]= 11 ;
tab2[ 0x104a ]= 12 ;
tab2[ 0x1153 ]= 13 ;
tab2[ 0x328e ]= 14 ;
tab2[ 0x23ae ]= 15 ;
tab2[ 0x2539 ]= 16 ;
tab2[ 0x25cc ]= 17 ;
tab2[ 0x2432 ]= 18 ;
tab2[ 0x1fae ]= 19 ;
tab2[ 0x8fe ]= 20 ;
tab2[ 0x36e6 ]= 21 ;
tab2[ 0x327d ]= 22 ;
tab2[ 0x37b3 ]= 23 ;
tab2[ 0x6d7 ]= 24 ;
tab2[ 0xff7 ]= 25 ;
tab2[ 0x1a07 ]= 26 ;
tab2[ 0x2272 ]= 27 ;
tab2[ 0xb1e ]= 28 ;
tab2[ 0x3e09 ]= 29 ;
tab2[ 0x2666 ]= 30 ;
tab2[ 0x1c06 ]= 31 ;
tab2[ 0x2ebb ]= 32 ;
tab2[ 0x3506 ]= 33 ;
tab2[ 0x28a9 ]= 34 ;
tab2[ 0x1e3f ]= 35 ;
tab2[ 0x1f4d ]= 36 ;
tab2[ 0x3c98 ]= 37 ;
tab2[ 0x1a69 ]= 38 ;
tab2[ 0x212e ]= 39 ;
tab2[ 0x70c ]= 40 ;
tab2[ 0x2b7e ]= 41 ;
tab2[ 0x372e ]= 42 ;
tab2[ 0x1ebe ]= 43 ;
tab2[ 0x1f32 ]= 44 ;
tab2[ 0x3b40 ]= 45 ;
tab2[ 0x3467 ]= 46 ;
tab2[ 0x2023 ]= 47 ;
tab2[ 0x1f52 ]= 48 ;
tab2[ 0x27d3 ]= 49 ;
tab2[ 0xe0f ]= 50 ;
tab2[ 0x2b2c ]= 51 ;
tab2[ 0x3e3d ]= 52 ;
tab2[ 0x2342 ]= 53 ;
tab2[ 0x319b ]= 54 ;
tab2[ 0x2abb ]= 55 ;
tab2[ 0xc71 ]= 56 ;
tab2[ 0x7ad ]= 57 ;
tab2[ 0x337 ]= 58 ;
tab2[ 0x2df ]= 59 ;
tab2[ 0x3534 ]= 60 ;
tab2[ 0x1644 ]= 61 ;
tab2[ 0x338c ]= 62 ;
tab2[ 0xe84 ]= 63 ;
tab2[ 0x37e3 ]= 64 ;
tab2[ 0x36db ]= 65 ;
tab2[ 0x3c1f ]= 66 ;
tab2[ 0x93d ]= 67 ;
tab2[ 0x3388 ]= 68 ;
tab2[ 0x3a85 ]= 69 ;
tab2[ 0x1476 ]= 70 ;
tab2[ 0x2a25 ]= 71 ;
tab2[ 0x2e55 ]= 72 ;
tab2[ 0x18aa ]= 73 ;
tab2[ 0x116d ]= 74 ;
tab2[ 0x3d90 ]= 75 ;
tab2[ 0x2063 ]= 76 ;
tab2[ 0xfb0 ]= 77 ;
tab2[ 0x1142 ]= 78 ;
tab2[ 0x394 ]= 79 ;
tab2[ 0x38d1 ]= 80 ;
tab2[ 0x9fc ]= 81 ;
tab2[ 0x325d ]= 82 ;
tab2[ 0x2590 ]= 83 ;
tab2[ 0x1c6d ]= 84 ;
tab2[ 0x211e ]= 85 ;
tab2[ 0x870 ]= 86 ;
tab2[ 0x3ad5 ]= 87 ;
tab2[ 0x2e75 ]= 88 ;
tab2[ 0x30e9 ]= 89 ;
tab2[ 0x19b ]= 90 ;
tab2[ 0x66 ]= 91 ;
tab2[ 0x2095 ]= 92 ;
tab2[ 0x163e ]= 93 ;
tab2[ 0x121b ]= 94 ;
tab2[ 0x392b ]= 95 ;
tab2[ 0x35cd ]= 96 ;
tab2[ 0x822 ]= 97 ;
tab2[ 0x17f2 ]= 98 ;
tab2[ 0x31d2 ]= 99 ;
tab2[ 0x182a ]= 100 ;
tab2[ 0x902 ]= 101 ;
tab2[ 0x112 ]= 102 ;
tab2[ 0x2c9f ]= 103 ;
tab2[ 0x3715 ]= 104 ;
tab2[ 0x1a6c ]= 105 ;
tab2[ 0x27ef ]= 106 ;
tab2[ 0x1f48 ]= 107 ;
tab2[ 0x2548 ]= 108 ;
tab2[ 0x3bba ]= 109 ;
tab2[ 0x287e ]= 110 ;
tab2[ 0x1308 ]= 111 ;
tab2[ 0x158c ]= 112 ;
tab2[ 0x185f ]= 113 ;
tab2[ 0xc07 ]= 114 ;
tab2[ 0xfb9 ]= 115 ;
tab2[ 0x35f5 ]= 116 ;
tab2[ 0x19c5 ]= 117 ;
tab2[ 0x17f1 ]= 118 ;
tab2[ 0x18a3 ]= 119 ;
tab2[ 0x10c0 ]= 120 ;
tab2[ 0x181d ]= 121 ;
tab2[ 0x132f ]= 122 ;
tab2[ 0xe74 ]= 123 ;
tab2[ 0x1929 ]= 124 ;
tab2[ 0x1789 ]= 125 ;
tab2[ 0x29af ]= 126 ;
tab2[ 0xb0c ]= 127 ;
tab2[ 0x0 ]= 128 ;
tab2[ 0xcba ]= 129 ;
tab2[ 0xdf2 ]= 130 ;
tab2[ 0x2d3e ]= 131 ;
tab2[ 0x89c ]= 132 ;
tab2[ 0x269 ]= 133 ;
tab2[ 0xa3a ]= 134 ;
tab2[ 0x508 ]= 135 ;
tab2[ 0x1756 ]= 136 ;
tab2[ 0xf88 ]= 137 ;
tab2[ 0x2573 ]= 138 ;
tab2[ 0x2dfc ]= 139 ;
tab2[ 0x2fb7 ]= 140 ;
tab2[ 0x2eae ]= 141 ;
tab2[ 0xd73 ]= 142 ;
tab2[ 0x1c53 ]= 143 ;
tab2[ 0x1ac8 ]= 144 ;
tab2[ 0x1a35 ]= 145 ;
tab2[ 0x1bcf ]= 146 ;
tab2[ 0x2053 ]= 147 ;
tab2[ 0x3703 ]= 148 ;
tab2[ 0x91b ]= 149 ;
tab2[ 0xd84 ]= 150 ;
tab2[ 0x84e ]= 151 ;
tab2[ 0x392a ]= 152 ;
tab2[ 0x300a ]= 153 ;
tab2[ 0x25fa ]= 154 ;
tab2[ 0x1d8f ]= 155 ;
tab2[ 0x34e3 ]= 156 ;
tab2[ 0x1f8 ]= 157 ;
tab2[ 0x199b ]= 158 ;
tab2[ 0x23fb ]= 159 ;
tab2[ 0x1146 ]= 160 ;
tab2[ 0xafb ]= 161 ;
tab2[ 0x1758 ]= 162 ;
tab2[ 0x21c2 ]= 163 ;
tab2[ 0x20b4 ]= 164 ;
tab2[ 0x369 ]= 165 ;
tab2[ 0x2598 ]= 166 ;
tab2[ 0x1ed3 ]= 167 ;
tab2[ 0x38f5 ]= 168 ;
tab2[ 0x1483 ]= 169 ;
tab2[ 0x8d3 ]= 170 ;
tab2[ 0x2143 ]= 171 ;
tab2[ 0x20cf ]= 172 ;
tab2[ 0x4c1 ]= 173 ;
tab2[ 0xb9a ]= 174 ;
tab2[ 0x1fde ]= 175 ;
tab2[ 0x20af ]= 176 ;
tab2[ 0x182e ]= 177 ;
tab2[ 0x31f2 ]= 178 ;
tab2[ 0x14d5 ]= 179 ;
tab2[ 0x1c4 ]= 180 ;
tab2[ 0x1cbf ]= 181 ;
tab2[ 0xe66 ]= 182 ;
tab2[ 0x1546 ]= 183 ;
tab2[ 0x3390 ]= 184 ;
tab2[ 0x3854 ]= 185 ;
tab2[ 0x3cca ]= 186 ;
tab2[ 0x3d22 ]= 187 ;
tab2[ 0xacd ]= 188 ;
tab2[ 0x29bd ]= 189 ;
tab2[ 0xc75 ]= 190 ;
tab2[ 0x317d ]= 191 ;
tab2[ 0x81e ]= 192 ;
tab2[ 0x926 ]= 193 ;
tab2[ 0x3e2 ]= 194 ;
tab2[ 0x36c4 ]= 195 ;
tab2[ 0xc79 ]= 196 ;
tab2[ 0x57c ]= 197 ;
tab2[ 0x2b8b ]= 198 ;
tab2[ 0x15dc ]= 199 ;
tab2[ 0x11ac ]= 200 ;
tab2[ 0x2757 ]= 201 ;
tab2[ 0x2e94 ]= 202 ;
tab2[ 0x271 ]= 203 ;
tab2[ 0x1f9e ]= 204 ;
tab2[ 0x3051 ]= 205 ;
tab2[ 0x2ebf ]= 206 ;
tab2[ 0x3c6d ]= 207 ;
tab2[ 0x730 ]= 208 ;
tab2[ 0x3605 ]= 209 ;
tab2[ 0xda4 ]= 210 ;
tab2[ 0x1a71 ]= 211 ;
tab2[ 0x2394 ]= 212 ;
tab2[ 0x1ee3 ]= 213 ;
tab2[ 0x3791 ]= 214 ;
tab2[ 0x52c ]= 215 ;
tab2[ 0x118c ]= 216 ;
tab2[ 0xf18 ]= 217 ;
tab2[ 0x3e66 ]= 218 ;
tab2[ 0x3f9b ]= 219 ;
tab2[ 0x1f6c ]= 220 ;
tab2[ 0x29c3 ]= 221 ;
tab2[ 0x2de6 ]= 222 ;
tab2[ 0x6d6 ]= 223 ;
tab2[ 0xa34 ]= 224 ;
tab2[ 0x37df ]= 225 ;
tab2[ 0x280f ]= 226 ;
tab2[ 0xe2f ]= 227 ;
tab2[ 0x27d7 ]= 228 ;
tab2[ 0x36ff ]= 229 ;
tab2[ 0x3eef ]= 230 ;
tab2[ 0x1362 ]= 231 ;
tab2[ 0x8ec ]= 232 ;
tab2[ 0x2595 ]= 233 ;
tab2[ 0x1812 ]= 234 ;
tab2[ 0x20b9 ]= 235 ;
tab2[ 0x1ab9 ]= 236 ;
tab2[ 0x447 ]= 237 ;
tab2[ 0x1783 ]= 238 ;
tab2[ 0x2cf9 ]= 239 ;
tab2[ 0x2a75 ]= 240 ;
tab2[ 0x27a2 ]= 241 ;
tab2[ 0x33fa ]= 242 ;
tab2[ 0x3048 ]= 243 ;
tab2[ 0xa0c ]= 244 ;
tab2[ 0x263c ]= 245 ;
tab2[ 0x2810 ]= 246 ;
tab2[ 0x275e ]= 247 ;
tab2[ 0x2f41 ]= 248 ;
tab2[ 0x27e4 ]= 249 ;
tab2[ 0x2cd2 ]= 250 ;
tab2[ 0x318d ]= 251 ;
tab2[ 0x26d8 ]= 252 ;
tab2[ 0x2878 ]= 253 ;
tab2[ 0x1652 ]= 254 ;
tab2[ 0x34f5 ]= 255 ;    
    

    
    
    
    
    
    
   
    fp_null(b);
    fp_new(b);
    
    fp_null(g);
    fp_new(g);
    
    fp_null(h);
    fp_new(h);

    fp_null(h1);
    fp_new(h1);
    fp_null(y);
    fp_new(y);
    
    fp_null(z);
    fp_new(z);
    
    fp_null(t3);
    fp_new(t3);
    
    bn_null(tmp);
    bn_new(tmp);
    
    bn_null(ee);
    bn_new(ee);
    
    bn_null(m);
    bn_new(m);
    
    fp_rand(b);
    while(fp_is_sqr(b)!=1){
        fp_rand(b);
    }
    fp_print(b);
    //bn_read_str(tmp,"4",1,16);
    //fp_prime_conv(b,tmp);    
    //fp_print(b);

    bn_read_str(tmp,"3",1,16);
    fp_prime_conv(z,tmp);
    
    bn_read_str(m,"800000000000011",32,16);
    fp_exp(g,z,m);
    
    bn_read_str(ee,"400000000000008",32,16);
    
    //h=g^(2^(n-w))=g^(2^(190))
    bn_t a1;
    bn_new(a1);
    bn_null(a1);
    bn_read_str(a1,"400000000000000000000000000000000000000000000000",48,16);
    fp_exp(h,g,a1);
    //h1=g^(2^(n-rem))=g^(2^(192))
    bn_read_str(a1,"1000000000000000000000000000000000000000000000000",49,16);
    fp_exp(h1,g,a1);
    
    
    precomputation(g,h,h1,tab1);

    printf("using tonelli-shank\n");
    
    fp_copy(t3, fp_prime_get_srt());
    
    MEASURE(squareroot(y, b, m, t3);)
    
    printf("RDTSC_clk_min=%f\n",RDTSC_clk_min);
    printf("RDTSC_clk_median=%f\n",RDTSC_clk_median);
    printf("RDTSC_clk_max=%f\n",RDTSC_clk_max);

    printf("using tonelli-shank look up table by bernstein\n");

    MEASURE(findsqroot(b,y,tab1,tab2,ee,tmp);)
    printf("RDTSC_clk_min=%f\n",RDTSC_clk_min);
    printf("RDTSC_clk_median=%f\n",RDTSC_clk_median);
    printf("RDTSC_clk_max=%f\n",RDTSC_clk_max);
    
    
    for(int i=0;i<k1;i++){
        for(int j=0;j<we;j++){
            fp_free(tab1[i][j]);
        }
    }

    

   
    fp_free(b);
    fp_free(y);
    bn_free(tmp);
    bn_free(ee);
    bn_free(m);
    fp_free(g);
    fp_free(z);
    fp_free(h);
    fp_free(h1);
    fp_free(t3);
    bn_free(a1);
    
    core_clean();
    return 0;
}

