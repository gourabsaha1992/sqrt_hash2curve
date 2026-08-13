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
#define k1 12
#define l 11
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
tab2[ 0xfbd ]= 1 ;
tab2[ 0x2a11 ]= 2 ;
tab2[ 0x27a6 ]= 3 ;
tab2[ 0xb4b ]= 4 ;
tab2[ 0x162f ]= 5 ;
tab2[ 0x272f ]= 6 ;
tab2[ 0x2f3f ]= 7 ;
tab2[ 0x1ed3 ]= 8 ;
tab2[ 0x3b45 ]= 9 ;
tab2[ 0xc10 ]= 10 ;
tab2[ 0x3aa2 ]= 11 ;
tab2[ 0x2be0 ]= 12 ;
tab2[ 0x1e8b ]= 13 ;
tab2[ 0x3a8d ]= 14 ;
tab2[ 0x1c92 ]= 15 ;
tab2[ 0x2f2f ]= 16 ;
tab2[ 0x2eaf ]= 17 ;
tab2[ 0x621 ]= 18 ;
tab2[ 0x1354 ]= 19 ;
tab2[ 0x3ea5 ]= 20 ;
tab2[ 0x1d70 ]= 21 ;
tab2[ 0x191f ]= 22 ;
tab2[ 0x1e72 ]= 23 ;
tab2[ 0x3ab3 ]= 24 ;
tab2[ 0x2dd0 ]= 25 ;
tab2[ 0x156e ]= 26 ;
tab2[ 0x381b ]= 27 ;
tab2[ 0x26c5 ]= 28 ;
tab2[ 0x1ddf ]= 29 ;
tab2[ 0x16e8 ]= 30 ;
tab2[ 0x1a63 ]= 31 ;
tab2[ 0xdbd ]= 32 ;
tab2[ 0x3e1c ]= 33 ;
tab2[ 0x115b ]= 34 ;
tab2[ 0x1cb6 ]= 35 ;
tab2[ 0x2405 ]= 36 ;
tab2[ 0x3989 ]= 37 ;
tab2[ 0xfe7 ]= 38 ;
tab2[ 0x2c36 ]= 39 ;
tab2[ 0x1762 ]= 40 ;
tab2[ 0x1da9 ]= 41 ;
tab2[ 0x142c ]= 42 ;
tab2[ 0x2189 ]= 43 ;
tab2[ 0x1222 ]= 44 ;
tab2[ 0x3c73 ]= 45 ;
tab2[ 0x3b85 ]= 46 ;
tab2[ 0x13c8 ]= 47 ;
tab2[ 0x39b0 ]= 48 ;
tab2[ 0x36b1 ]= 49 ;
tab2[ 0x2257 ]= 50 ;
tab2[ 0x1c0c ]= 51 ;
tab2[ 0x3f37 ]= 52 ;
tab2[ 0x23e9 ]= 53 ;
tab2[ 0x16a4 ]= 54 ;
tab2[ 0x1417 ]= 55 ;
tab2[ 0x237c ]= 56 ;
tab2[ 0x35d1 ]= 57 ;
tab2[ 0x60 ]= 58 ;
tab2[ 0x1139 ]= 59 ;
tab2[ 0x3002 ]= 60 ;
tab2[ 0x3b63 ]= 61 ;
tab2[ 0x3fca ]= 62 ;
tab2[ 0x2e1c ]= 63 ;
tab2[ 0x3e19 ]= 64 ;
tab2[ 0x73e ]= 65 ;
tab2[ 0x24f8 ]= 66 ;
tab2[ 0x1f75 ]= 67 ;
tab2[ 0xf8b ]= 68 ;
tab2[ 0x3abd ]= 69 ;
tab2[ 0x3c5e ]= 70 ;
tab2[ 0x2f6f ]= 71 ;
tab2[ 0x345e ]= 72 ;
tab2[ 0x51 ]= 73 ;
tab2[ 0xc71 ]= 74 ;
tab2[ 0x21fd ]= 75 ;
tab2[ 0x687 ]= 76 ;
tab2[ 0x1eb9 ]= 77 ;
tab2[ 0x331b ]= 78 ;
tab2[ 0x40 ]= 79 ;
tab2[ 0x28b8 ]= 80 ;
tab2[ 0xd74 ]= 81 ;
tab2[ 0x15fb ]= 82 ;
tab2[ 0x49f ]= 83 ;
tab2[ 0x2663 ]= 84 ;
tab2[ 0x10cc ]= 85 ;
tab2[ 0x1f1d ]= 86 ;
tab2[ 0x1d6d ]= 87 ;
tab2[ 0x210a ]= 88 ;
tab2[ 0x2a0e ]= 89 ;
tab2[ 0x1ae ]= 90 ;
tab2[ 0x127d ]= 91 ;
tab2[ 0x383 ]= 92 ;
tab2[ 0x2989 ]= 93 ;
tab2[ 0x1a29 ]= 94 ;
tab2[ 0x17c1 ]= 95 ;
tab2[ 0x3109 ]= 96 ;
tab2[ 0x456 ]= 97 ;
tab2[ 0x1ffa ]= 98 ;
tab2[ 0x27b0 ]= 99 ;
tab2[ 0x1e69 ]= 100 ;
tab2[ 0xcaf ]= 101 ;
tab2[ 0x1fb8 ]= 102 ;
tab2[ 0x19 ]= 103 ;
tab2[ 0xe96 ]= 104 ;
tab2[ 0xd8f ]= 105 ;
tab2[ 0x2a74 ]= 106 ;
tab2[ 0x18a2 ]= 107 ;
tab2[ 0x3765 ]= 108 ;
tab2[ 0x3b86 ]= 109 ;
tab2[ 0x1559 ]= 110 ;
tab2[ 0x12b5 ]= 111 ;
tab2[ 0x2cac ]= 112 ;
tab2[ 0x298 ]= 113 ;
tab2[ 0x2ec6 ]= 114 ;
tab2[ 0x17c2 ]= 115 ;
tab2[ 0x1a72 ]= 116 ;
tab2[ 0x3db2 ]= 117 ;
tab2[ 0x314f ]= 118 ;
tab2[ 0x322d ]= 119 ;
tab2[ 0x301 ]= 120 ;
tab2[ 0x873 ]= 121 ;
tab2[ 0x2776 ]= 122 ;
tab2[ 0x1c81 ]= 123 ;
tab2[ 0x28d3 ]= 124 ;
tab2[ 0x1b07 ]= 125 ;
tab2[ 0x265b ]= 126 ;
tab2[ 0x2ea4 ]= 127 ;
tab2[ 0x0 ]= 128 ;
tab2[ 0x3044 ]= 129 ;
tab2[ 0x15f0 ]= 130 ;
tab2[ 0x185b ]= 131 ;
tab2[ 0x34b6 ]= 132 ;
tab2[ 0x29d2 ]= 133 ;
tab2[ 0x18d2 ]= 134 ;
tab2[ 0x10c2 ]= 135 ;
tab2[ 0x212e ]= 136 ;
tab2[ 0x4bc ]= 137 ;
tab2[ 0x33f1 ]= 138 ;
tab2[ 0x55f ]= 139 ;
tab2[ 0x1421 ]= 140 ;
tab2[ 0x2176 ]= 141 ;
tab2[ 0x574 ]= 142 ;
tab2[ 0x236f ]= 143 ;
tab2[ 0x10d2 ]= 144 ;
tab2[ 0x1152 ]= 145 ;
tab2[ 0x39e0 ]= 146 ;
tab2[ 0x2cad ]= 147 ;
tab2[ 0x15c ]= 148 ;
tab2[ 0x2291 ]= 149 ;
tab2[ 0x26e2 ]= 150 ;
tab2[ 0x218f ]= 151 ;
tab2[ 0x54e ]= 152 ;
tab2[ 0x1231 ]= 153 ;
tab2[ 0x2a93 ]= 154 ;
tab2[ 0x7e6 ]= 155 ;
tab2[ 0x193c ]= 156 ;
tab2[ 0x2222 ]= 157 ;
tab2[ 0x2919 ]= 158 ;
tab2[ 0x259e ]= 159 ;
tab2[ 0x3244 ]= 160 ;
tab2[ 0x1e5 ]= 161 ;
tab2[ 0x2ea6 ]= 162 ;
tab2[ 0x234b ]= 163 ;
tab2[ 0x1bfc ]= 164 ;
tab2[ 0x678 ]= 165 ;
tab2[ 0x301a ]= 166 ;
tab2[ 0x13cb ]= 167 ;
tab2[ 0x289f ]= 168 ;
tab2[ 0x2258 ]= 169 ;
tab2[ 0x2bd5 ]= 170 ;
tab2[ 0x1e78 ]= 171 ;
tab2[ 0x2ddf ]= 172 ;
tab2[ 0x38e ]= 173 ;
tab2[ 0x47c ]= 174 ;
tab2[ 0x2c39 ]= 175 ;
tab2[ 0x651 ]= 176 ;
tab2[ 0x950 ]= 177 ;
tab2[ 0x1daa ]= 178 ;
tab2[ 0x23f5 ]= 179 ;
tab2[ 0xca ]= 180 ;
tab2[ 0x1c18 ]= 181 ;
tab2[ 0x295d ]= 182 ;
tab2[ 0x2bea ]= 183 ;
tab2[ 0x1c85 ]= 184 ;
tab2[ 0xa30 ]= 185 ;
tab2[ 0x3fa1 ]= 186 ;
tab2[ 0x2ec8 ]= 187 ;
tab2[ 0xfff ]= 188 ;
tab2[ 0x49e ]= 189 ;
tab2[ 0x37 ]= 190 ;
tab2[ 0x11e5 ]= 191 ;
tab2[ 0x1e8 ]= 192 ;
tab2[ 0x38c3 ]= 193 ;
tab2[ 0x1b09 ]= 194 ;
tab2[ 0x208c ]= 195 ;
tab2[ 0x3076 ]= 196 ;
tab2[ 0x544 ]= 197 ;
tab2[ 0x3a3 ]= 198 ;
tab2[ 0x1092 ]= 199 ;
tab2[ 0xba3 ]= 200 ;
tab2[ 0x3fb0 ]= 201 ;
tab2[ 0x3390 ]= 202 ;
tab2[ 0x1e04 ]= 203 ;
tab2[ 0x397a ]= 204 ;
tab2[ 0x2148 ]= 205 ;
tab2[ 0xce6 ]= 206 ;
tab2[ 0x3fc1 ]= 207 ;
tab2[ 0x1749 ]= 208 ;
tab2[ 0x328d ]= 209 ;
tab2[ 0x2a06 ]= 210 ;
tab2[ 0x3b62 ]= 211 ;
tab2[ 0x199e ]= 212 ;
tab2[ 0x2f35 ]= 213 ;
tab2[ 0x20e4 ]= 214 ;
tab2[ 0x2294 ]= 215 ;
tab2[ 0x1ef7 ]= 216 ;
tab2[ 0x15f3 ]= 217 ;
tab2[ 0x3e53 ]= 218 ;
tab2[ 0x2d84 ]= 219 ;
tab2[ 0x3c7e ]= 220 ;
tab2[ 0x1678 ]= 221 ;
tab2[ 0x25d8 ]= 222 ;
tab2[ 0x2840 ]= 223 ;
tab2[ 0xef8 ]= 224 ;
tab2[ 0x3bab ]= 225 ;
tab2[ 0x2007 ]= 226 ;
tab2[ 0x1851 ]= 227 ;
tab2[ 0x2198 ]= 228 ;
tab2[ 0x3352 ]= 229 ;
tab2[ 0x2049 ]= 230 ;
tab2[ 0x3fe8 ]= 231 ;
tab2[ 0x316b ]= 232 ;
tab2[ 0x3272 ]= 233 ;
tab2[ 0x158d ]= 234 ;
tab2[ 0x275f ]= 235 ;
tab2[ 0x89c ]= 236 ;
tab2[ 0x47b ]= 237 ;
tab2[ 0x2aa8 ]= 238 ;
tab2[ 0x2d4c ]= 239 ;
tab2[ 0x1355 ]= 240 ;
tab2[ 0x3d69 ]= 241 ;
tab2[ 0x113b ]= 242 ;
tab2[ 0x283f ]= 243 ;
tab2[ 0x258f ]= 244 ;
tab2[ 0x24f ]= 245 ;
tab2[ 0xeb2 ]= 246 ;
tab2[ 0xdd4 ]= 247 ;
tab2[ 0x3d00 ]= 248 ;
tab2[ 0x378e ]= 249 ;
tab2[ 0x188b ]= 250 ;
tab2[ 0x2380 ]= 251 ;
tab2[ 0x172e ]= 252 ;
tab2[ 0x24fa ]= 253 ;
tab2[ 0x19a6 ]= 254 ;
tab2[ 0x115d ]= 255 ;

    
    
    
    
    
    
   
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

    bn_read_str(tmp,"b",1,16);
    fp_prime_conv(z,tmp);
    
    bn_read_str(m,"ffffffffffffffffffffffffffffffff",32,16);
    fp_exp(g,z,m);
    
    bn_read_str(ee,"7fffffffffffffffffffffffffffffff",32,16);
    
    //h=g^(2^(n-w))=g^(2^(94))
    bn_t a1;
    bn_new(a1);
    bn_null(a1);
    bn_read_str(a1,"400000000000000000000000",24,16);
    fp_exp(h,g,a1);
    //h1=g^(2^(n-rem))=g^(2^(196))
    bn_read_str(a1,"1000000000000000000000000",25,16);
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

