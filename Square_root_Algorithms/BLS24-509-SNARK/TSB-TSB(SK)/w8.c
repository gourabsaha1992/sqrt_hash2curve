#include <stdio.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include<immintrin.h>

#include <relic/relic_core.h>
#include "relic/relic_md.h"

#define w 8
#define we 256
#define n 37
#define k1 5
#define l 4
#define rem 5
#define rem_e 32
#define rem_and 0x1f
int i,j,ii,jj,tt;
int sqr=0;
uint64_t mask = (1ULL << (w-rem + 1)) - 1;
int d1[k1];
int e1[k1]={0};


void precomputation(fp_t g,fp_t h,fp_t h1,fp_t tab1[k1][we],fp_t tab3[k1][we]){
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
                bn_lsh(temp,one,n-i*w);
                bn_mul_dig(temp,temp,j);
                fp_exp(tab1[i][j],g,temp);
                fp_inv(tab1[i][j],tab1[i][j]);
            }
    }
   
    

    for(int i=0;i<k1;i++){
            for(int j=0;j<we;j++){
                bn_lsh(temp,one,i*w);
                bn_mul_dig(temp,temp,j);
                fp_exp(tab3[i][j],g,temp);
                fp_inv(tab3[i][j],tab3[i][j]);
            }
    }

}


void findsqroot(fp_t u, fp_t y,fp_t tab1[k1][we],int tab2[32768], fp_t tab3[k1][we],bn_t ee,bn_t tmp) {
	fp_t v,v1,v2,x[k1],temp;
        
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

                
		fp_exp(v, u, ee);
		
		fp_mul(v2,u,v);
		
		fp_mul(x[0],v,v2);
		
		for(int i=1; i<l;i++){
		    fp_copy(x[i],x[i-1]);
		    for(int ii=0; ii<w ; ii++){
		        fp_sqr(x[i],x[i]);
		        sqr++;
		    }
		}
		fp_copy(x[l],x[l-1]);
	        for(int ii=0; ii<rem ; ii++){
	            fp_sqr(x[l],x[l]);
	            sqr++;
	        }
	        
	        
		
		fp_copy(temp,x[l]);
		fp_prime_back(tmp, temp);
	        dig_t d;
	        bn_get_dig(&d, tmp);
	        d=d&0x7fff;
	        e1[0]=tab2[d];
	        //printf("\n\n\n%d\n",e1[0]);
		for(int k=1; k<=l;k++){
		    fp_copy(temp,x[l-k]);
		    fp_mul(temp,temp,tab3[l-k][e1[0]]);
		    for(int j=2;j<=k;j++){
		        fp_mul(temp,temp,tab1[j][e1[k-j+1]]);
		    }
		    fp_prime_back(tmp, temp);
		    dig_t d;
		    bn_get_dig(&d, tmp);
		    d=d&0x7fff;
		    e1[k]=tab2[d];
		    //printf("\n\n\n%d\n",e1[k]);
		}
		
                d1[0] = e1[0] >> 1;
                uint64_t carry = (e1[1]>>(w-rem)) & 1;
                d1[0] |= carry << (w - 1);

                for (int i = 1; i <=l ; i++) {
                    d1[i] = e1[i] >> (w - rem +1);

                    if (i < l) {
                        carry = e1[i + 1] & 0xf;
                        d1[i] |= carry << (rem - 1);
                    }
                    //printf("\n\nd[%d]=%d\n",i,d1[i]);
                }
	        fp_copy(temp,tab3[0][d1[0]]);
	        for(int i=1;i<k1;i++){
	            fp_mul(temp,temp,tab3[i][d1[i]]);
	        }
	        fp_mul(y,v2,temp);
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

    if (ep_param_set_any_pairf() != RLC_OK) {
        printf("Error setting BLS12 curve parameters.\n");
        core_clean();
        return 1;
    }

    printf("Using Primes of bls12-377snark\n");

    fp_t tab1[k1][we],tab3[k1][we],g,z,b,h,h1,y,t3;
    int tab2[32768]={0};
    bn_t ee,m,tmp;
    dig_t q[4][4],qq[3];
    for(int i=0;i<k1;i++){
        for(int j=0;j<we;j++){
            fp_null(tab1[i][j]);
            fp_new(tab1[i][j]);
        }
    }
    for(int i=0;i<k1;i++){
        for(int j=0;j<we;j++){
            fp_null(tab3[i][j]);
            fp_new(tab3[i][j]);
        }
    }

    tab2[ 1 ]= 0 ;
    tab2[ 9178 ]= 1 ;
    tab2[ 4107 ]= 2 ;
    tab2[ 5773 ]= 3 ;
    tab2[ 678 ]= 4 ;
    tab2[ 18051 ]= 5 ;
    tab2[ 15840 ]= 6 ;
    tab2[ 28757 ]= 7 ;
    tab2[ 6116 ]= 8 ;
    tab2[ 21652 ]= 9 ;
    tab2[ 17877 ]= 10 ;
    tab2[ 1358 ]= 11 ;
    tab2[ 4205 ]= 12 ;
    tab2[ 1482 ]= 13 ;
    tab2[ 30722 ]= 14 ;
    tab2[ 20569 ]= 15 ;
    tab2[ 30917 ]= 16 ;
    tab2[ 2999 ]= 17 ;
    tab2[ 20255 ]= 18 ;
    tab2[ 23559 ]= 19 ;
    tab2[ 13284 ]= 20 ;
    tab2[ 20241 ]= 21 ;
    tab2[ 4959 ]= 22 ;
    tab2[ 28183 ]= 23 ;
    tab2[ 31780 ]= 24 ;
    tab2[ 23374 ]= 25 ;
    tab2[ 28843 ]= 26 ;
    tab2[ 16324 ]= 27 ;
    tab2[ 5156 ]= 28 ;
    tab2[ 10533 ]= 29 ;
    tab2[ 19322 ]= 30 ;
    tab2[ 7170 ]= 31 ;
    tab2[ 29725 ]= 32 ;
    tab2[ 29403 ]= 33 ;
    tab2[ 13593 ]= 34 ;
    tab2[ 30124 ]= 35 ;
    tab2[ 16781 ]= 36 ;
    tab2[ 3712 ]= 37 ;
    tab2[ 6603 ]= 38 ;
    tab2[ 13739 ]= 39 ;
    tab2[ 22153 ]= 40 ;
    tab2[ 16184 ]= 41 ;
    tab2[ 30142 ]= 42 ;
    tab2[ 6008 ]= 43 ;
    tab2[ 21995 ]= 44 ;
    tab2[ 1470 ]= 45 ;
    tab2[ 18527 ]= 46 ;
    tab2[ 12703 ]= 47 ;
    tab2[ 13090 ]= 48 ;
    tab2[ 28832 ]= 49 ;
    tab2[ 30614 ]= 50 ;
    tab2[ 13814 ]= 51 ;
    tab2[ 16081 ]= 52 ;
    tab2[ 12608 ]= 53 ;
    tab2[ 10903 ]= 54 ;
    tab2[ 28103 ]= 55 ;
    tab2[ 18039 ]= 56 ;
    tab2[ 6226 ]= 57 ;
    tab2[ 2106 ]= 58 ;
    tab2[ 2036 ]= 59 ;
    tab2[ 16048 ]= 60 ;
    tab2[ 28480 ]= 61 ;
    tab2[ 10437 ]= 62 ;
    tab2[ 26500 ]= 63 ;
    tab2[ 22755 ]= 64 ;
    tab2[ 4126 ]= 65 ;
    tab2[ 20174 ]= 66 ;
    tab2[ 14304 ]= 67 ;
    tab2[ 24322 ]= 68 ;
    tab2[ 26876 ]= 69 ;
    tab2[ 23334 ]= 70 ;
    tab2[ 2200 ]= 71 ;
    tab2[ 23176 ]= 72 ;
    tab2[ 14938 ]= 73 ;
    tab2[ 1016 ]= 74 ;
    tab2[ 31214 ]= 75 ;
    tab2[ 12231 ]= 76 ;
    tab2[ 9246 ]= 77 ;
    tab2[ 8693 ]= 78 ;
    tab2[ 23308 ]= 79 ;
    tab2[ 32323 ]= 80 ;
    tab2[ 31618 ]= 81 ;
    tab2[ 2140 ]= 82 ;
    tab2[ 28834 ]= 83 ;
    tab2[ 10819 ]= 84 ;
    tab2[ 21234 ]= 85 ;
    tab2[ 29006 ]= 86 ;
    tab2[ 24018 ]= 87 ;
    tab2[ 14427 ]= 88 ;
    tab2[ 2154 ]= 89 ;
    tab2[ 18153 ]= 90 ;
    tab2[ 31647 ]= 91 ;
    tab2[ 28442 ]= 92 ;
    tab2[ 32272 ]= 93 ;
    tab2[ 15938 ]= 94 ;
    tab2[ 7340 ]= 95 ;
    tab2[ 5612 ]= 96 ;
    tab2[ 30049 ]= 97 ;
    tab2[ 9474 ]= 98 ;
    tab2[ 9836 ]= 99 ;
    tab2[ 13490 ]= 100 ;
    tab2[ 8235 ]= 101 ;
    tab2[ 23629 ]= 102 ;
    tab2[ 13981 ]= 103 ;
    tab2[ 9158 ]= 104 ;
    tab2[ 193 ]= 105 ;
    tab2[ 31690 ]= 106 ;
    tab2[ 10185 ]= 107 ;
    tab2[ 19909 ]= 108 ;
    tab2[ 17361 ]= 109 ;
    tab2[ 2586 ]= 110 ;
    tab2[ 21630 ]= 111 ;
    tab2[ 11702 ]= 112 ;
    tab2[ 31028 ]= 113 ;
    tab2[ 1065 ]= 114 ;
    tab2[ 9571 ]= 115 ;
    tab2[ 18448 ]= 116 ;
    tab2[ 11360 ]= 117 ;
    tab2[ 14861 ]= 118 ;
    tab2[ 8319 ]= 119 ;
    tab2[ 5913 ]= 120 ;
    tab2[ 18567 ]= 121 ;
    tab2[ 14925 ]= 122 ;
    tab2[ 4018 ]= 123 ;
    tab2[ 11106 ]= 124 ;
    tab2[ 31240 ]= 125 ;
    tab2[ 17057 ]= 126 ;
    tab2[ 1769 ]= 127 ;
    tab2[ 0 ]= 128 ;
    tab2[ 23591 ]= 129 ;
    tab2[ 28662 ]= 130 ;
    tab2[ 26996 ]= 131 ;
    tab2[ 32091 ]= 132 ;
    tab2[ 14718 ]= 133 ;
    tab2[ 16929 ]= 134 ;
    tab2[ 4012 ]= 135 ;
    tab2[ 26653 ]= 136 ;
    tab2[ 11117 ]= 137 ;
    tab2[ 14892 ]= 138 ;
    tab2[ 31411 ]= 139 ;
    tab2[ 28564 ]= 140 ;
    tab2[ 31287 ]= 141 ;
    tab2[ 2047 ]= 142 ;
    tab2[ 12200 ]= 143 ;
    tab2[ 1852 ]= 144 ;
    tab2[ 29770 ]= 145 ;
    tab2[ 12514 ]= 146 ;
    tab2[ 9210 ]= 147 ;
    tab2[ 19485 ]= 148 ;
    tab2[ 12528 ]= 149 ;
    tab2[ 27810 ]= 150 ;
    tab2[ 4586 ]= 151 ;
    tab2[ 989 ]= 152 ;
    tab2[ 9395 ]= 153 ;
    tab2[ 3926 ]= 154 ;
    tab2[ 16445 ]= 155 ;
    tab2[ 27613 ]= 156 ;
    tab2[ 22236 ]= 157 ;
    tab2[ 13447 ]= 158 ;
    tab2[ 25599 ]= 159 ;
    tab2[ 3044 ]= 160 ;
    tab2[ 3366 ]= 161 ;
    tab2[ 19176 ]= 162 ;
    tab2[ 2645 ]= 163 ;
    tab2[ 15988 ]= 164 ;
    tab2[ 29057 ]= 165 ;
    tab2[ 26166 ]= 166 ;
    tab2[ 19030 ]= 167 ;
    tab2[ 10616 ]= 168 ;
    tab2[ 16585 ]= 169 ;
    tab2[ 2627 ]= 170 ;
    tab2[ 26761 ]= 171 ;
    tab2[ 10774 ]= 172 ;
    tab2[ 31299 ]= 173 ;
    tab2[ 14242 ]= 174 ;
    tab2[ 20066 ]= 175 ;
    tab2[ 19679 ]= 176 ;
    tab2[ 3937 ]= 177 ;
    tab2[ 2155 ]= 178 ;
    tab2[ 18955 ]= 179 ;
    tab2[ 16688 ]= 180 ;
    tab2[ 20161 ]= 181 ;
    tab2[ 21866 ]= 182 ;
    tab2[ 4666 ]= 183 ;
    tab2[ 14730 ]= 184 ;
    tab2[ 26543 ]= 185 ;
    tab2[ 30663 ]= 186 ;
    tab2[ 30733 ]= 187 ;
    tab2[ 16721 ]= 188 ;
    tab2[ 4289 ]= 189 ;
    tab2[ 22332 ]= 190 ;
    tab2[ 6269 ]= 191 ;
    tab2[ 10014 ]= 192 ;
    tab2[ 28643 ]= 193 ;
    tab2[ 12595 ]= 194 ;
    tab2[ 18465 ]= 195 ;
    tab2[ 8447 ]= 196 ;
    tab2[ 5893 ]= 197 ;
    tab2[ 9435 ]= 198 ;
    tab2[ 30569 ]= 199 ;
    tab2[ 9593 ]= 200 ;
    tab2[ 17831 ]= 201 ;
    tab2[ 31753 ]= 202 ;
    tab2[ 1555 ]= 203 ;
    tab2[ 20538 ]= 204 ;
    tab2[ 23523 ]= 205 ;
    tab2[ 24076 ]= 206 ;
    tab2[ 9461 ]= 207 ;
    tab2[ 446 ]= 208 ;
    tab2[ 1151 ]= 209 ;
    tab2[ 30629 ]= 210 ;
    tab2[ 3935 ]= 211 ;
    tab2[ 21950 ]= 212 ;
    tab2[ 11535 ]= 213 ;
    tab2[ 3763 ]= 214 ;
    tab2[ 8751 ]= 215 ;
    tab2[ 18342 ]= 216 ;
    tab2[ 30615 ]= 217 ;
    tab2[ 14616 ]= 218 ;
    tab2[ 1122 ]= 219 ;
    tab2[ 4327 ]= 220 ;
    tab2[ 497 ]= 221 ;
    tab2[ 16831 ]= 222 ;
    tab2[ 25429 ]= 223 ;
    tab2[ 27157 ]= 224 ;
    tab2[ 2720 ]= 225 ;
    tab2[ 23295 ]= 226 ;
    tab2[ 22933 ]= 227 ;
    tab2[ 19279 ]= 228 ;
    tab2[ 24534 ]= 229 ;
    tab2[ 9140 ]= 230 ;
    tab2[ 18788 ]= 231 ;
    tab2[ 23611 ]= 232 ;
    tab2[ 32576 ]= 233 ;
    tab2[ 1079 ]= 234 ;
    tab2[ 22584 ]= 235 ;
    tab2[ 12860 ]= 236 ;
    tab2[ 15408 ]= 237 ;
    tab2[ 30183 ]= 238 ;
    tab2[ 11139 ]= 239 ;
    tab2[ 21067 ]= 240 ;
    tab2[ 1741 ]= 241 ;
    tab2[ 31704 ]= 242 ;
    tab2[ 23198 ]= 243 ;
    tab2[ 14321 ]= 244 ;
    tab2[ 21409 ]= 245 ;
    tab2[ 17908 ]= 246 ;
    tab2[ 24450 ]= 247 ;
    tab2[ 26856 ]= 248 ;
    tab2[ 14202 ]= 249 ;
    tab2[ 17844 ]= 250 ;
    tab2[ 28751 ]= 251 ;
    tab2[ 21663 ]= 252 ;
    tab2[ 1529 ]= 253 ;
    tab2[ 15712 ]= 254 ;
    tab2[ 31000 ]= 255 ;


    
    
   
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



    bn_read_str(tmp,"b",1,16);
    fp_prime_conv(z,tmp);
    
    bn_read_str(m,"ba29500be5eeb41528150f09d4eb38e93cac77669619e1d1b6156f110ecdcab5f527a4d95af73ebeb95690032ee595c74e5000ada8a9600000408f",128,16);
    fp_exp(g,z,m);
    
    bn_read_str(ee,"5d14a805f2f75a0a940a8784ea759c749e563bb34b0cf0e8db0ab7888766e55afa93d26cad7b9f5f5cab48019772cae3a7280056d454b000002047",128,16);
    
    //h=g^(2^(n-w))=g^(2^(33))
    bn_t a1;
    bn_new(a1);
    bn_null(a1);
    bn_read_str(a1,"200000000",9,16);
    fp_exp(h,g,a1);
    //h1=g^(2^(n-rem))=g^(2^(36))
    bn_read_str(a1,"1000000000",10,16);
    fp_exp(h1,g,a1);
    
    
    precomputation(g,h,h1,tab1,tab3);

    printf("using tonelli-shank\n");
    
    fp_copy(t3, fp_prime_get_srt());
    
    //MEASURE(squareroot(y, b, m, t3);)
    
    printf("RDTSC_clk_min=%f\n",RDTSC_clk_min);
    printf("RDTSC_clk_median=%f\n",RDTSC_clk_median);
    printf("RDTSC_clk_max=%f\n",RDTSC_clk_max);

    printf("using tonelli-shank look up table by bernstein\n");

    MEASURE(findsqroot(b,y,tab1,tab2,tab3,ee,tmp);)
    printf("RDTSC_clk_min=%f\n",RDTSC_clk_min);
    printf("RDTSC_clk_median=%f\n",RDTSC_clk_median);
    printf("RDTSC_clk_max=%f\n",RDTSC_clk_max);
    
    
    for(int i=0;i<k1;i++){
        for(int j=0;j<we;j++){
            fp_free(tab1[i][j]);
        }
    }

    
    
    for(int i=0;i<k1;i++){
        for(int j=0;j<we;j++){
            fp_free(tab3[i][j]);
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

