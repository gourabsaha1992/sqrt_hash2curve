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
dig_t d;
int i,j,ii,jj,tt,k;
fp_t tab1[k1][we],tab3[k1][we],g,z,b,h;
bn_t ee,m,tmp;
int tab2[32768]={0};


int sqr=0;
uint64_t mask = (1ULL << (w - rem + 1)) - 1;
int d1[k1];
int e1[k1]={0};


void precomputation(fp_t g,fp_t h){
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


void findsqroot(fp_t u, fp_t y) {
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
		    }
		}
		fp_copy(x[l],x[l-1]);
	        for(int ii=0; ii<rem ; ii++){
	            fp_sqr(x[l],x[l]);
	        }
	        
	        
		
		fp_copy(temp,x[l]);
		fp_prime_back(tmp, temp);
	        bn_get_dig(&d, tmp);
	        d=d&0x7fff;
	        e1[0]=tab2[d]& rem_and;
	        //printf("\n\n%d\n\n",e1[0]);
	        
	        
		for(int k=1; k<=l;k++){
		    fp_copy(temp,x[l-k]);
		    fp_mul(temp,temp,tab3[l-k][e1[0]]);
		    for(int j=2;j<=k;j++){
		        fp_mul(temp,temp,tab1[j][e1[k-j+1]]);
		    }
		    fp_prime_back(tmp, temp);
		    bn_get_dig(&d, tmp);
		    d=d&0x7fff;
		    e1[k]=tab2[d];
		    //printf("\n\n%d\n\n",e1[k]);
		    
		}
		
                d1[0] = e1[0] >> 1;

                uint64_t carry = e1[1] & mask;
                d1[0] |= carry << (rem - 1);

                for (int i = 1; i <=l ; i++) {
                    d1[i] = e1[i] >> (w - rem + 1);

                    if (i < l) {
                        carry = e1[i + 1] & mask;
                        d1[i] |= carry << (rem - 1);
                    }
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



void ff_impl(ep_t p, const uint8_t *random, size_t len, fp_t b, fp_t c, fp_t D, fp_t A, fp_t B, fp_t alpha) {
	fp_t t1, t2, y1, y2, y3,tmp1,tmp2,x1,x2,x3,z1,x,y;
	ctx_t *ctx = core_get();
	bn_t kk;
	ep_t q;
        ep_null(q);
        
	uint8_t s;
	int c1, c2, c3;
	bn_null(kk);	
	fp_null(t1);
	fp_null(t2);
	fp_null(tmp1);
	fp_null(tmp2);
	fp_null(y1);
	fp_null(y2);
	fp_null(y3);
	fp_null(x1);
	fp_null(x2);
	fp_null(x3);
	fp_null(z1);
	fp_null(x);
	fp_null(y);

	RLC_TRY {
		bn_new(kk);
		fp_new(t1);
		fp_new(t2);
		fp_new(tmp1);
		fp_new(tmp2);
		fp_new(y1);
		fp_new(y2);
		fp_new(y3);
		fp_new(x1);
		fp_new(x2);
		fp_new(x3);
		fp_new(z1);
		fp_new(x);
		fp_new(y);
		ep_new(q);

		bn_read_bin(kk, random, len / 2);
		fp_prime_conv(t1, kk);
		bn_read_bin(kk, random + len / 2, len / 2);
		fp_prime_conv(t2, kk);
		
	        c1 = fp_is_zero(t1);
	        if (c1) {
		    ep_set_infty(p);
		} else {
		    fp_sqr(tmp1, t1);
		    fp_mul_dig(tmp1,tmp1,5);
		    fp_add_dig(tmp2,tmp1,1);
		    fp_inv(tmp2,tmp2);
		    fp_mul(tmp2,tmp2,A);
		    fp_neg(x,tmp2);  
		    fp_mul(x2,x,tmp1);

		    
		    fp_sqr(tmp1,x);   
		    fp_mul(y,tmp1,x); 
		    fp_mul(tmp2,tmp1,A);
		    fp_add(y,y,tmp2);
		    fp_mul(tmp2,x,B);
		    fp_add(y,y,tmp2);
		    
		    c1=fp_is_sqr(y);
                
                    fp_mul_dig(x,x,c1);
                    c1=1-c1;
                    fp_mul_dig(x2,x2,c1);
                    fp_add(x,x,x2);
                    
                    fp_sqr(tmp1,x);   
		    fp_mul(y,tmp1,x); 
		    fp_mul(tmp2,tmp1,A);
		    fp_add(y,y,tmp2);
		    fp_mul(tmp2,x,B);
		    fp_add(y,y,tmp2);

                    findsqroot(y,y);
                    fp_add(x,x,alpha);

		    //fp_sqr(tmp1,x);
		    //fp_mul(tmp1,x,tmp1);
		    //fp_sqr(tmp2,y);
		    //fp_add_dig(tmp1,tmp1,1);
		    //fp_sub(tmp1,tmp1,tmp2);
		    //fp_print(tmp1);

		    fp_copy(p->x,x);
		    fp_copy(p->y,y);
		    fp_set_dig(p->z,1);
		}
		
		c1 = fp_is_zero(t2);
		if (c1) {
		    ep_set_infty(q);
		} else {
		    fp_sqr(tmp1, t2);
		    fp_mul_dig(tmp1,tmp1,5);
		    fp_add_dig(tmp2,tmp1,1);
		    fp_inv(tmp2,tmp2);
		    fp_mul(tmp2,tmp2,A);
		    fp_neg(x,tmp2);  
		    fp_mul(x2,x,tmp1);

		    
		    fp_sqr(tmp1,x);   
		    fp_mul(y,tmp1,x); 
		    fp_mul(tmp2,tmp1,A);
		    fp_add(y,y,tmp2);
		    fp_mul(tmp2,x,B);
		    fp_add(y,y,tmp2);
		    
		    c1=fp_is_sqr(y);
                
                    fp_mul_dig(x,x,c1);
                    c1=1-c1;
                    fp_mul_dig(x2,x2,c1);
                    fp_add(x,x,x2);
                    
                    fp_sqr(tmp1,x);   
		    fp_mul(y,tmp1,x); 
		    fp_mul(tmp2,tmp1,A);
		    fp_add(y,y,tmp2);
		    fp_mul(tmp2,x,B);
		    fp_add(y,y,tmp2);

                    findsqroot(y,y);
                    fp_add(x,x,alpha);                    

		    //fp_sqr(tmp1,x);
		    //fp_mul(tmp1,x,tmp1);
		    //fp_sqr(tmp2,y);
		    //fp_add_dig(tmp1,tmp1,1);
		    //fp_sub(tmp1,tmp1,tmp2);
		    //fp_print(tmp1);

		    fp_copy(q->x,x);
		    fp_copy(q->y,y);
		    fp_set_dig(q->z,1);
		}    
		ep_add(p,p,q);	
		ep_mul_cof(p, p);
	}

	RLC_CATCH_ANY {
		RLC_THROW(ERR_CAUGHT);
	}
	RLC_FINALLY {
		bn_free(kk);
		fp_free(t1);
		fp_free(t2);
		fp_free(tmp1);
		fp_free(tmp2);
		fp_free(y1);
		fp_free(y2);
		fp_free(y3);
		fp_free(x1);
		fp_free(x2);
		fp_free(x3);
		fp_free(z1);
		fp_free(x);
		fp_free(y);
		ep_free(q);
	}

}

void ep_map_ff(ep_t p, const uint8_t *msg, size_t len,  fp_t b, fp_t c, fp_t D, fp_t A, fp_t B, fp_t alpha) {
	/* enough space for two field elements plus extra bytes for uniformity */
	const size_t elm = (FP_PRIME + ep_param_level() + 7) / 8;
	uint8_t *r = RLC_ALLOCA(uint8_t, 2 * elm + 1);
	ctx_t *ctx = core_get();

	if (ep_curve_is_super()) {
		RLC_FREE(r); 
		RLC_THROW(ERR_NO_CONFIG);
		return;
	}

	if (ctx->mod18 % 3 == 2) {
		RLC_FREE(r); 
		RLC_THROW(ERR_NO_CONFIG);
		return;
	}

	RLC_TRY {
		static const uint8_t DST[] = "ELLIGATOR-BLS24-509-G1";

		md_xmd(r, 2*elm + 1, msg, len,
			   DST, sizeof(DST) - 1);


		ff_impl(p, r, 2 * elm + 1, b, c, D, A, B, alpha);
	}
	RLC_CATCH_ANY {
		RLC_THROW(ERR_CAUGHT);
	}
	RLC_FINALLY {
		RLC_FREE(r);
	}
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
        printf("Error setting BLS24 curve parameters.\n");
        core_clean();
        return 1;
    }

    printf("Using Primes of bls24-509snark\n");

    
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
    
    
    
    bn_read_str(tmp,"b",1,16);
    fp_prime_conv(z,tmp);
    
    bn_read_str(m,"ba29500be5eeb41528150f09d4eb38e93cac77669619e1d1b6156f110ecdcab5f527a4d95af73ebeb95690032ee595c74e5000ada8a9600000408f",128,16);
    fp_exp(g,z,m);
    
    bn_read_str(ee,"5d14a805f2f75a0a940a8784ea759c749e563bb34b0cf0e8db0ab7888766e55afa93d26cad7b9f5f5cab48019772cae3a7280056d454b000002047",128,16);
    
    //h=g^(2^(n-w))=g^(2^(31))
    bn_t a1;
    bn_new(a1);
    bn_null(a1);
    bn_read_str(a1,"80000000",9,16);
    fp_exp(h,g,a1);
    
    
    precomputation(g,h);

    
    

    
	
        ep_t P;
        ep_null(P);
        ep_new(P);
        
        fp_t A, B, D, c, alpha, point[6];
    bn_t tmp;   
    
    bn_null(tmp);
    bn_new(tmp);
    
    fp_null(alpha); 
    fp_null(b);
    fp_null(A);
    fp_null(B);
    fp_null(D);
    fp_null(c);
    
    fp_new(alpha); 
    fp_new(b);
    fp_new(A);
    fp_new(B);
    fp_new(D);
    fp_new(c);
    
    for(int i=0;i<6;i++){
        fp_new(point[i]);
        fp_null(point[i]);
    }


    // Get curve parameter b and lambda
        fp_set_dig(b, 1);
        fp_set_dig(c, 5);
        bn_read_str(tmp,"17452A017CBDD682A502A1E13A9D671D27958EECD2C33C3A36C2ADE221D9B956BEA4F49B2B5EE7D7D72AD20065DCB2B8E9CA0015B5152C00000811E000000000",128,16);
	fp_prime_conv(alpha, tmp);
        //fp_print(alpha);
	fp_neg(D,alpha);
	fp_mul_dig(A,alpha,3);
	fp_mul(B,A,alpha);
	
	
	
    
	const char *msg = "Hashing to BLS24-509 and this is the msg";
        size_t msg_len = strlen(msg);
    
	
   	MEASURE(ep_map_ff(P, (const uint8_t *)msg, msg_len, b, c, D, A, B, alpha);)
	printf("RDTSC_clk_min=%f\n",RDTSC_clk_min);
	printf("RDTSC_clk_median=%f\n",RDTSC_clk_median);
	printf("RDTSC_clk_max=%f\n",RDTSC_clk_max);
    
    
    fp_free(alpha); 
    fp_free(b);
    fp_free(A);
    fp_free(B);
    fp_free(D);
    fp_free(c);
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
    bn_free(a1);
    
    core_clean();
    return 0;
}

