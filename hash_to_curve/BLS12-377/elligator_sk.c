


#include <stdio.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include<immintrin.h>

#include <relic/relic_core.h>
#include "relic/relic_md.h"

#define w 8
#define we 256
#define n 46
#define k1 6
#define l 5
#define rem 6
#define rem_e 64
#define rem_and 63
fp_t tab1[k1][we],tab3[k1][we],g,z,b,h;
bn_t ee,tmp,m;
int i,j,ii,jj,tt;
int sqr=0;
uint64_t mask = (1ULL << (w-rem + 1)) - 1;
int d1[k1];
int e1[k1]={0};
int tab2[65536]={0};
dig_t d;



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
	        bn_get_dig(&d, tmp);
	        d=d&0xffff;
	        e1[0]=tab2[d];
		for(int k=1; k<=l;k++){
		    fp_copy(temp,x[l-k]);
		    fp_mul(temp,temp,tab3[l-k][e1[0]]);
		    for(int j=2;j<=k;j++){
		        fp_mul(temp,temp,tab1[j][e1[k-j+1]]);
		    }
		    fp_prime_back(tmp, temp);
		    bn_get_dig(&d, tmp);
		    d=d&0xffff;
		    e1[k]=tab2[d];
		}
		
                d1[0] = e1[0] >> 1;

                uint64_t carry = (e1[1]>>(w-rem)) & 1;
                d1[0] |= carry << (w - 1);
                //printf("\n\nd[0]=%d\n",d1[0]);

                for (int i = 1; i <=l ; i++) {
                    d1[i] = e1[i] >> (w - rem +1);

                    if (i < l) {
                        carry = e1[i + 1] & 0x7;
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




void elligator2_impl(ep_t p, const uint8_t *random, size_t len, fp_t b, fp_t c, fp_t D, fp_t A, fp_t B, fp_t alpha) {
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
		fp_rand(t1);
		fp_rand(t2);
				
	      
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
		fp_free(temp1);
		fp_free(temp2);
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

void ep_map_elligator2(ep_t p, const uint8_t *msg, size_t len,  fp_t b, fp_t c, fp_t D, fp_t A, fp_t B, fp_t alpha) {
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
		static const uint8_t DST[] = "ELLIGATOR-BLS12-377-G1";

		md_xmd(r, 2*elm + 1, msg, len,
			   DST, sizeof(DST) - 1);


		elligator2_impl(p, r, 2 * elm + 1, b, c, D, A, B, alpha);
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
int ii,jj,tt;

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
    tab2[ 0x1 ]= 0 ;
tab2[ 0x74fa ]= 1 ;
tab2[ 0xb34e ]= 2 ;
tab2[ 0xc9bc ]= 3 ;
tab2[ 0xf359 ]= 4 ;
tab2[ 0x371a ]= 5 ;
tab2[ 0x1e32 ]= 6 ;
tab2[ 0x3a37 ]= 7 ;
tab2[ 0xd52c ]= 8 ;
tab2[ 0xacee ]= 9 ;
tab2[ 0x588f ]= 10 ;
tab2[ 0xc525 ]= 11 ;
tab2[ 0xc61f ]= 12 ;
tab2[ 0x72d7 ]= 13 ;
tab2[ 0xac ]= 14 ;
tab2[ 0xf3e3 ]= 15 ;
tab2[ 0xd143 ]= 16 ;
tab2[ 0x5c50 ]= 17 ;
tab2[ 0x41ce ]= 18 ;
tab2[ 0x5e8d ]= 19 ;
tab2[ 0x560e ]= 20 ;
tab2[ 0x3c21 ]= 21 ;
tab2[ 0xd6f8 ]= 22 ;
tab2[ 0x32f1 ]= 23 ;
tab2[ 0x430 ]= 24 ;
tab2[ 0x8ad ]= 25 ;
tab2[ 0xecad ]= 26 ;
tab2[ 0xb0cc ]= 27 ;
tab2[ 0xf9e4 ]= 28 ;
tab2[ 0x9587 ]= 29 ;
tab2[ 0x34bd ]= 30 ;
tab2[ 0x575a ]= 31 ;
tab2[ 0x80f0 ]= 32 ;
tab2[ 0x3e5b ]= 33 ;
tab2[ 0xc50e ]= 34 ;
tab2[ 0xdffe ]= 35 ;
tab2[ 0x3401 ]= 36 ;
tab2[ 0xe21c ]= 37 ;
tab2[ 0x8f03 ]= 38 ;
tab2[ 0x4a3e ]= 39 ;
tab2[ 0x22e6 ]= 40 ;
tab2[ 0xc31d ]= 41 ;
tab2[ 0x8155 ]= 42 ;
tab2[ 0xf56a ]= 43 ;
tab2[ 0x575b ]= 44 ;
tab2[ 0x35b1 ]= 45 ;
tab2[ 0x27ec ]= 46 ;
tab2[ 0x9d47 ]= 47 ;
tab2[ 0x4dca ]= 48 ;
tab2[ 0x8d60 ]= 49 ;
tab2[ 0xeaa2 ]= 50 ;
tab2[ 0x3c29 ]= 51 ;
tab2[ 0xe49a ]= 52 ;
tab2[ 0xf94e ]= 53 ;
tab2[ 0x8664 ]= 54 ;
tab2[ 0x737d ]= 55 ;
tab2[ 0x3f14 ]= 56 ;
tab2[ 0x1045 ]= 57 ;
tab2[ 0xbfea ]= 58 ;
tab2[ 0x3f04 ]= 59 ;
tab2[ 0xe476 ]= 60 ;
tab2[ 0xb579 ]= 61 ;
tab2[ 0x545 ]= 62 ;
tab2[ 0xc885 ]= 63 ;
tab2[ 0xe39e ]= 64 ;
tab2[ 0xafde ]= 65 ;
tab2[ 0xe78a ]= 66 ;
tab2[ 0xca01 ]= 67 ;
tab2[ 0xc0e4 ]= 68 ;
tab2[ 0x6c3c ]= 69 ;
tab2[ 0x9f20 ]= 70 ;
tab2[ 0x20a1 ]= 71 ;
tab2[ 0x247 ]= 72 ;
tab2[ 0x15d0 ]= 73 ;
tab2[ 0x9997 ]= 74 ;
tab2[ 0xc210 ]= 75 ;
tab2[ 0x652f ]= 76 ;
tab2[ 0xa0e5 ]= 77 ;
tab2[ 0x9497 ]= 78 ;
tab2[ 0x1b62 ]= 79 ;
tab2[ 0xea4 ]= 80 ;
tab2[ 0xdce2 ]= 81 ;
tab2[ 0x44e8 ]= 82 ;
tab2[ 0xd9b6 ]= 83 ;
tab2[ 0x871f ]= 84 ;
tab2[ 0x9fad ]= 85 ;
tab2[ 0xa2b6 ]= 86 ;
tab2[ 0x8659 ]= 87 ;
tab2[ 0xeb2 ]= 88 ;
tab2[ 0x5583 ]= 89 ;
tab2[ 0xff25 ]= 90 ;
tab2[ 0x58ea ]= 91 ;
tab2[ 0x9d76 ]= 92 ;
tab2[ 0x5b77 ]= 93 ;
tab2[ 0x6e28 ]= 94 ;
tab2[ 0x275b ]= 95 ;
tab2[ 0x85ba ]= 96 ;
tab2[ 0x8b67 ]= 97 ;
tab2[ 0x2977 ]= 98 ;
tab2[ 0x9b65 ]= 99 ;
tab2[ 0xb575 ]= 100 ;
tab2[ 0x913 ]= 101 ;
tab2[ 0xe93 ]= 102 ;
tab2[ 0x589d ]= 103 ;
tab2[ 0x5e66 ]= 104 ;
tab2[ 0xea44 ]= 105 ;
tab2[ 0x16ce ]= 106 ;
tab2[ 0xdca9 ]= 107 ;
tab2[ 0xb97c ]= 108 ;
tab2[ 0x2e41 ]= 109 ;
tab2[ 0x2f44 ]= 110 ;
tab2[ 0xc67f ]= 111 ;
tab2[ 0x12b1 ]= 112 ;
tab2[ 0x5749 ]= 113 ;
tab2[ 0x12d1 ]= 114 ;
tab2[ 0x6b18 ]= 115 ;
tab2[ 0xbff4 ]= 116 ;
tab2[ 0x1d46 ]= 117 ;
tab2[ 0x32a2 ]= 118 ;
tab2[ 0x44ea ]= 119 ;
tab2[ 0xd1ab ]= 120 ;
tab2[ 0x398f ]= 121 ;
tab2[ 0x4041 ]= 122 ;
tab2[ 0xfa47 ]= 123 ;
tab2[ 0x5e55 ]= 124 ;
tab2[ 0x35e2 ]= 125 ;
tab2[ 0x4fe9 ]= 126 ;
tab2[ 0x610c ]= 127 ;
tab2[ 0x0 ]= 128 ;
tab2[ 0x8b07 ]= 129 ;
tab2[ 0x4cb3 ]= 130 ;
tab2[ 0x3645 ]= 131 ;
tab2[ 0xca8 ]= 132 ;
tab2[ 0xc8e7 ]= 133 ;
tab2[ 0xe1cf ]= 134 ;
tab2[ 0xc5ca ]= 135 ;
tab2[ 0x2ad5 ]= 136 ;
tab2[ 0x5313 ]= 137 ;
tab2[ 0xa772 ]= 138 ;
tab2[ 0x3adc ]= 139 ;
tab2[ 0x39e2 ]= 140 ;
tab2[ 0x8d2a ]= 141 ;
tab2[ 0xff55 ]= 142 ;
tab2[ 0xc1e ]= 143 ;
tab2[ 0x2ebe ]= 144 ;
tab2[ 0xa3b1 ]= 145 ;
tab2[ 0xbe33 ]= 146 ;
tab2[ 0xa174 ]= 147 ;
tab2[ 0xa9f3 ]= 148 ;
tab2[ 0xc3e0 ]= 149 ;
tab2[ 0x2909 ]= 150 ;
tab2[ 0xcd10 ]= 151 ;
tab2[ 0xfbd1 ]= 152 ;
tab2[ 0xf754 ]= 153 ;
tab2[ 0x1354 ]= 154 ;
tab2[ 0x4f35 ]= 155 ;
tab2[ 0x61d ]= 156 ;
tab2[ 0x6a7a ]= 157 ;
tab2[ 0xcb44 ]= 158 ;
tab2[ 0xa8a7 ]= 159 ;
tab2[ 0x7f11 ]= 160 ;
tab2[ 0xc1a6 ]= 161 ;
tab2[ 0x3af3 ]= 162 ;
tab2[ 0x2003 ]= 163 ;
tab2[ 0xcc00 ]= 164 ;
tab2[ 0x1de5 ]= 165 ;
tab2[ 0x70fe ]= 166 ;
tab2[ 0xb5c3 ]= 167 ;
tab2[ 0xdd1b ]= 168 ;
tab2[ 0x3ce4 ]= 169 ;
tab2[ 0x7eac ]= 170 ;
tab2[ 0xa97 ]= 171 ;
tab2[ 0xa8a6 ]= 172 ;
tab2[ 0xca50 ]= 173 ;
tab2[ 0xd815 ]= 174 ;
tab2[ 0x62ba ]= 175 ;
tab2[ 0xb237 ]= 176 ;
tab2[ 0x72a1 ]= 177 ;
tab2[ 0x155f ]= 178 ;
tab2[ 0xc3d8 ]= 179 ;
tab2[ 0x1b67 ]= 180 ;
tab2[ 0x6b3 ]= 181 ;
tab2[ 0x799d ]= 182 ;
tab2[ 0x8c84 ]= 183 ;
tab2[ 0xc0ed ]= 184 ;
tab2[ 0xefbc ]= 185 ;
tab2[ 0x4017 ]= 186 ;
tab2[ 0xc0fd ]= 187 ;
tab2[ 0x1b8b ]= 188 ;
tab2[ 0x4a88 ]= 189 ;
tab2[ 0xfabc ]= 190 ;
tab2[ 0x377c ]= 191 ;
tab2[ 0x1c63 ]= 192 ;
tab2[ 0x5023 ]= 193 ;
tab2[ 0x1877 ]= 194 ;
tab2[ 0x3600 ]= 195 ;
tab2[ 0x3f1d ]= 196 ;
tab2[ 0x93c5 ]= 197 ;
tab2[ 0x60e1 ]= 198 ;
tab2[ 0xdf60 ]= 199 ;
tab2[ 0xfdba ]= 200 ;
tab2[ 0xea31 ]= 201 ;
tab2[ 0x666a ]= 202 ;
tab2[ 0x3df1 ]= 203 ;
tab2[ 0x9ad2 ]= 204 ;
tab2[ 0x5f1c ]= 205 ;
tab2[ 0x6b6a ]= 206 ;
tab2[ 0xe49f ]= 207 ;
tab2[ 0xf15d ]= 208 ;
tab2[ 0x231f ]= 209 ;
tab2[ 0xbb19 ]= 210 ;
tab2[ 0x264b ]= 211 ;
tab2[ 0x78e2 ]= 212 ;
tab2[ 0x6054 ]= 213 ;
tab2[ 0x5d4b ]= 214 ;
tab2[ 0x79a8 ]= 215 ;
tab2[ 0xf14f ]= 216 ;
tab2[ 0xaa7e ]= 217 ;
tab2[ 0xdc ]= 218 ;
tab2[ 0xa717 ]= 219 ;
tab2[ 0x628b ]= 220 ;
tab2[ 0xa48a ]= 221 ;
tab2[ 0x91d9 ]= 222 ;
tab2[ 0xd8a6 ]= 223 ;
tab2[ 0x7a47 ]= 224 ;
tab2[ 0x749a ]= 225 ;
tab2[ 0xd68a ]= 226 ;
tab2[ 0x649c ]= 227 ;
tab2[ 0x4a8c ]= 228 ;
tab2[ 0xf6ee ]= 229 ;
tab2[ 0xf16e ]= 230 ;
tab2[ 0xa764 ]= 231 ;
tab2[ 0xa19b ]= 232 ;
tab2[ 0x15bd ]= 233 ;
tab2[ 0xe933 ]= 234 ;
tab2[ 0x2358 ]= 235 ;
tab2[ 0x4685 ]= 236 ;
tab2[ 0xd1c0 ]= 237 ;
tab2[ 0xd0bd ]= 238 ;
tab2[ 0x3982 ]= 239 ;
tab2[ 0xed50 ]= 240 ;
tab2[ 0xa8b8 ]= 241 ;
tab2[ 0xed30 ]= 242 ;
tab2[ 0x94e9 ]= 243 ;
tab2[ 0x400d ]= 244 ;
tab2[ 0xe2bb ]= 245 ;
tab2[ 0xcd5f ]= 246 ;
tab2[ 0xbb17 ]= 247 ;
tab2[ 0x2e56 ]= 248 ;
tab2[ 0xc672 ]= 249 ;
tab2[ 0xbfc0 ]= 250 ;
tab2[ 0x5ba ]= 251 ;
tab2[ 0xa1ac ]= 252 ;
tab2[ 0xca1f ]= 253 ;
tab2[ 0xb018 ]= 254 ;
tab2[ 0x9ef5 ]= 255 ;

    fp_t A, B, D,  c, alpha;  
    ep_t P;
    ep_null(P);
    ep_new(P);
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
    
   
    
    
    fp_null(g);
    fp_new(g);
    
    fp_null(h);
    fp_new(h);

    
    
    
    fp_null(z);
    fp_new(z);

    
    
    
    bn_null(ee);
    bn_new(ee);
    
    bn_null(m);
    bn_new(m);
    
    bn_read_str(tmp,"b",1,16);
    fp_prime_conv(z,tmp);
    
    bn_read_str(m,"6b8e9185f1443ab18ec1701b28524ec688b67cc03d44e3c7bcd88bee82520005c2d7510c00000021423",83,16);
    fp_exp(g,z,m);
    
    bn_read_str(ee,"35c748c2f8a21d58c760b80d94292763445b3e601ea271e3de6c45f741290002e16ba88600000010a11",83,16);
    
    //h=g^(2^(n-w))=g^(2^(42))
    bn_t a1;
    bn_new(a1);
    bn_null(a1);
    bn_read_str(a1,"40000000000",11,16);
    fp_exp(h,g,a1);
    
    
    
    precomputation(g,h);

    


    // Get curve parameter b and lambda
        fp_set_dig(b, 1);
        fp_set_dig(c, 5);
        bn_read_str(tmp,"1AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000000",96,16);
	fp_prime_conv(alpha, tmp);
	fp_neg(D,alpha);
	fp_mul_dig(A,alpha,3);
	fp_mul(B,A,alpha);
	
    
	const char *msg = "Hashing to BLS12-377 and this is the massege";
        size_t msg_len = strlen(msg);
    
	
   	MEASURE(ep_map_elligator2(P, (const uint8_t *)msg, msg_len, b, c, D, A, B, alpha);)
	printf("RDTSC_clk_min=%f\n",RDTSC_clk_min);
	printf("RDTSC_clk_median=%f\n",RDTSC_clk_median);
	printf("RDTSC_clk_max=%f\n",RDTSC_clk_max);
	ep_print(P);
    
    ep_free(P);
    fp_free(alpha); 
    fp_free(b);
    fp_free(A);
    fp_free(B);
    fp_free(D);
    fp_free(c);
    ep_free(P);

    core_clean();
    return 0;
}


