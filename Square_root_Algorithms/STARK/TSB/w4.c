#include <stdio.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include<immintrin.h>

#include <relic/relic_core.h>
#include "relic/relic_md.h"

#define w 4
#define we 16
#define n 192
#define k1 48
#define l 47
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


void findsqroot(fp_t u, fp_t y,fp_t tab1[k1][we],int tab2[32],bn_t ee, bn_t tmp) {
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
	        d=d&0x1f;
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
	            d=d&0x1f;
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
    int tab2[32]={0};

    for(int i=0;i<k1;i++){
        for(int j=0;j<we;j++){
            fp_null(tab1[i][j]);
            fp_new(tab1[i][j]);
        }
    }

    tab2[ 0x1 ]= 0 ;
    tab2[ 0x19 ]= 1 ;
    tab2[ 0x1b ]= 2 ;
    tab2[ 0x12 ]= 3 ;
    tab2[ 0x3 ]= 4 ;
    tab2[ 0x11 ]= 5 ;
    tab2[ 0xd ]= 6 ;
    tab2[ 0xc ]= 7 ;
    tab2[ 0x0 ]= 8 ;
    tab2[ 0x8 ]= 9 ;
    tab2[ 0x6 ]= 10 ;
    tab2[ 0xf ]= 11 ;
    tab2[ 0x1e ]= 12 ;
    tab2[ 0x10 ]= 13 ;
    tab2[ 0x14 ]= 14 ;
    tab2[ 0x15 ]= 15 ;

    
    
    
    
    
    
   
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

