#include <stdio.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include<immintrin.h>

#include <relic/relic_core.h>
#include "relic/relic_md.h"


void ff_impl(ep_t p, const uint8_t *random, size_t len, fp_t b, fp_t c, fp_t D, fp_t A, fp_t B, fp_t alpha) {
	fp_t t1, t2,t3, yy1, yy2,tmp1,tmp2,tmp3,x1,y1,z1,x,y,z;
	ctx_t *ctx = core_get();
	bn_t kk;
	ep_t q;
        ep_null(q);
	uint8_t s;
	int c1, c2, c3;
	bn_null(kk);	
	fp_null(t1);
	fp_null(t2);
	fp_null(t3);
	fp_null(tmp1);
	fp_null(tmp2);
	fp_null(tmp3);
	fp_null(yy1);
	fp_null(yy2);
	fp_null(x1);
	fp_null(y1);
	fp_null(z1);
	fp_null(x);
	fp_null(y);

	RLC_TRY {
		bn_new(kk);
		fp_new(t1);
		fp_new(t2);
		fp_new(t3);
		fp_new(tmp1);
		fp_new(tmp2);
		fp_new(tmp3);
		fp_new(yy1);
		fp_new(yy2);
		fp_new(x1);
		fp_new(y1);
		fp_new(z1);
		fp_new(x);
		fp_new(y);
		fp_new(z);
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
	            fp_mul(t3,tmp1,t1);
		    fp_neg(tmp1,tmp1);
		    fp_add_dig(x,tmp1,1);
		    fp_set_dig(z,1);
		    
		    fp_sqr(tmp3,x);   
		    fp_mul(y,tmp3,x); 
		    fp_mul(tmp2,tmp3,A);
		    fp_add(y,y,tmp2);
		    fp_mul(tmp2,x,B);
		    fp_add(y,y,tmp2);
		    
		    fp_srt(yy1,y);
		    fp_sqr(yy2,yy1);
		    fp_mul(y1,yy1,tmp1);
		    fp_neg(y1,y1);
		    fp_mul(x1,x,t3);
		    fp_mul(z1,t3,tmp1); 
		    
		    int d=1;
		    if(fp_cmp(yy2, y)==RLC_EQ){
		        d=0;
		    }
		    fp_copy(y,yy1);
		    
		    fp_copy_sec(x, x1, d);
		    fp_copy_sec(y, y1, d);
		    fp_copy_sec(z, z1, d);
		    
		    fp_mul(tmp1,alpha,z);
                    fp_add(x,x,tmp1);
                    fp_inv(z,z);
                    fp_mul(p->x,x,z);
                    fp_mul(p->y,y,z);
		    fp_set_dig(p->z,1);
		}
		
                c1 = fp_is_zero(t2);
		if (c1) {
		    ep_set_infty(q);
		} else {
	            fp_sqr(tmp1, t2);
	            fp_mul(t3,tmp1,t2);
		    fp_neg(tmp1,tmp1);
		    fp_add_dig(x,tmp1,1);
		    
		    fp_set_dig(z,1);
		    
		    
		    fp_sqr(tmp3,x);   
		    fp_mul(y,tmp3,x); 
		    fp_mul(tmp2,tmp3,A);
		    fp_add(y,y,tmp2);
		    fp_mul(tmp2,x,B);
		    fp_add(y,y,tmp2);
		    
		    fp_srt(yy1,y);
		    fp_sqr(yy2,yy1);
		    fp_mul(y1,yy1,tmp1);
		    fp_neg(y1,y1);
		    fp_mul(x1,x,t3);
		    fp_mul(z1,t3,tmp1); 
		    
		    int d=1;
		    if(fp_cmp(yy2, y)==RLC_EQ){
		        d=0;
		    }
		    fp_copy(y,yy1);
		    
		    fp_copy_sec(x, x1, d);
		    fp_copy_sec(y, y1, d);
		    fp_copy_sec(z, z1, d);
		    
		    fp_mul(tmp1,alpha,z);
                    fp_add(x,x,tmp1);
                    fp_inv(z,z);
                    fp_mul(q->x,x,z);
                    fp_mul(q->y,y,z);
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
		fp_free(t3);
		fp_free(tmp1);
		fp_free(tmp2);
		fp_free(tmp3);
		fp_free(yy1);
		fp_free(yy2);
		fp_free(x1);
		fp_free(y1);
		fp_free(z1);
		fp_free(x);
		fp_free(y);
		fp_free(z);
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
		static const uint8_t DST[] = "FF-BLS48-575-G1";

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
        printf("Error setting BLS48 curve parameters.\n");
        core_clean();
        return 1;
    }

    printf("Using curve: bls48-575\n");

    fp_t A, B, D, b, c, alpha;
    bn_t tmp;   
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


    // Get curve parameter b and lambda
        fp_set_dig(b, 1);
        fp_set_dig(c, 2);
        bn_read_str(tmp,"526222098BE0D6809EC22A43AB79D1F2120ECBCFBAB8934E6C19B1B30A442EEA74C92F3ED3940030DBE05F531CD41C69464C793821806366EC094551C2828E22D811DCC31BEB02AA",144,16);
	fp_prime_conv(alpha, tmp);
	fp_neg(D,alpha);
	fp_mul_dig(A,alpha,3);
	fp_mul(B,A,alpha);
	
    
	const char *msg = "Hashing to BLS48-575 and this is the msg";
        size_t msg_len = strlen(msg);
    
	
   	MEASURE(ep_map_ff(P, (const uint8_t *)msg, msg_len, b, c, D, A, B, alpha);)
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

    core_clean();
    return 0;
}

