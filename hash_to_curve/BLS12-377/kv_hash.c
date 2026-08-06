#include <stdio.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include<immintrin.h>

#include <relic/relic_core.h>
#include "relic/relic_md.h"


static void fp_cmov(fp_t r, const fp_t a, int flag) {
    if (flag) {
        fp_copy(r, a);
    }
}

void koshelov_a0_implt(ep_t p, const uint8_t *random, size_t len, fp_t c[2], fp_t w[2],bn_t e) {
	fp_t t1, t2, g0, den, num0, num1, num2, y0, y1, y2,temp1,temp2,four,one,theta,theta3,s1,s2,s12,s22,s3,nthree,x,y,k,zero;
	ctx_t *ctx = core_get();
	bn_t kk;
	uint8_t s;

	bn_null(kk);	
	fp_null(k);
	fp_null(g0);
	fp_null(den);
	fp_null(t1);
	fp_null(t2);
	fp_null(s1);
	fp_null(s2);
	fp_null(s12);
	fp_null(s22);
	fp_null(s3);
	fp_null(temp1);
	fp_null(temp2);
	fp_null(num0);
	fp_null(num1);
	fp_null(num2);
	fp_null(y0);
	fp_null(y1);
	fp_null(y2);
	fp_null(four);
	fp_null(nthree);
	fp_null(one);
	fp_null(zero);
	fp_null(theta);
	fp_null(theta3);
	fp_null(x);
	fp_null(y);

	RLC_TRY {
		fp_new(k);
		bn_new(kk);
		fp_new(g0);
		fp_new(x);
		fp_new(y);
		fp_new(t1);
		fp_new(t2);
		fp_new(temp1);
		fp_new(temp2);
		fp_new(x1);
		fp_new(x2);
		fp_new(x3);
		fp_new(num0);
		fp_new(num1);
		fp_new(num2);
		fp_new(four);
		fp_new(nthree);
		fp_new(one);
		fp_new(zero);
		fp_new(theta);
		fp_new(theta3);
		fp_new(s1);
		fp_new(s2);
		fp_new(s12);
		fp_new(s22);
		fp_new(s3);
		for (size_t i = 0; i < 8; i++) {
			fp_null(h[i]);
			fp_new(h[i]);
		}

		bn_read_bin(kk, random, len / 2);
		fp_prime_conv(t1, kk);
		bn_read_bin(kk, random + len / 2, len / 2);
		fp_prime_conv(t2, kk);
		
		//fp_print(t1);
		//fp_print(t2);
		
	      
		
		fp_sqr(temp1, t1);//temp1=t1^2
		fp_mul(s1,t1,temp1);//s1=t1*temp1

		fp_sqr(temp1, t2);//temp1=t2^2
		fp_mul(s2,t2,temp1);//s2=t2*temp1

		fp_mul(s3,s1,s2);//s3=s1*s2

	
		fp_sqr(s12,s1);//s12=s1^2

		fp_sqr(s22,s2);//s22=s2^2

		fp_set_dig(zero, 0);
		
		fp_mul_dig(num0,s12,16);//num0=2^4*s12
		fp_copy(den,num0);//den=num0
		fp_add(temp1,num0,num0);
		fp_add(temp2,num0,temp1);
		fp_neg(num1,temp2);
		
		//fp_set_dig(k,128);
		fp_mul_dig(temp1,s3,128);
		fp_neg(temp2,temp1);
		fp_add(num1,num1,temp1);
		fp_add(num0,num0,temp2);
		fp_add(den,den,temp2);
		
		
		//fp_set_dig(k,8);
		fp_mul_dig(temp1,s1,8);
		fp_neg(temp2,temp1);
		fp_add(num1,num1,temp1);
		fp_add(num0,num0,temp1);
		fp_add(den,den,temp2);


		//fp_set_dig(k,256);
		fp_mul_dig(temp1,s22,256);
		fp_add(num1,num1,temp1);
		fp_add(num0,num0,temp1);
		fp_add(den,den,temp1);

		//fp_set_dig(k,32);
		fp_mul_dig(temp1,s2,32);
		fp_neg(temp2,temp1);
		fp_add(num1,num1,temp2);
		fp_add(num0,num0,temp1);
		fp_add(den,den,temp2);
		//fp_print(den);

		fp_set_dig(nthree, 3);
		fp_set_dig(one, 1);
		
		
		fp_sub(num0,num0,nthree);
		fp_add(num1,num1,one);
		fp_add(den,den,one);
		
		int c1;
                c1 = fp_is_zero(den);
		if (c1) {
		    ep_set_infty(p);
		}else{
		    fp_inv(den,den);
		    fp_mul(y0,num0,den);
		    fp_mul(y1,num1,den);
		    fp_add(temp1,y1,y0);
		    fp_add(temp2,temp1,one);
		    fp_neg(y2,temp2);		
		    fp_sqr(temp1,y0);
		    fp_sub(g0,temp1,one);
		    

		    fp_exp(theta, g0, e);
		    fp_sqr(theta3,theta);
		    fp_mul(theta3,theta3,theta);
	      
		    
				    

		    if(fp_cmp(theta3, g0)==RLC_EQ){
			    fp_mul(x, w[0], theta);
			    fp_mul(x, w[0], x);
			    fp_copy(y,y0);
		    }
		    

		    fp_mul(temp1,g0,w[0]);
		    if(fp_cmp(theta3, temp1)==RLC_EQ){
			    fp_mul(x,theta,c[0]);
			    fp_mul(x,x,t1);
			    fp_copy(y,y1);
		    }
		    
		    fp_mul(temp1,g0,w[1]);
		    if(fp_cmp(theta3, temp1)==RLC_EQ){
			    fp_mul(x,theta,c[1]);
			    fp_mul(x,x,t2);
			    fp_copy(y,y2);
		    }
		    //fp_sqr(temp1,x);
		    //fp_mul(temp1,x,temp1);
		    //fp_add(temp1,temp1,one);
		    //fp_sqr(temp2,y);
		    //fp_sub(temp1,temp1,temp2);
		    //fp_print(temp1);
			    
		    fp_copy(p->x,x);
		    fp_copy(p->y,y);
		    fp_set_dig(p->z, 1);
		}    
		/* Multiply by cofactor. */
		ep_mul_cof(p, p);
	}
	RLC_CATCH_ANY {
		RLC_THROW(ERR_CAUGHT);
	}
	RLC_FINALLY {
		fp_free(k);
		bn_free(kk);
		fp_free(v);
		fp_free(y);
		fp_free(t1);
		fp_free(t2);
		fp_free(num0);
		fp_free(num1);
		fp_free(num2);
		fp_free(y0);
		fp_free(y1);
		fp_free(y2);
		fp_free(four);
		fp_free(nthree);
		fp_free(one);
		fp_free(zero);
		fp_free(theta);
		fp_free(theta3);
		fp_free(s1);
		fp_free(s2);
		fp_free(s12);
		fp_free(s22);
		fp_free(s3);
		for (size_t i = 0; i < 8; i++) {
			fp_free(h[i]);
		}
	}

}

void ep_map_koshelov(ep_t p, const uint8_t *msg, size_t len, fp_t c[2], fp_t w[2],bn_t e) {
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
		static const uint8_t DST[] = "KOSHELOV-BLS12-377-G1";

		md_xmd(r, 2*elm + 1, msg, len,
			   DST, sizeof(DST) - 1);


		koshelov_a0_implt(p, r, 2 * elm + 1, c, w,e);
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

    printf("Using curve: bls12-377\nMethod Koshelov\n");

    /* -------------------------------
     * Inputs to SwiftEC
     * ------------------------------- */
    fp_t w[2];
	for (int i = 0; i < 2; i++) {
		fp_null(w[i]);
		fp_new(w[i]);
	}

	bn_t tmp;
	bn_null(tmp);
	bn_new(tmp);
	bn_read_str(tmp,
    "1ae3a4617c510eabc8756ba8f8c524eb8882a75cc9bc8e359064ee822fb5bffd1e945779fffffffffffffffffffffff",
    96,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[0], tmp);
	
	bn_read_str(tmp,
    "9b3af05dd14f6ec619aaf7d34594aabc5ed1347970dec00452217cc900000008508c00000000001",
    96,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[1], tmp);

	
	
	
	fp_t c[2];
	for (int i = 0; i < 2; i++) {
		fp_null(c[i]);
		fp_new(c[i]);
	}

	bn_read_str(tmp,
    "16e0866cfa8a6dddb67ec3ff52b32e660d147851baa0103d1fa275b31c8c01e29f5ed5c07bf8c069771d34745ef6681",
    96,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[0], tmp);
	
	bn_read_str(tmp,
    "18eecbd5b0539aa5f0abaa21f1a8c583995fcf94c5442420064be9dd2a33540f899187ad7094ae2249dd5cb5682977e",
    96,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[1], tmp);
	





	bn_t e;
	bn_null(e);
	bn_new(e);
	

	/* Make e = p. */
	e->used = RLC_FP_DIGS;
	dv_copy(e->dp, fp_prime_get(), RLC_FP_DIGS);		
	bn_add_dig(e, e, 2);
	bn_div_dig(e, e, 9);
    const char *msg = "hashing to BLS12-377 and this is the massege";
    size_t msg_len = strlen(msg);

    ep_t P;
    ep_null(P);
    ep_new(P);

    
	
   	MEASURE(ep_map_koshelov(P, (const uint8_t *)msg, msg_len,c,w,e);)
	printf("RDTSC_clk_min=%f\n",RDTSC_clk_min);
	printf("RDTSC_clk_median=%f\n",RDTSC_clk_median);
	printf("RDTSC_clk_max=%f\n",RDTSC_clk_max);
    
    /* Normalize and print result */
    ep_norm(P, P);

    printf("P = ");
    ep_print(P);
    printf("\n");
	
    /* -------------------------------
     * Cleanup
     * ------------------------------- */
    fp_free(u);
    ep_free(P);
    for (int i = 0; i < 2; i++) {
		fp_free(c[i]);
	}
    for (int i = 0; i < 2; i++) {
		fp_free(w[i]);
	}

    core_clean();
    return 0;
}

