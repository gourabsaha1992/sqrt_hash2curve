#include <stdio.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include<immintrin.h>

#include <relic/relic_core.h>
#include "relic/relic_md.h"




void koshelov_a0_implt(ep_t p, const uint8_t *random, size_t len, fp_t c[8], fp_t w[8],bn_t e,fp_t B,fp_t bs) {
	fp_t t1, t2, g0, den, num0, num1, num2, y0, y1, y2,temp1,temp2,four,one,theta,theta3,s1,s2,s12,s22,s3,nthree,x,y,k;
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
		fp_new(theta);
		fp_new(theta3);
		fp_new(s1);
		fp_new(s2);
		fp_new(s12);
		fp_new(s22);
		fp_new(s3);
		

		bn_read_bin(kk, random, len / 2);
		fp_prime_conv(t1, kk);
		bn_read_bin(kk, random + len / 2, len / 2);
		fp_prime_conv(t2, kk);
	
		
		
		fp_sqr(temp1, t1);//temp1=t1^2
		fp_mul(s1,t1,temp1);//s1=t1*temp1

		
		
		fp_sqr(temp1, t2);//temp1=t2^2
		fp_mul(s2,t2,temp1);//s2=t2*temp1

		fp_mul(s3,s1,s2);//s3=s1*s2
	
		fp_sqr(s12,s1);//s12=s1^2
		fp_sqr(s22,s2);//s22=s2^2

		
		
		fp_mul_dig(num0,s12,256);
		fp_copy(den,num0);//den=num0
		fp_copy(num2,num0);
		fp_dbl(temp1,num0);
		fp_add(temp2,num0,temp1);
		fp_neg(num1,temp2);
		
		fp_mul_dig(temp1,s3,4096);
		fp_dbl(temp1,temp1);
		fp_neg(temp2,temp1);
		fp_add(num1,num1,temp1);
		fp_add(num0,num0,temp2);
		fp_add(den,den,temp2);
		
		fp_mul_dig(temp1,s1,16);
		fp_dbl(temp1,temp1);
		fp_neg(temp2,temp1);
		fp_add(num1,num1,temp1);
		fp_add(num0,num0,temp1);
		fp_add(den,den,temp2);
		
		fp_mul_dig(temp1,s22,65536);
		fp_add(num1,num1,temp1);
		fp_add(num0,num0,temp1);
		fp_add(den,den,temp1);
		
		fp_mul_dig(temp1,s2,256);
		fp_dbl(temp1,temp1);
		fp_neg(temp2,temp1);
		fp_add(num1,num1,temp2);
		fp_add(num0,num0,temp1);
		fp_add(den,den,temp2);

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
		    fp_mul(den,den,bs);		    

		    fp_mul(y0,num0,den);
		    fp_mul(y1,num1,den);	        

		    fp_add(temp1,y1,y0);
		    fp_add(temp2,temp1,bs);
		    fp_neg(y2,temp2);		    
		    
		    fp_sqr(temp1,y0);
		    fp_sub(g0,temp1,B);		    

		    fp_exp(theta, g0, e);
		    fp_sqr(theta3,theta);
		    fp_mul(theta3,theta3,theta);				    

		    if(fp_cmp(theta3, g0)==RLC_EQ){
			    fp_mul(x,theta,w[0]);
			    fp_mul(x,x,w[0]);
			    fp_copy(y,y0);
		    }
		    

		    fp_mul(temp1,g0,w[0]);
		    if(fp_cmp(theta3, temp1)==RLC_EQ){
			    fp_mul(x,theta,c[0]);
			    fp_mul(x,x,w[0]);
			    fp_copy(y,y0);
		    }
		    
		    fp_mul(temp1,g0,w[1]);
		    if(fp_cmp(theta3, temp1)==RLC_EQ){
			    fp_mul(x,theta,c[1]);
			    fp_mul(x,x,w[0]);
			    fp_copy(y,y0);
		    }
		    
		    fp_mul(temp1,g0,w[2]);
		    if(fp_cmp(theta3, temp1)==RLC_EQ){
			    fp_mul(x,theta,c[2]);
			    fp_mul(x,x,t1);
			    fp_copy(y,y1);
		    }
		    
		    
		    fp_mul(temp1,g0,w[3]);
		    if(fp_cmp(theta3, temp1)==RLC_EQ){
			    fp_mul(x,theta,c[3]);
			    fp_mul(x,x,t1);
			    fp_copy(y,y1);
		    }
		    
		    fp_mul(temp1,g0,w[4]);
		    if(fp_cmp(theta3, temp1)==RLC_EQ){
			    fp_mul(x,theta,c[4]);
			    fp_mul(x,x,t1);
			    fp_copy(y,y1);
		    }
		    
		    fp_mul(temp1,g0,w[5]);
		    if(fp_cmp(theta3, temp1)==RLC_EQ){
			    fp_mul(x,theta,c[5]);
			    fp_mul(x,x,t2);
			    fp_copy(y,y2);
		    }
		    
		    fp_mul(temp1,g0,w[6]);
		    if(fp_cmp(theta3, temp1)==RLC_EQ){
			    fp_mul(x,theta,c[6]);
			    fp_mul(x,x,t2);
			    fp_copy(y,y2);
		    }
		    
		    fp_mul(temp1,g0,w[7]);
		    if(fp_cmp(theta3, temp1)==RLC_EQ){
			    fp_mul(x,theta,c[7]);
			    fp_mul(x,x,t2);
			    fp_copy(y,y2);
		    }
				    
		    fp_copy(p->x,x);
		    fp_copy(p->y,y);
		    fp_set_dig(p->z, 1);
                }
		//fp_print(x);
		//fp_print(y);
		
		//fp_sqr(temp1,x);
		//fp_mul(temp1,x,temp1);
		//fp_sqr(temp2,y);
		//fp_add(g0,B,temp1);
		//fp_sub(g0,g0,temp2);

		//fp_print(g0);
		
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
		fp_free(theta);
		fp_free(theta3);
		fp_free(s1);
		fp_free(s2);
		fp_free(s12);
		fp_free(s22);
		fp_free(s3);
	}

}

void ep_map_koshelov(ep_t p, const uint8_t *msg, size_t len, fp_t c[8], fp_t w[8],bn_t e,fp_t B,fp_t bs) {
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
		static const uint8_t DST[] = "KOSHELOV-BLS12-455-G1";

		md_xmd(r, 2*elm + 1, msg, len,
			   DST, sizeof(DST) - 1);


		koshelov_a0_implt(p, r, 2 * elm + 1, c, w,e,B,bs);
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

    printf("Using curve: bls12-455\nMethod Koshelov\n");

    /* -------------------------------
     * Inputs to SwiftEC
     * ------------------------------- */
    fp_t w[8];
	for (int i = 0; i < 8; i++) {
		fp_null(w[i]);
		fp_new(w[i]);
	}

	bn_t tmp;
	bn_null(tmp);
	bn_new(tmp);
	bn_read_str(tmp,
    "100000a00005000040fd01a8e80a01584e3f61a092d7e1132d32e6daef13241cddb5e43c5be071f4017fd0057ffff801",
    115,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[0], tmp);
	
	fp_sqr(w[1],w[0]);
	
	bn_read_str(tmp,
    "f8f1dc06a505edbcdc97c6467849182cdf68066de6c75b4af49a780be8b9204055816a05c237f3b80b10d1a688bed84b0578e10dc210f3e2e",
    115,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[2], tmp);
	
	fp_mul(w[3],w[0],w[2]);
	fp_mul(w[4],w[1],w[2]);
	fp_sqr(w[5],w[2]);
	fp_mul(w[6],w[5],w[0]);
	fp_mul(w[7],w[5],w[1]);
	
	
	
	fp_t c[8];
	for (int i = 0; i < 8; i++) {
		fp_null(c[i]);
		fp_new(c[i]);
	}

	fp_inv(c[0],w[2]);
	
	fp_sqr(c[1],c[0]);
	
	bn_read_str(tmp,
    "305a448e53ce3bea2765e26d9aeb8e7cc0398343fd57f00a8aef17b0252655540fa464380fb4a768beae3abd9590acfec18e125043a2ab5398",
    115,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[2], tmp);
	
	fp_mul(c[3],c[2],c[0]);
	fp_mul(c[4],c[2],c[1]);
	
	bn_read_str(tmp,
    "249e4a71327faa80fdd88040578cbe11580e0124579c40981c8fc8676e9a6496ac2068017aa0d76bf230395988a401f7617e2e11f2fcc926f1",
    115,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[5], tmp);
	
	fp_mul(c[6],c[5],c[0]);
	fp_mul(c[7],c[5],c[1]);

fp_t b,b2,b3,b4,bs,B;
	fp_null(b);
	fp_null(B);
	fp_null(b2);
	fp_null(b3);
	fp_null(b4);
	fp_null(bs);
	
	
	fp_new(b);
	fp_new(B);
	fp_new(b2);
	fp_new(b3);
	fp_new(b4);
	fp_new(bs);
	

	fp_set_dig(b,16);
	fp_set_dig(b2,256);
	fp_set_dig(b3,4096);
	fp_set_dig(b4,65536);
	fp_set_dig(B,10);
	fp_srt(bs,B);



	bn_t e;
	bn_null(e);
	bn_new(e);
	/* Make e = p. */
	e->used = RLC_FP_DIGS;
	dv_copy(e->dp, fp_prime_get(), RLC_FP_DIGS);
	bn_dbl(e,e);		
	bn_add_dig(e, e, 7);
	bn_div_dig(e, e, 27);
    const char *msg = "hashing to BLS12-455 and this is the message";
    size_t msg_len = strlen(msg);

    ep_t P;
    ep_null(P);
    ep_new(P);

    
	
   	MEASURE(ep_map_koshelov(P, (const uint8_t *)msg, msg_len,c,w,e,B,bs);)
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
    fp_free(b);
    fp_free(b2);
    fp_free(b3);
    fp_free(b4);
    ep_free(P);

    core_clean();
    return 0;
}

