#include <stdio.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include<immintrin.h>

#include <relic/relic_core.h>
#include "relic/relic_md.h"




void koshelov_a0_implt(ep_t p, const uint8_t *random, size_t len, fp_t c[8], fp_t w[8],bn_t e,fp_t b,fp_t b2,fp_t b3,fp_t b4) {
	fp_t h[8], t1, t2, g0, den, num0, num1, num2, y0, y1, y2,temp1,temp2,nfour,one,theta,theta3,s1,s2,s12,s22,s3,nthree,x,y,k;
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
	fp_null(nfour);
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
		fp_new(nfour);
		fp_new(nthree);
		fp_new(one);
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
	
//		fp_print(t1);
//		fp_print(t2);
		
		
		fp_sqr(temp1, t1);//temp1=t1^2
		fp_mul(s1,t1,temp1);//s1=t1*temp1

		
		
		fp_sqr(temp1, t2);//temp1=t2^2
		fp_mul(s2,t2,temp1);//s2=t2*temp1

		fp_mul(s3,s1,s2);//s3=s1*s2
	
		fp_sqr(s12,s1);//s12=s1^2
		fp_sqr(s22,s2);//s22=s2^2

		
		
		fp_mul_dig(num0,s12,36);
		fp_copy(den,num0);
		fp_dbl(temp1,num0);
		fp_add(temp2,num0,temp1);
		fp_neg(num1,temp2);

		
		fp_mul_dig(temp1,s3,216);
		fp_dbl(temp1,temp1);
		fp_neg(temp2,temp1);
		fp_add(num1,num1,temp1);
		fp_add(num0,num0,temp2);
		fp_add(den,den,temp2);

		
		fp_mul_dig(temp1,s1,6);
		fp_dbl(temp1,temp1);
		fp_neg(temp2,temp1);
		fp_add(num1,num1,temp1);
		fp_add(num0,num0,temp1);
		fp_add(den,den,temp2);

		
		
		fp_mul_dig(temp1,s22,1296);
		fp_add(num1,num1,temp1);
		fp_add(num0,num0,temp1);
		fp_add(den,den,temp1);

		
		fp_mul_dig(temp1,s2,36);
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

		
		int c1 = fp_is_zero(den);
		if (c1) {
		    ep_set_infty(p);
		}else{		
		    fp_inv(den,den);

		    fp_mul(y0,num0,den);
		    fp_mul(y1,num1,den);
		    fp_add(y2,y0,y1);
		    fp_add(y2,y2,one);
		    fp_neg(y2,y2);
		    
		    fp_add(temp1,y0,y1);
		    fp_add(temp1,temp1,y2);
		    fp_add(temp1,temp1,one);

		    fp_sqr(temp1,y0);
		    fp_sub(g0,temp1,one);


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

		    //fp_print(x);
		    //fp_print(y);
		    
    //	        fp_sqr(temp1,x);
    //		fp_mul(temp1,x,temp1);
    //		fp_sqr(temp2,y);
    //		fp_add(g0,one,temp1);
    //		fp_sub(g0,g0,temp2);

    //		fp_print(g0);
	        
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
		fp_free(nfour);
		fp_free(nthree);
		fp_free(one);
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

void ep_map_koshelov(ep_t p, const uint8_t *msg, size_t len, fp_t c[8], fp_t w[8],bn_t e,fp_t b,fp_t b2,fp_t b3,fp_t b4) {
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
		static const uint8_t DST[] = "KOSHELOV-BLS24-509-G1";

		md_xmd(r, 2*elm + 1, msg, len,
			   DST, sizeof(DST) - 1);


		koshelov_a0_implt(p, r, 2 * elm + 1, c, w,e,b,b2,b3,b4);
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
        printf("Error setting BLS24 curve parameters.\n");
        core_clean();
        return 1;
    }

    printf("Using curve: bls24-509\n");

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
    "155556ffff39c29bfc5df2f86dc55735cb26710c0bea08834769617ba8ef8d725bdc82c320d1d2baef84b27c18d18d4f3249d7c7f2299f62b363c36fe1429aa2",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[0], tmp);
	
	bn_read_str(tmp,
    "800008fffbc8bfb95c10064573d512e7608eb809f8837210349069202fc5b46ce35e046a9e458e94683bc19e53f7c63f39dedd94e1dd77d3808",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[1], tmp);
	
	bn_read_str(tmp,
    "6689a0792bc2a7adb12c1c879a1c246cc48a68b282884b38230d6d3b8e95b7d117b55155680911463e0e53a180ea9b9b857a5ebeb7675f9ebcd58a7d6ef240a",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[2], tmp);
	
	bn_read_str(tmp,
    "7e72bf3d06c27a4b306c9d3ade2161d228c0a1cce054343d0543cc80784f6b91bc6e3e6b3c4e77b42a862075edfec2bc4a4355caeb88ed688e4cdcab66045b",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[3], tmp);
	
	bn_read_str(tmp,
    "e6e4a392f76dda6d6aac44f4541094e2d19611c680fa5a473b3e663a6aee5d2bf36c26bba5bc164b7bff105e3be2b5679c1d3c5b82b94194ce16c09366aaa46",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[4], tmp);
	
	bn_read_str(tmp,
    "11a9a01ed549c1f3c93f4f18be9b5d083f53a02b96c221fd196a297c3b57f41bc8c54c49b1abbde2121dea85e15a7b75b058af1bb6dbe51693106ca4fcf6c933",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[5], tmp);
	
	bn_read_str(tmp,
    "1bf314ee1c45844a5750d2b368093b30b363e876f4bbac92bebd4c2614c5201f819c9396611ceacbf6b6948d6bec1106273c72ad50c67ea1f0b50bf8c95ec3",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[6], tmp);
	
	bn_read_str(tmp,
    "38fc3cc3bd3c323e95752c987bd86b35b83c4354f61a0e286c11a3b7ea3d77f7a279ce2340bc623f1b6bb4602f46c4c37e3d1790a60e76b6c3befdcc2ffaab5",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[7], tmp);
	
	
	fp_t c[8];
	for (int i = 0; i < 8; i++) {
		fp_null(c[i]);
		fp_new(c[i]);
	}

	bn_read_str(tmp,
    "38fc3cc3bd3c323e95752c987bd86b35b83c4354f61a0e286c11a3b7ea3d77f7a279ce2340bc623f1b6bb4602f46c4c37e3d1790a60e76b6c3befdcc2ffaab5",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[0], tmp);
	
	bn_read_str(tmp,
    "e6e4a392f76dda6d6aac44f4541094e2d19611c680fa5a473b3e663a6aee5d2bf36c26bba5bc164b7bff105e3be2b5679c1d3c5b82b94194ce16c09366aaa46",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[1], tmp);
	
	bn_read_str(tmp,
    "11f6b0275f100aa0f72dde73c6fb2c2a61327609fe0f3e7ce5f47f7063fe5efce86b4b567fd3f6a154d4854b11bebd3774436dd475de28a93a28fab80b0b829f",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[2], tmp);
	
	bn_read_str(tmp,
    "c155a05579b5392d1c066bd2fedacddf7a8a908a5f03173d981b848be4d45ac94370e42619f5608d51a5c867ffa12483d4d427ccac31485a7b974d610eac4c3",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[3], tmp);
	
	bn_read_str(tmp,
    "60a80e118c0dfff6adaf5a5b2e339a0fe23434abafb5c7ef9ce18d57a27c0af29074b2306082c53e1b780516f5df3ccd890a11516cc14356b1c7973e516a46a",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[4], tmp);
	
	bn_read_str(tmp,
    "4de32f7ccc2281ac109b8f304049c931451e17dfd237e2e6c259c0cfb3ce91a2ba7ba3d55113619af35696cb9baa6f50ed615f32c8a38874014634997427158",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[5], tmp);
	
	bn_read_str(tmp,
    "1b7afe97813ae089cf1fd05819d60f3b2e6cf03d47075fc3efaa95c2322f27ac4a7165d8d3192bd1db56b40c983ad42c2f834e7e6011cf12d939fa14a7be173",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[6], tmp);
	
	bn_read_str(tmp,
    "6e24cd6a37febd8f8c3101b00a4a5d9dcd11c080a81f5f4e12e94ca83f9a31290d55a204135462bc46f9638613f748e5866d19123a860d4286a28864e041cc6",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[7], tmp);
	
	fp_t b,b2,b3,b4;
	fp_null(b);
	fp_null(b2);
	fp_null(b3);
	fp_null(b4);
	
	
	fp_new(b);
	fp_new(b2);
	fp_new(b3);
	fp_new(b4);
	

	fp_set_dig(b,6);
	fp_set_dig(b2,36);
	fp_set_dig(b3,216);
	fp_set_dig(b4,1296);
	

	bn_t e;
	bn_null(e);
	bn_new(e);
	/* Make e = p. */
	e->used = RLC_FP_DIGS;
	dv_copy(e->dp, fp_prime_get(), RLC_FP_DIGS);
	bn_dbl(e,e);		
	bn_add_dig(e, e, 7);
	bn_div_dig(e, e, 27);
    const char *msg = "Hash to BLS24-509 ";
    size_t msg_len = strlen(msg);

    ep_t P;
    ep_null(P);
    ep_new(P);

    
	
   	MEASURE(ep_map_koshelov(P, (const uint8_t *)msg, msg_len,c,w,e,b,b2,b3,b4);)
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
    fp_free(b);
	fp_free(b2);
	fp_free(b3);
	fp_free(b4);
    ep_free(P);

    core_clean();
    return 0;
}

