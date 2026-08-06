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

void koshelov_a0_implt(ep_t p, const uint8_t *random, size_t len, fp_t c[8], fp_t w[8],bn_t e) {
	fp_t t1, t2, g0, den, num0, num1, y0, y1, y2,temp1,temp2,one,theta,theta3,s1,s2,s12,s22,s3,nthree,x,y,k;
	ctx_t *ctx = core_get();
	bn_t kk;
	uint8_t s;

	bn_null(kk);	//
	fp_null(k);//
	fp_null(g0);//
	fp_null(den);//
	fp_null(t1);//
	fp_null(t2);//
	fp_null(s1);
	fp_null(s2);
	fp_null(s12);
	fp_null(s22);
	fp_null(s3);
	fp_null(temp1);//
	fp_null(temp2);//
	fp_null(num0);//
	fp_null(num1);//
	fp_null(y0);//
	fp_null(y1);//
	fp_null(y2);//
	fp_null(nthree);//
	fp_null(one);//
	fp_null(theta);//
	fp_null(theta3);//
	fp_null(x);//
	fp_null(y);//

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
		fp_new(y0);
		fp_new(y1);
		fp_new(y2);
		fp_new(den);
		fp_new(num0);
		fp_new(num1);
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

		
		//fp_print(t1);
		//fp_print(t2);
		fp_sqr(temp1, t2);//temp1=t2^2
		fp_mul(s2,t2,temp1);//s2=t2*temp1

		fp_mul(s3,s1,s2);//s3=s1*s2
	
		fp_sqr(s12,s1);//s12=s1^2
		fp_sqr(s22,s2);//s22=s2^2

		
		//fp_print(s12);
		fp_mul_dig(num0,s12,256);
		fp_copy(den,num0);
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

		    //fp_print(y0);
		    //fp_print(y1);
		    //fp_print(y2);

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
			    fp_copy(y,y0);
		    }
		    
		    fp_mul(temp1,g0,w[2]);
		    if(fp_cmp(theta3, temp1)==RLC_EQ){
			    fp_mul(x,theta,c[2]);
			    fp_mul(x,t1,x);
			    fp_copy(y,y1);
		    }
		    
		    
		    fp_mul(temp1,g0,w[3]);
		    if(fp_cmp(theta3, temp1)==RLC_EQ){
			    fp_mul(x,theta,c[3]);
			    fp_mul(x,t1,x);
			    fp_copy(y,y1);
		    }
		    
		    fp_mul(temp1,g0,w[4]);
		    if(fp_cmp(theta3, temp1)==RLC_EQ){
			    fp_mul(x,theta,c[4]);
			    fp_mul(x,t1,x);
			    fp_copy(y,y1);
		    }
		    
		    fp_mul(temp1,g0,w[5]);
		    if(fp_cmp(theta3, temp1)==RLC_EQ){
			    fp_mul(x,theta,c[5]);
			    fp_mul(x,t2,x);
			    fp_copy(y,y2);
		    }
		    
		    fp_mul(temp1,g0,w[6]);
		    if(fp_cmp(theta3, temp1)==RLC_EQ){
			    fp_mul(x,theta,c[6]);
			    fp_mul(x,t2,x);
			    fp_copy(y,y2);
		    }
		    
		    fp_mul(temp1,g0,w[7]);
		    if(fp_cmp(theta3, temp1)==RLC_EQ){
			    fp_mul(x,theta,c[7]);
			    fp_mul(x,t2,x);
			    fp_copy(y,y2);
		    }
		    
		    //fp_sqr(temp1,x);
		    //fp_mul(temp1,x,temp1);
		    //fp_sqr(temp2,y);
		    //fp_add(g0,one,temp1);
		    //fp_sub(g0,g0,temp2);

		    //fp_print(g0);

		    
		    
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
		fp_free(x);
		fp_free(y);
		fp_free(g0);
		fp_free(t1);
		fp_free(t2);
		fp_free(num0);
		fp_free(num1);
		fp_free(y0);
		fp_free(y1);
		fp_free(y2);
		fp_free(den);
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
		static const uint8_t DST[] = "KOSHELOV-BLS24-509-G1";

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
        printf("Error setting BLS48 curve parameters.\n");
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
    "8a69b7d232c96a6ba790d6f4ba454df0923d69c34cb7da02cfd6dc0b2ba4215948bacc03b009ce6182abd5e00823e7f0800000811e000000001",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[0], tmp);
	
	bn_read_str(tmp,
    "17452a017cbdcddc09857eb4a3f6aca41a2643487de4331660267916a4398c5950e441e0e949534c2a6a96ffc8f69a8e2c6bff93769623ffffffffffffffffff",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[1], tmp);
	
	bn_read_str(tmp,
    "5b22dfc55c18f8aa1d8ba30267eb32a621a9fb87d0516d4ee78b9d92cfb069ee478141e4bb6fc5a8224868df8ee386efe2cb01bbad2dec8ad87bf535a78ebcc",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[2], tmp);
	
	bn_read_str(tmp,
    "434595f9c9fc5040705bb68827518bdd80813fd9edd975415889d59b9352a9501b666eb14fb5d43a1182dc3417ee93eb148b3e03678e9c57ff557e16db64656",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[3], tmp);
	
	bn_read_str(tmp,
    "d5ea2a58a5c81f3fc242c4891a99b34ed72db36b6e08e1132c156af3ba98822d8767991caac8e39b3ee1daf2b6f910b3a549c19c3c96371d28afaab37d0cddf",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[4], tmp);
	
	bn_read_str(tmp,
    "1077dd4208c5697426d3287a67be0cbc4043742143c1b973a5f7d57ad48ff90ebb01c5da46e5a7cbbefc22887a905d47f39baf05eab4b7bc7c3196872404e05c",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[5], tmp);
	
	bn_read_str(tmp,
    "658cd7c3c56fe8fa78981b8b537585cf443a7f06fbfcb984a107748fc24c8a43e53708e071533f3527e3ea695eaf574ff710991ac47dd9f204973c82dd70997",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[6], tmp);
	
	bn_read_str(tmp,
    "747f4337a16e7ed6a5f7ae1da80203f30e72db1f41b72e46ba611e5124f7a3c54fbe32dd640c18c5b070d155615ffbf6bd477e1e1896a4638d0790ae24160e",
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
    "747f4337a16e7ed6a5f7ae1da80203f30e72db1f41b72e46ba611e5124f7a3c54fbe32dd640c18c5b070d155615ffbf6bd477e1e1896a4638d0790ae24160e",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[0], tmp);
	
	bn_read_str(tmp,
    "d5ea2a58a5c81f3fc242c4891a99b34ed72db36b6e08e1132c156af3ba98822d8767991caac8e39b3ee1daf2b6f910b3a549c19c3c96371d28afaab37d0cddf",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[1], tmp);
	
	bn_read_str(tmp,
    "20fbd45ad16235abcf1d41505f1f84fe87e1f9329644446bbdd53699687ae25e517ba28295a622dcfac9cd3dc44bb36b6e91fd5b08af29bbe48f7f62351d609",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[2], tmp);
	
	bn_read_str(tmp,
    "939d18faafeaf8a07872205c0beab61d207f0bf537568d230f300f18cac1841fb96f351f39a86a3ab0a628b7c7a259c801821e1130bd6a7ce2f510e291de34d",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[3], tmp);
	
	bn_read_str(tmp,
    "e4169c6737feeedcf3e44cd8fb29482bbc37516dd6ad41e67c51c393cd85b0f62a631bdb33c9e7473ebdd0a1b5f80b9bf9d08242382e941a98a4bef26ebe0e9",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[4], tmp);
	
	bn_read_str(tmp,
    "8f365b8c74b1e6d74bac7b082effff6276ba5c9ada00c1d343fac22b87726b6796743f5f5541a7339162bb462b94bf6f771c9505142065bcfc0c49fa6bccb7d",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[5], tmp);
	
	bn_read_str(tmp,
    "1129153c9736cb27ab8a478553aa46a7cffd19d88f4f5c828819b296f01cf75fafe05fdeef44bf013275ccf43979a0566fbd85b3c2ee766f52ff4c2526bb663b",
    128,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[6], tmp);
	
	bn_read_str(tmp,
    "13c6735d8cd5c7445282c2e1411459fce9cdb7186ddf05afbd0538b233191a70e2d50427f1942a0bd0636aef839f0d352adc29e241307bd0724cd7b31caf347d",
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
	

	fp_set_dig(b,16);
	fp_set_dig(b2,256);
	fp_set_dig(b3,4096);
	fp_set_dig(b4,65536);
	

	bn_t e;
	bn_null(e);
	bn_new(e);
	/* Make e = p. */
	e->used = RLC_FP_DIGS;
	dv_copy(e->dp, fp_prime_get(), RLC_FP_DIGS);		
	bn_dbl(e,e);		
	bn_add_dig(e, e, 7);
	bn_div_dig(e, e, 27);
    const char *msg = "Hashing to BLS24-509 and this is the massege";
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
    fp_free(b);
	fp_free(b2);
	fp_free(b3);
	fp_free(b4);
    ep_free(P);
	for (int i = 0; i < 8; i++) {
		fp_free(w[i]);
		fp_free(c[i]);
	}	
	
    core_clean();
    return 0;
}

