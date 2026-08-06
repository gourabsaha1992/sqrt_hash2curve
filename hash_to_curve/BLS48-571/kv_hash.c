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

		    //fp_print(x);
		    //fp_print(y);
		    
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
		static const uint8_t DST[] = "KOSHELOV-BLS48-571-G1";

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

    printf("Using curve: bls48-571\n");

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
    "1a62a67742a66582d3bf35696cd23b56d0976ef26a1fb039e140306a484294c738ec80bae9be7ce94de0498e09b6cf4d2d7c162c52ca5ce3eb9e58eaeb57a8930059d9c2",
    143,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[0], tmp);
	
	bn_read_str(tmp,
    "7b1d5f5205697299d138c928cf381d8fcccde5c40da6d2ce510396de03c5d9291f929320e3def3e2ab0d768fc146ec93077ba6bf498e1988d3eb4181120c9bd30b7dd6d53624eea",
    143,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[1], tmp);
	bn_read_str(tmp,
    "15e249e41d13420c78022c9a335791a64d97f06345bb1979579530d80614c4ddd659c4d8b79fbce2f788998890f3cbadd8b66dda5fc77a1474bbf185259435a6189ebea12261788",
    143,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[2], tmp);
	
	bn_read_str(tmp,
    "36f9b78301cbe83f00c4dd3f1fdd71260946c3a31813f39012ca4d14dec3f426a66f0295f4a651115f323fa31c754c2990c14d41e6ecb1d9e30fe54e49ebfbda4e6a569c732888f",
    143,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[3], tmp);
	
	bn_read_str(tmp,
    "2e415dec8cb4afc282d8177cb7f6715a4312e72ab94eb4eb889f1c8f32f026c926f31825c660f19cf03a6bf8f1e26d9c3970e075da9b505fa8c538ec5c71f90159ef4ac7a638896",
    143,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[4], tmp);
	
	bn_read_str(tmp,
    "60c8966a5f135aee0e1f53ff46f90d098e557125ec271f3c83004f1d00da7c6dc04d7d492631343b78fdc59b542de21c8cb7684f0dfa1aeee9147b7dd6fbe1d8fb058a0718ad3e8",
    143,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[5], tmp);
	
	bn_read_str(tmp,
    "785d43a92c8e46350002fca88ee1891c9d9f9aec6b6346d092d0c08017fa090d1eaf0dd036aa5a51a340d41f50cb4de5d7d1bb29588659014b05f2db3b3ea4c786ebd9e8b4243ba",
    143,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[6], tmp);
	
	bn_read_str(tmp,
    "1d14e493cb8612f8e91bf204407c522707ee2a4fd6b11ddcd02c255b16bd3a20687b340f8872709571abf08e999ddae4e14813abdc1e84abcd07b12685a9ce62ffff5c1aaab39b8",
    143,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[7], tmp);

	
	fp_t c[8];
	for (int i = 0; i < 8; i++) {
		fp_null(c[i]);
		fp_new(c[i]);
	}

	bn_read_str(tmp,
    "1d14e493cb8612f8e91bf204407c522707ee2a4fd6b11ddcd02c255b16bd3a20687b340f8872709571abf08e999ddae4e14813abdc1e84abcd07b12685a9ce62ffff5c1aaab39b8",
    143,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[0], tmp);
	
	bn_read_str(tmp,
    "2e415dec8cb4afc282d8177cb7f6715a4312e72ab94eb4eb889f1c8f32f026c926f31825c660f19cf03a6bf8f1e26d9c3970e075da9b505fa8c538ec5c71f90159ef4ac7a638896",
    143,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[1], tmp);
	
	bn_read_str(tmp,
    "618ec467a3f02ee226aa8bdd1727f44dc9d70a2d88c157caffa4a5ad9b412df17cb36182897614660c8cfa198491e6fde3e00e4a2289df06a5b8869258d7bcfc6445ca8ff3b3dc0",
    143,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[2], tmp);
	
	bn_read_str(tmp,
    "246af68f9b4c193a5dbed26b1fd343b6df224af376aeca62a5b99890d0ce187a3e14d53d636431e351121beafa0c62bc1309eeef9cdfde40f12c5407787e40600c69de0214436b4",
    143,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[3], tmp);
	
	bn_read_str(tmp,
    "48d4d1cad6982da7f80771a48f9de2c8abe640d6bc3230c80656f272cb0d7e9a6498887ae3c194bdcf876d623204863b6cf24a12c52ff9265f7cfb83cb86d846b5058482320bd45",
    143,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[4], tmp);
	
	bn_read_str(tmp,
    "7900a1bc3087690ea7c4db54fd7b6d7514e749e0312fda97eb127491b6bce02974525150f51ac7a4278900e375469ca4d6eafa3eb70aa2017a658024c2835fda4b62ce3c6cb3c94",
    143,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[5], tmp);
	
	bn_read_str(tmp,
    "6bc1db5366b72665dbf3f4f6bdc24ec9835aa200e79bf6e3eb6bc1bb2433099a02c20669c0def912a1a4ce431d22fee7bac70dc676dbc357e679aa5f18afb72b756e52ffffa233c",
    143,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[6], tmp);
	
	bn_read_str(tmp,
    "249682c4a38fde7c7fdd428992dafc66348f2e628250b38221cfe2a7367fa18836e6bf0f88ee44e19817ddc3f23c60c7b8642c8c1518a544fa93d2693c0b2ebba1fa277ae409d27",
    143,   // number of hex chars
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
	bn_add_dig(e, e, 8);
	bn_div_dig(e, e, 27);
    const char *msg = "Hash to BLS48-571 and this is the msg";
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

