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

void ff_impl(ep_t p, const uint8_t *random, size_t len, fp_t b, fp_t c, fp_t D, fp_t A, fp_t B, fp_t alpha, fp_t point[6]) {
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
	        if (c1 || fp_cmp(t1, point[0])==RLC_EQ || fp_cmp(t1, point[1])==RLC_EQ || fp_cmp(t1, point[2])==RLC_EQ || fp_cmp(t1, point[3])==RLC_EQ|| fp_cmp(t1, point[4])==RLC_EQ|| fp_cmp(t1, point[5])==RLC_EQ) {
		    ep_set_infty(p);
		} else {
		    fp_sqr(tmp1, t1);
		    fp_neg(tmp1,tmp1);
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
                

                    fp_srt(p->y,y);
                    fp_add(p->x,x,alpha);
		    fp_set_dig(p->z,1);
		}
		
		c1 = fp_is_zero(t2);
		if (c1 || fp_cmp(t2, point[0])==RLC_EQ || fp_cmp(t2, point[1])==RLC_EQ || fp_cmp(t2, point[2])==RLC_EQ || fp_cmp(t2, point[3])==RLC_EQ || fp_cmp(t2, point[4])==RLC_EQ || fp_cmp(t2, point[5])==RLC_EQ) {
		    ep_set_infty(q);
		} else {
		    fp_sqr(tmp1, t2);
		    fp_neg(tmp1,tmp1);
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
                

                    fp_srt(q->y,y);
                    fp_add(q->x,x,alpha);
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

void ep_map_ff(ep_t p, const uint8_t *msg, size_t len,  fp_t b, fp_t c, fp_t D, fp_t A, fp_t B, fp_t alpha, fp_t point[6]) {
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
		static const uint8_t DST[] = "ELLIGATOR-BLS48-575-G1";

		md_xmd(r, 2*elm + 1, msg, len,
			   DST, sizeof(DST) - 1);


		ff_impl(p, r, 2 * elm + 1, b, c, D, A, B, alpha, point);
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

    fp_t A, B, D, b, c, alpha, point[6];
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
    
    for(int i=0;i<6;i++){
        fp_new(point[i]);
        fp_null(point[i]);
    }


    // Get curve parameter b and lambda
        fp_set_dig(b, 1);
        fp_set_dig(c, 2);
        bn_read_str(tmp,"526222098BE0D6809EC22A43AB79D1F2120ECBCFBAB8934E6C19B1B30A442EEA74C92F3ED3940030DBE05F531CD41C69464C793821806366EC094551C2828E22D811DCC31BEB02AA",144,16);
	fp_prime_conv(alpha, tmp);
        //fp_print(alpha);
	fp_neg(D,alpha);
	fp_mul_dig(A,alpha,3);
	fp_mul(B,A,alpha);
	
	bn_read_str(tmp,"5120649A9D0DB422EB5FC5083E666932FFEE8BEE7E11BB79B212E8358311950A51E43055E2C1A72B586D28EB4E340560455387C7B9F0BB90BB1B320105E4E7D77022B5C1999E2EB2",144,16);
	fp_prime_conv(point[0], tmp);
	
	bn_read_str(tmp,"2FEFAA9AF77904B8DBE90F992E45F611F02690C3DF9E135C1C317CE157161A1B19CCC513DF7608877444DE7E2EEEC75A496C341A18E8468F9C75206021D5CF4B1F98FD6E545936F4",144,16);
	fp_prime_conv(point[1], tmp);
	
	bn_read_str(tmp,"2272776E9467D1C7C2D91AAA7D33DBE021E83B0BDB1A7FF24FE834D1B32E14CF5AFC6A2AF41DF7A9679B80D4EDE5550EFCE0451E08981CD74F9424F1A0ACBED7B878DF54C791CBB7",144,16);
	fp_prime_conv(point[2], tmp);
	
	bn_read_str(tmp,"141BD6EEED3225DB362653B6D1368BF12203FE13CA6D7D4BA06C97D873299E022E4FEE8F0D2590583733667CEA0170900F8F170678FA7D630EE1350BC9DA64B67EF2701824CD3F9",144,16);
	fp_prime_conv(point[3], tmp);
	
	bn_read_str(tmp,"31316809E64C27168F4B74D49B595ED10246D0A51C44EB30D638465EDE48B3FB3CB1C3FCD048618CF7B814E5FD8EDE634A65258A8077EE65CD6333B0DE7375968788246FD6A60AED",144,16);
	fp_prime_conv(point[4], tmp);
	
	bn_read_str(tmp,"2130B9FFA594AF6A0F76B56F102073210FC7FB2A9E73A81D95E16B542BFB7AEF38176B42034B9EA3E4284A6D1F453E05FBE753ADA10875011EA611A0E40F188C5089B8534544F7BE",144,16);
	fp_prime_conv(point[5], tmp);

	
    
	const char *msg = "Hashing to BLS48-575 and this is the msg";
        size_t msg_len = strlen(msg);
    
	
   	MEASURE(ep_map_ff(P, (const uint8_t *)msg, msg_len, b, c, D, A, B, alpha, point);)
	printf("RDTSC_clk_min=%f\n",RDTSC_clk_min);
	printf("RDTSC_clk_median=%f\n",RDTSC_clk_median);
	printf("RDTSC_clk_max=%f\n",RDTSC_clk_max);
    
    ep_free(P);
    fp_free(alpha); 
    fp_free(b);
    fp_free(A);
    fp_free(B);
    fp_free(D);
    fp_free(c);
    for(int i=0;i<6;i++){
        fp_free(point[i]);
    }
    bn_free(tmp);

    core_clean();
    return 0;
}

