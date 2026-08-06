#include <stdio.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include<immintrin.h>

#include <relic/relic_core.h>
#include "relic/relic_md.h"




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
		static const uint8_t DST[] = "ELLIGATOR-BLS12-446-G1";

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
        printf("Error setting BLS12 curve parameters.\n");
        core_clean();
        return 1;
    }

    printf("Using curve: bls12-446\n");

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
        bn_read_str(tmp,"3CDEE0FB28C5E535200FC34965AAD6400095A4B78A02FE320F75A64BBAC71602824E6DC3E23ACDEE56EE4528C573B5CC311C0026AAB0AAAA",112,16);
	fp_prime_conv(alpha, tmp);
        //fp_print(alpha);
	fp_neg(D,alpha);
	fp_mul_dig(A,alpha,3);
	fp_mul(B,A,alpha);
	
	bn_read_str(tmp,"3514888B5A68614DCA85C3E806A84E3CAFBAD24EDF9BB4845B5B4CCDC4279A3F44BBF6BC0B19371358557D96D843A85C6A27E928E15571CE",112,16);
	fp_prime_conv(point[0], tmp);
	
	bn_read_str(tmp,"2589AC78A6D3923274443B7966E52831558F87A2CB2790BD445C66B18B94910536FE5E95D511D87E4B380B4AA262EB113A30A5520ABF7939",112,16);
	fp_prime_conv(point[1], tmp);
	
	bn_read_str(tmp,"1755348281F25302ABCB87CFFEC5AE0EAB061D14BEDB6D74CB193F9A2F3284FD4B500F2E0D28F5700BB639DE2310CABAF6EB5AD49FF13172",112,16);
	fp_prime_conv(point[2], tmp);
	
	bn_read_str(tmp,"7CA586FCE5D83E75589FF615F02880350DAD268AA6749ADB41A597DF69F7BC33D927707D72196DAFE98C791ED300D6FC6F416FDC95B38DD",112,16);
	fp_prime_conv(point[3], tmp);
	
	bn_read_str(tmp,"1F1F8CF2504FD6EA015587315DC83611FBE0EF7D6942B7227F33991825D200C088E28635E44A8C4B0A4F01701040D82ABDDF71D2694C6A4F",112,16);
	fp_prime_conv(point[4], tmp);
	
	bn_read_str(tmp,"1DBF5408D8760E4B1EBA3C1807E2A02E04B4B53A20C0470F90420D3394F51541F96BE78DFDF041A34C9F43B8B532DDA1733C8E544164405C",112,16);
	fp_prime_conv(point[5], tmp);
	
	
    
	const char *msg = "Hashing to BLS12-446 and this is the msg";
        size_t msg_len = strlen(msg);
    
	
   	MEASURE(ep_map_ff(P, (const uint8_t *)msg, msg_len, b, c, D, A, B, alpha, point);)
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
    for(int i=0;i<6;i++){
        fp_free(point[i]);
    }

    core_clean();
    return 0;
}

