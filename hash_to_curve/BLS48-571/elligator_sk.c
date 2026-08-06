#include <stdio.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include<immintrin.h>

#include <relic/relic_core.h>
#include "relic/relic_md.h"






void findsqroot(fp_t y, fp_t u) {
    fp_t b,I,tmp3;
    fp_t bs;
    bn_t tmp;
    
    
    bn_null(tmp);
    fp_null(I);
    fp_null(b);
    fp_null(bs);
    fp_null(tmp3);
    
    RLC_TRY {
        bn_new(tmp);
        fp_new(b);
        fp_new(I);
        fp_new(bs);
        fp_new(tmp3);
        
        bn_read_str(tmp,
            "2FD2C10DA03FAC68144538C5BA29D4E72EAF66BC66CF1D8992666BAF984D6A1ED0ACC24E225240726B4741FEB723385D27B442F2F542EDD0A4E13CD7FFF92AF4D403A32F7FB552",
            142, 16);

        fp_prime_conv(I, tmp);
        
        bn_read_str(tmp,
            "F63ABEA75727B41BF73E42AC1656E84D33E336622E3B83E9E5FD34F82F91BF9B4777BF28E54DFF228DEA8A493E970AE745D13724429EF89C01221F7F97E4550381F0C00A778516",
            142, 16);
        
        fp_exp(b,u,tmp);
        fp_sqr(bs,b);
        fp_mul(tmp3,b,I);
        if(fp_cmp(bs,u) == RLC_EQ){
            fp_copy(y,b);
        }
        else{
            fp_copy(y,tmp3);
        }
        
    }

    RLC_CATCH_ANY {
	    RLC_THROW(ERR_CAUGHT);
    }
    RLC_FINALLY {
            fp_free(b);
            fp_free(I);
            fp_free(bs);
            fp_free(tmp3);
            bn_free(tmp);
    }
}



void ff_impl(ep_t p, const uint8_t *random, size_t len, fp_t b, fp_t c, fp_t D, fp_t A, fp_t B, fp_t alpha) {
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
		fp_rand(t2);
		
	        c1 = fp_is_zero(t1);
	        if (c1) {
		    ep_set_infty(p);
		} else {
		    fp_sqr(tmp1, t1);
		    fp_mul_dig(tmp1,tmp1,2);
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

		    //fp_sqr(tmp1,x);
		    //fp_mul(tmp1,x,tmp1);
		    //fp_sqr(tmp2,y);
		    //fp_add_dig(tmp1,tmp1,1);
		    //fp_sub(tmp1,tmp1,tmp2);
		    //fp_print(tmp1);

		    fp_copy(p->x,x);
		    fp_copy(p->y,y);
		    fp_set_dig(p->z,1);
		}
		
		c1 = fp_is_zero(t2);
		if (c1) {
		    ep_set_infty(q);
		} else {
		    fp_sqr(tmp1, t2);
		    fp_mul_dig(tmp1,tmp1,2);
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

		    //fp_sqr(tmp1,x);
		    //fp_mul(tmp1,x,tmp1);
		    //fp_sqr(tmp2,y);
		    //fp_add_dig(tmp1,tmp1,1);
		    //fp_sub(tmp1,tmp1,tmp2);
		    //fp_print(tmp1);

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
		static const uint8_t DST[] = "ELLIGATOR-BLS48-571-G1";

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

    printf("Using curve: bls48-571\n");
    
    ep_t P;
    bn_t tmp;
    fp_t alpha, A, B, D, b, c;

    ep_null(P);

    bn_null(tmp);

    fp_null(alpha);
    fp_null(A);
    fp_null(B);
    fp_null(D);
    fp_null(b);
    fp_null(c);

    RLC_TRY {

        ep_new(P);

        bn_new(tmp);

        fp_new(alpha);
        fp_new(A);
        fp_new(B);
        fp_new(D);
        fp_new(b);
        fp_new(c);
    



        
        fp_set_dig(b, 1);
        fp_set_dig(c, 2);
        bn_read_str(tmp,"7B1D5F53AB93DA0DFB9F21560B2B742699F19B31171DC1F4F2FE9A7C17C8DFCDA3BBDF9472A6FF9146F545249F4B8573A2E89B92214F7C4E00910FBFCBF22A81C0F860053BC28AC",143,16);
	fp_prime_conv(alpha, tmp);
        //fp_print(alpha);
	fp_neg(D,alpha);
	fp_mul_dig(A,alpha,3);
	fp_mul(B,A,alpha);
	
	const char *msg = "Hashing to BLS48-571 and this is the msg";
        size_t msg_len = strlen(msg);
    
	
   	MEASURE(ep_map_ff(P, (const uint8_t *)msg, msg_len, b, c, D, A, B, alpha);)
	printf("RDTSC_clk_min=%f\n",RDTSC_clk_min);
	printf("RDTSC_clk_median=%f\n",RDTSC_clk_median);
	printf("RDTSC_clk_max=%f\n",RDTSC_clk_max);
    }	
	
    
    RLC_CATCH_ANY {
        RLC_THROW(ERR_CAUGHT);
    }
    RLC_FINALLY {

        ep_free(P);

        bn_free(tmp);

        fp_free(alpha);
        fp_free(A);
        fp_free(B);
        fp_free(D);
        fp_free(b);
        fp_free(c);
    }

    core_clean();
    return 0;
}

