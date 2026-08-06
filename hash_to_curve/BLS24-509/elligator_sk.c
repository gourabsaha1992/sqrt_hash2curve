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
	int c1;
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
		static const uint8_t DST[] = "ELLIGATOR-BLS24-509-G1";

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
        printf("Error setting BLS24 curve parameters.\n");
        core_clean();
        return 1;
    }

    printf("Using curve: bls24-509\n");

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
        bn_read_str(tmp,"155556FFFF39CA9BFCEDF2B4F9C0ECF6CB8AC8495D187E8C32EA0103E01090BB626E85BF7C18A0F0CFCB5C6071BAD3D2EE63BD076E8D9300A13D118DB8BFD2AA",128,16);
	fp_prime_conv(alpha, tmp);
        //fp_print(alpha);
	fp_neg(D,alpha);
	fp_mul_dig(A,alpha,3);
	fp_mul(B,A,alpha);
	
	bn_read_str(tmp,"12BF078CF16682196FAF4E4DFBD2D2B10691603AA2FDA0B8130F7A0AA198A64CF88798F97B5A5C2C34D786CF45436CBA6EC28602044FED90A6CA1F47477B9F56",128,16);
	fp_prime_conv(point[0], tmp);
	
	bn_read_str(tmp,"F4F02A8BD4E63BC1B37BB748CA5F34B44BB430C4F575D16C2C1ECA7DFC46DF9E905445BD37A7E336ACEB92BD8AB871591573A29A9BAC4D0F815DE0F884DCE34",128,16);
	fp_prime_conv(point[1], tmp);
	
	bn_read_str(tmp,"606545741EB66DFE1B637406D1AF9AB86CF853D0DC121757028145C004C22C179694163A89E22BD64FCA334990F4CBD5D0C82DDC4D2CE2FA927337E30720477",128,16);
	fp_prime_conv(point[2], tmp);
	
	bn_read_str(tmp,"2964F730DD348828D3EA466FDEE1A45C4F9680EBA1ADDD41FDA86F93E77EA6E69E6ECC600BE44C49AF3D5912C7767187FA137056A3DA56FFA72F24671443355",128,16);
	fp_prime_conv(point[3], tmp);
	
	bn_read_str(tmp,"11E5521BCB21AC3EA8765FDB8A940D9109B4AB1B09723AEAE29C73A11E3C586852EC3121D438C2F805C28EBD0522EE2E10F8712F13F86A40F288D055F9920189",128,16);
	fp_prime_conv(point[4], tmp);
	
	bn_read_str(tmp,"37004E434181E5D547792D96F2CDF65C1D61D2E53A643A1504D8D62C1D438530F82549DA7DFDDF8CA08CDA36C97E5A4DD6B4BD85A9528BFAEB44137BF2DD122",128,16);
	fp_prime_conv(point[5], tmp);

    
	const char *msg = "Hashing to BLS24-509 and this is the msg";
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

