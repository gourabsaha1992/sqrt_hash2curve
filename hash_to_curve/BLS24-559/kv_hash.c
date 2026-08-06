#include <stdio.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include<immintrin.h>

#include <relic/relic_core.h>
#include "relic/relic_md.h"




void koshelov_a0_implt(ep_t p, const uint8_t *random, size_t len, fp_t c[8], fp_t w[8],bn_t e,fp_t b,fp_t b2,fp_t b3,fp_t b4,fp_t bs) {
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
	
		//fp_print(t1);
		//fp_print(t2);
		
		
		fp_sqr(temp1, t1);//temp1=t1^2
		fp_mul(s1,t1,temp1);//s1=t1*temp1

		
		
		fp_sqr(temp1, t2);//temp1=t2^2
		fp_mul(s2,t2,temp1);//s2=t2*temp1

		fp_mul(s3,s1,s2);//s3=s1*s2
	
		fp_sqr(s12,s1);//s12=s1^2
		fp_sqr(s22,s2);//s22=s2^2
                //fp_print(s1);
                //fp_print(s2);
                //fp_print(s3);
                //fp_print(s12);
                //fp_print(s22);
		
		
		fp_mul(num0,s12,b2);
		fp_copy(den,num0);
		fp_dbl(temp1,num0);
		fp_add(temp2,num0,temp1);
		fp_neg(num1,temp2);
		
                
		
		fp_mul(temp1,s3,b3);
		fp_dbl(temp1,temp1);
		fp_neg(temp2,temp1);
		fp_add(num1,num1,temp1);
		fp_add(num0,num0,temp2);
		fp_add(den,den,temp2);

		
		fp_mul(temp1,s1,b);
		fp_dbl(temp1,temp1);
		fp_neg(temp2,temp1);
		fp_add(num1,num1,temp1);
		fp_add(num0,num0,temp1);
		fp_add(den,den,temp2);

		
		
		fp_mul(temp1,s22,b4);
		fp_add(num1,num1,temp1);
		fp_add(num0,num0,temp1);
		fp_add(den,den,temp1);

		
		fp_mul(temp1,s2,b2);
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
		//fp_print(num0);
                
		
		fp_mul(num0,num0,bs);
		fp_mul(num1,num1,bs);
		
		
		int c1 = fp_is_zero(den);
		if (c1) {
		    ep_set_infty(p);
		}else{
		    fp_inv(den,den);

		    fp_mul(y0,num0,den);
		    fp_mul(y1,num1,den);
		    fp_add(y2,y0,y1);
		    fp_add(y2,y2,bs);
		    fp_neg(y2,y2);
		    
		    
		    //fp_print(y0);
		    //fp_print(y1);
		    //fp_print(y2);

		    fp_add(temp1,y0,y1);
		    fp_add(temp1,temp1,y2);
		    fp_add(temp1,temp1,one);
		    
		    fp_sqr(temp1,y0);
		    fp_sub(g0,temp1,b);
		    
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
		    
		    //fp_sqr(temp1,x);
		    //fp_mul(temp1,x,temp1);
		    //fp_sqr(temp2,y);
		    //fp_add(g0,b,temp1);
		    //fp_sub(g0,g0,temp2);

		    //fp_print(g0);
	    	    
		    fp_copy(p->x,x);
		    fp_copy(p->y,y);
		    fp_set_dig(p->z,1);
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

void ep_map_koshelov(ep_t p, const uint8_t *msg, size_t len, fp_t c[8], fp_t w[8],bn_t e,fp_t b,fp_t b2,fp_t b3,fp_t b4,fp_t bs) {
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
		static const uint8_t DST[] = "KOSHELOV-BLS24-559-G1";

		md_xmd(r, 2*elm + 1, msg, len,
			   DST, sizeof(DST) - 1);


		koshelov_a0_implt(p, r, 2 * elm + 1, c, w,e,b,b2,b3,b4,bs);
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

    printf("Using curve: bls24-559\n");

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
    "100480900a7f81f4f531e21f95a82680eee7759bf554f0221eaa041055962826449970c48302f3504aa8c135bf9de4300d79cc37ad16a35b61e3b02cab1e1be",
    140,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[0], tmp);
	
	bn_read_str(tmp,
    "557003c04ffe88db09e6861d7101b9cafe50d88ea12fc52fc0689677f5edce67234ea2ba4413618bdd85f3beb41ea8f683f6af65d3ce194fe07518916d4a1539e4e5ea4f242c",
    140,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[1], tmp);
	
	bn_read_str(tmp,
    "17a28e62b696e211d099202fdd1fd1479e1e1f2b36b0aaf5541ba40e8c79d76ca28ea7e0de03ebd749d8cfb1e5b3f18e09b46666460551aa3bb77fff932b60250986e347100b",
    140,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[2], tmp);
	
	bn_read_str(tmp,
    "11897bda4c24cfdd632db5032d36780067a4c21c7a64425f01af40708d40c617536e417650fd43a8e5195385e77641d655f37d8b7dae95385d924a3a7f866538b3220530a2a5",
    140,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[3], tmp);
	
	bn_read_str(tmp,
    "2c43f9834d42d7ec1e28b1925ecabfd616aff0a17282e6c9e1f7714e2b3552cdcd92bebc7794965545a018b716297a3cb062276dee5d3344e3eec928c4ce05fa633fcc89533b",
    140,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[4], tmp);
	
	bn_read_str(tmp,
    "35316d434d506389975e7541ddc807308f3ebe72b187974d8ae2948d7ddb22f1a86203ab7fc09076249edbc95c726c4a088f8d7ca3bead490b0c1a2bb0d41bac961f943d5453",
    140,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[5], tmp);
	
	bn_read_str(tmp,
    "4f7a3570fd302d99f72f8589d5854e3709e43c8ee695e5492d7c3cbf483574689df282a30da76538b54abe45cb76f17a0e674fb87ba37403154d5c12ac81d04e6b04010dec46",
    140,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[6], tmp);
	
	bn_read_str(tmp,
    "263464cc557c8293155112bf1ef4bcd49fc2a8d0af122ba5b725da4dc3cf494940cac9d8bfc395fc0f3addce9ebdfd7e091d398a44c01302da17b08751a9aab53eadd4b6cb3d",
    140,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(w[7], tmp);
	
	
	fp_t c[8];
	for (int i = 0; i < 8; i++) {
		fp_null(c[i]);
		fp_new(c[i]);
	}


	bn_read_str(tmp,
    "263464cc557c8293155112bf1ef4bcd49fc2a8d0af122ba5b725da4dc3cf494940cac9d8bfc395fc0f3addce9ebdfd7e091d398a44c01302da17b08751a9aab53eadd4b6cb3d",
    140,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[0], tmp);
	
	bn_read_str(tmp,
    "2c43f9834d42d7ec1e28b1925ecabfd616aff0a17282e6c9e1f7714e2b3552cdcd92bebc7794965545a018b716297a3cb062276dee5d3344e3eec928c4ce05fa633fcc89533b",
    140,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[1], tmp);
	
	bn_read_str(tmp,
    "1beaed14e2c40711a64c68a7b71d935e2d25613c15aee5af8452c3966b255f81620f59276cdaf5512fa9a18295ae121132ae451069e199602c29182e7e7a83bf4a3038cac857",
    140,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[2], tmp);
	
	bn_read_str(tmp,
    "32f3be0431a7a289514a53bf03457248eaa134e3a9ef8353b2c90d2555ebda5714dfeee3230d37f8a7c90d82cb72c7856c480a87eec11b0a4d5362ec5e672b2b27d478dc1859",
    140,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[3], tmp);
	
	bn_read_str(tmp,
    "461fc35f8ef009c1df7aef4c52acba62e25a8026a15e7f131fd339b4ef31b4e27cc1e04f09c55a638a9941ae95d342e982281961f2c70346e0768530a0b6bad19e9c488151f8",
    140,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[4], tmp);
	
	bn_read_str(tmp,
    "14d73f76756550c0bafce0ca5c4ea3c1dc65ac155ae073f27f8e9c63c46e398a5bf5b35704ac6c329d1a859c608c9913941c092854d210d8bc3e81b4e159e96ef4ceda3b3821",
    140,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[5], tmp);
	
	bn_read_str(tmp,
    "506a9e0aee0979b5feceb2b6b928deff95dacadd80bae17c1d7d3b82109f460c620393b004ad10409f61f7947bb35432c001357a4fdb454932b05f35f547a8ef26164712b5b4",
    140,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[6], tmp);
	
	bn_read_str(tmp,
    "154609fccc1a8eaae29c180fcad98fd522ebc0a611c269f97a93c5d6712b6da9c4aca6dff60f09c7296f006ada63d661499d7a87d8710242fd264f53d7ed962271244bcccd13",
    140,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(c[7], tmp);
	
	fp_t b,b2,b3,b4,bs;
	fp_null(b);
	fp_null(b2);
	fp_null(b3);
	fp_null(b4);
	fp_null(bs);
	
	
	fp_new(b);
	fp_new(b2);
	fp_new(b3);
	fp_new(b4);
	fp_new(bs);
	
        bn_read_str(tmp,
    "557003c04ffe89db51ef86c56921091e1c72d1e92397d41e37c255cd44eff051c38fa813a695c5d574923beee353ada1100a0b5fb2111a277d389362d77fcb581fe8b50105e9",
    140,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(b, tmp);
	fp_set_dig(b2,4);
	bn_read_str(tmp,
    "557003c04ffe89db51ef86c56921091e1c72d1e92397d41e37c255cd44eff051c38fa813a695c5d574923beee353ada1100a0b5fb2111a277d389362d77fcb581fe8b50105e3",
    140,   // number of hex chars
    16);

	/* Convert integer → field element */
	fp_prime_conv(b3, tmp);
	fp_set_dig(b4,16);
	fp_srt(bs,b);
	

	bn_t e;
	bn_null(e);
	bn_new(e);
	/* Make e = p. */
	e->used = RLC_FP_DIGS;
	dv_copy(e->dp, fp_prime_get(), RLC_FP_DIGS);		
	bn_add_dig(e, e, 8);
	bn_div_dig(e, e, 27);
    const char *msg = "Hash to BLS24-559";
    size_t msg_len = strlen(msg);

    ep_t P;
    ep_null(P);
    ep_new(P);

    
	
   	MEASURE(ep_map_koshelov(P, (const uint8_t *)msg, msg_len,c,w,e,b,b2,b3,b4,bs);)
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
	fp_free(bs);
    ep_free(P);

    core_clean();
    return 0;
}

