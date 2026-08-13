#include <stdio.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include<immintrin.h>
#include <relic/relic_core.h>
#include "relic/relic_md.h"

#define w 4
#define we 16
#define n 96
#define k3 4
#define nw 24
#define count 1 


int i,j,ii,jj,i3;
int l[k3]= {23, 24, 24, 24} ;
uint64_t ll[k3]={
0x7fffff ,
0xffffff ,
0xffffff ,
0xffffff };
int kk[k3+1][k3+1]= {{0, 0, 0, 0, 0}, {49, 0, 0, 0, 0}, {25, 48, 0, 0, 0}, {1, 24, 48, 0, 0}, {1, 24, 48, 72, 0}} ;
int s[k3]= {18, 18, 18, 18} ;
int t[k3]= {5, 5, 5, 5} ;
int t_max= 5 ;
int r[k3]= {1, 0, 0, 0} ;
int ep[k3+1][k3+1]= {{0, 0, 0, 0, 0}, {12, 0, 0, 0, 0}, {6, 12, 0, 0, 0}, {0, 6, 12, 0, 0}, {0, 5, 11, 17, 0}} ;
int rho[k3+1][k3+1]= {{0, 0, 0, 0, 0}, {1, 0, 0, 0, 0}, {1, 0, 0, 0, 0}, {1, 0, 0, 0, 0}, {0, 3, 3, 3, 0}} ;

/* ---------------- 120-bit fixed-width integer support ---------------- */
#define LIMB_BITS 60
#define LIMB_MASK ((1ULL << LIMB_BITS) - 1)

typedef struct { uint64_t low; uint64_t high; } uint120_t;

static inline uint120_t make120(uint64_t h,uint64_t l)
{ return (uint120_t){.low=l&LIMB_MASK,.high=h&LIMB_MASK}; }

static inline uint120_t add120(uint120_t a,uint120_t b){
    uint64_t r=a.low+b.low,c=r>>LIMB_BITS;
    return make120((a.high+b.high+c)&LIMB_MASK,r&LIMB_MASK);}

static inline uint120_t sub120(uint120_t a,uint120_t b){
    uint64_t r=(1ull<<LIMB_BITS)+a.low-b.low,bw=1-(r>>LIMB_BITS);
    return make120((a.high-b.high-bw)&LIMB_MASK,r&LIMB_MASK);}

static inline uint120_t rshift120(uint120_t a,int sh){
    if(sh<=0)return a; if(sh>=120)return make120(0,0);
    if(sh<LIMB_BITS)return make120((a.high>>sh)&LIMB_MASK,
        ((a.low>>sh)|(a.high<<(LIMB_BITS-sh)))&LIMB_MASK);
    return make120(0,(a.high>>(sh-LIMB_BITS))&LIMB_MASK);}

static inline uint120_t lshift120(uint120_t a, int sh)
{
    if (sh <= 0) return a;
    if (sh >= 120) return make120(0, 0);
    if (sh < LIMB_BITS)
        return make120(((a.high << sh) | (a.low >> (LIMB_BITS - sh))) & LIMB_MASK,
                        (a.low << sh) & LIMB_MASK);
    return make120((a.low << (sh - LIMB_BITS)) & LIMB_MASK, 0);
}

#define POW96  make120(1ull<<36,0)
#define MASK48 0x0000FFFFFFFFFFFFull
/* CHUNK96 w=4: k<15 from low, k>=15 from high, no straddle */
#define CHUNK96(e,k) (((k)<15)?(uint8_t)((e).low>>((k)*4))&0xF \
                               :(uint8_t)((e).high>>(((k)-15)*4))&0xF)
/* ----------------------------------------------------------------------- */



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
int iii,jjj,ttt;
uint64_t d;
uint64_t c1,c,one=1;
uint64_t and_f=15;
bn_t Q,temp2;

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
	for (iii = 0; iii < REPEAT; iii++){ \
		min = 	RDTSC_clk[iii]; \
		for (jjj = iii+1; jjj< REPEAT; jjj++){ \
			if (min > RDTSC_clk[jjj]){ \
				min = RDTSC_clk[jjj]; \
				ttt = jjj; \
			} \
		} \
		l1 = RDTSC_clk[ttt]; RDTSC_clk[ttt] = RDTSC_clk[iii]; RDTSC_clk[iii] = l1; \
	}; \
	RDTSC_clk_min = RDTSC_clk[0]; \
	RDTSC_clk_median = RDTSC_clk[REPEAT/2]; \
	RDTSC_clk_max = RDTSC_clk[REPEAT-1];\
}




void precomputation(fp_t g,fp_t tabl2[we][nw]){
    bn_t temp,one,itemp;
    bn_null(temp);
    bn_new(temp);
    bn_null(one);
    bn_new(one);
    bn_set_dig(one,1);
    bn_null(itemp);
    bn_new(itemp);
    
    for(int vv=0;vv<we;vv++){
        for(int i=0;i<nw;i++){
            bn_set_dig(itemp,vv);
            bn_lsh(temp,one,i*w);
            bn_mul(temp,temp,itemp);
            fp_exp(tabl2[vv][i],g,temp);
        }
    }
}

uint120_t Qacc,term,shifted;


void findsqroot(fp_t u, fp_t y,fp_t tabl2[we][nw],int tab3[128],uint64_t q[k3][t_max+1],uint64_t qq[k3],bn_t e, bn_t tmp) {
	fp_t v,v1,x[k3],gamma,alpha,temp,f,a[t_max+1];
	uint120_t ss, tt;
	
	
	fp_null(f);
	for(i=0;i<t_max+1;i++){
		fp_null(a[i]);
	}

	
	
	fp_null(v);
	fp_null(f);
	fp_null(v1);
	fp_null(gamma);
	fp_null(alpha);
	fp_null(temp);

	RLC_TRY {
		
		for(i=0;i<t_max+1;i++){
		        fp_new(a[i]);
	        }
		fp_new(v);
		fp_new(f);
		fp_new(v1);
		fp_new(gamma);
		fp_new(alpha);
		fp_new(temp);

		
		fp_exp(v, u, e);
		fp_mul(v1,v,u);
		fp_mul(x[k3-1],v,v1);
		for(int j=0;j<k3-1;j++)
		{
		   fp_copy(x[k3-2-j],x[k3-1-j]);
		   for(i=1; i<=l[k3-1-j];i++)
		   {
		      fp_sqr(x[k3-2-j],x[k3-2-j]);
		   }
		}
		ss = make120(0, 0);
		tt = make120(0, 0);
		i3=0;
		
	        tt = add120(ss, tt);
	        tt = rshift120(tt, l[i3]);
	        fp_set_dig(gamma,one);
                for(ii=i3-1;ii>=0;ii--){
                    shifted = rshift120(tt, kk[i3][ii]);
                    qq[ii] = shifted.low & ll[ii];
                }

                for(j=0;j<i3;j++){
                    fp_set_dig(temp,one);
                    for(jj=0;jj<t[j]+1;jj++){
                        q[j][jj]=qq[j] & and_f;
                        qq[j]=qq[j]>>w;
                    }
                    for(jj=0;jj<t[j]+1;jj++){
                        fp_mul(temp,temp,tabl2[q[j][jj]][(jj+ep[i3][j])]);
                    }
                    for(jj=1;jj<=rho[i3][j];jj++){
                        fp_sqr(temp,temp);
                    }
                    fp_mul(gamma,gamma,temp);
                }
	        fp_mul(alpha,x[i3],gamma);
	        fp_copy(a[t[i3]],alpha);
	        for(i=1;i<t[i3]+1;i++){
                        fp_copy(a[t[i3]-i],a[t[i3]+1-i]);
		        for(j=1;j<=w;j++)
		        {
		            fp_sqr(a[t[i3]-i],a[t[i3]-i]);
		        }
	        }
	        
	        for(j=0;j<t[i3]+1;j++){
		        fp_copy(f,a[j]);
		        if(j!=0){
		            fp_copy(temp,tabl2[q[i3][0]][s[i3]+t[i3]-j]);
		            for(i=1;i<=r[i3];i++)
		            {
		                fp_sqr(temp,temp);
		            }
		            fp_mul(f,f,temp);
	                }
		        for(i=1;i<j;i++)
		        {
		            fp_mul(f,f,tabl2[q[i3][i]][nw-(j+1-i)]);
		        }
		        if(j==0){
                            fp_prime_back(tmp, f);
                            bn_get_dig(&d, tmp);
                            d=d&0x7f;
                            q[i3][j]=tab3[d]>>(w-l[0]+t[0]*w);
                                    }
                                    if(j!=0){
                                        fp_prime_back(tmp, f);
                            bn_get_dig(&d, tmp);
                            d=d&0x7f;
                            q[i3][j]=tab3[d];
                        }
	        }

	        Qacc = make120(0, q[i3][0]);
	        for (i = 1; i < t[i3] + 1; i++) {
	            term = lshift120(make120(0, q[i3][i]), (l[i3] - (t[i3]+1-i) * w));
	            Qacc = add120(Qacc, term);
	        }
	        ss = lshift120(Qacc, (n - l[i3]));
		i3=1;
		
	        tt = add120(ss, tt);
	        tt = rshift120(tt, l[i3]);
	        fp_set_dig(gamma,one);
                for(ii=i3-1;ii>=0;ii--){
                    shifted = rshift120(tt, kk[i3][ii]);
                    qq[ii] = shifted.low & ll[ii];  
                }

                for(j=0;j<i3;j++){
                    fp_set_dig(temp,one);
                    for(jj=0;jj<t[j]+1;jj++){
                        q[j][jj]=qq[j] & and_f;
                        qq[j]=qq[j]>>w;
                    }
                    for(jj=0;jj<t[j]+1;jj++){
                        fp_mul(temp,temp,tabl2[q[j][jj]][(jj+ep[i3][j])]);
                    }
                    for(jj=1;jj<=rho[i3][j];jj++){
                        fp_sqr(temp,temp);
                    }
                    fp_mul(gamma,gamma,temp);
                }
	        fp_mul(alpha,x[i3],gamma);
	        fp_copy(a[t[i3]],alpha);
	        for(i=1;i<t[i3]+1;i++){
                        fp_copy(a[t[i3]-i],a[t[i3]+1-i]);
		        for(j=1;j<=w;j++)
		        {
		            fp_sqr(a[t[i3]-i],a[t[i3]-i]);
		        }
	        }
	        
	        for(j=0;j<t[i3]+1;j++){
		        fp_copy(f,a[j]);
		        if(j!=0){
		            fp_copy(temp,tabl2[q[i3][0]][s[i3]+t[i3]-j]);
		            for(i=1;i<=r[i3];i++)
		            {
		                fp_sqr(temp,temp);
		            }
		            fp_mul(f,f,temp);
	                }
		        for(i=1;i<j;i++)
		        {
		            fp_mul(f,f,tabl2[q[i3][i]][nw-(j+1-i)]);
		        }
		        if(j==0){
                            fp_prime_back(tmp, f);
                            bn_get_dig(&d, tmp);
                            d=d&0x7f;
                            q[i3][j]=tab3[d]>>(w-l[k3-1]+t[k3-1]*w);
                                    }
                                    if(j!=0){
                                        fp_prime_back(tmp, f);
                            bn_get_dig(&d, tmp);
                            d=d&0x7f;
                            q[i3][j]=tab3[d];
                        }
	        }

	        Qacc = make120(0, q[i3][0]);
	        for (i = 1; i < t[i3] + 1; i++) {
	            term = lshift120(make120(0, q[i3][i]), (l[i3] - (t[i3]+1-i) * w));
	            Qacc = add120(Qacc, term);
	        }
	        ss = lshift120(Qacc, (n - l[i3]));
	    
		i3=2;
		tt = add120(ss, tt);
	        tt = rshift120(tt, l[i3]);
	        fp_set_dig(gamma,one);
                for(ii=i3-1;ii>=0;ii--){
                    shifted = rshift120(tt, kk[i3][ii]);
                    qq[ii] = shifted.low & ll[ii];  
                }

                for(j=0;j<i3;j++){
                    fp_set_dig(temp,one);
                    for(jj=0;jj<t[j]+1;jj++){
                        q[j][jj]=qq[j] & and_f;
                        qq[j]=qq[j]>>w;
                    }
                    for(jj=0;jj<t[j]+1;jj++){
                        fp_mul(temp,temp,tabl2[q[j][jj]][(jj+ep[i3][j])]);
                    }
                    for(jj=1;jj<=rho[i3][j];jj++){
                        fp_sqr(temp,temp);
                    }
                    fp_mul(gamma,gamma,temp);
                }
	        fp_mul(alpha,x[i3],gamma);
	        fp_copy(a[t[i3]],alpha);
	        for(i=1;i<t[i3]+1;i++){
                        fp_copy(a[t[i3]-i],a[t[i3]+1-i]);
		        for(j=1;j<=w;j++)
		        {
		            fp_sqr(a[t[i3]-i],a[t[i3]-i]);
		        }
	        }
	        
	        for(j=0;j<t[i3]+1;j++){
		        fp_copy(f,a[j]);
		        if(j!=0){
		            fp_copy(temp,tabl2[q[i3][0]][s[i3]+t[i3]-j]);
		            for(i=1;i<=r[i3];i++)
		            {
		                fp_sqr(temp,temp);
		            }
		            fp_mul(f,f,temp);
	                }
		        for(i=1;i<j;i++)
		        {
		            fp_mul(f,f,tabl2[q[i3][i]][nw-(j+1-i)]);
		        }
		        if(j==0){
                            fp_prime_back(tmp, f);
                            bn_get_dig(&d, tmp);
                            d=d&0x7f;
                            q[i3][j]=tab3[d]>>(w-l[k3-1]+t[k3-1]*w);
                                    }
                                    if(j!=0){
                                        fp_prime_back(tmp, f);
                            bn_get_dig(&d, tmp);
                            d=d&0x7f;
                            q[i3][j]=tab3[d];
                        }
	        }

	        Qacc = make120(0, q[i3][0]);
	        for (i = 1; i < t[i3] + 1; i++) {
	            term = lshift120(make120(0, q[i3][i]), (l[i3] - (t[i3]+1-i) * w));
	            Qacc = add120(Qacc, term);
	        }
	        ss = lshift120(Qacc, (n - l[i3]));
	    
		i3=3;
		tt = add120(ss, tt);
	        tt = rshift120(tt, l[i3]);
	        fp_set_dig(gamma,one);
                for(ii=i3-1;ii>=0;ii--){
                    shifted = rshift120(tt, kk[i3][ii]);
                    qq[ii] = shifted.low & ll[ii];  
                }

                for(j=0;j<i3;j++){
                    fp_set_dig(temp,one);
                    for(jj=0;jj<t[j]+1;jj++){
                        q[j][jj]=qq[j] & and_f;
                        qq[j]=qq[j]>>w;
                    }
                    for(jj=0;jj<t[j]+1;jj++){
                        fp_mul(temp,temp,tabl2[q[j][jj]][(jj+ep[i3][j])]);
                    }
                    for(jj=1;jj<=rho[i3][j];jj++){
                        fp_sqr(temp,temp);
                    }
                    fp_mul(gamma,gamma,temp);
                }
	        fp_mul(alpha,x[i3],gamma);
	        fp_copy(a[t[i3]],alpha);
	        for(i=1;i<t[i3]+1;i++){
                        fp_copy(a[t[i3]-i],a[t[i3]+1-i]);
		        for(j=1;j<=w;j++)
		        {
		            fp_sqr(a[t[i3]-i],a[t[i3]-i]);
		        }
	        }
	        
	        for(j=0;j<t[i3]+1;j++){
		        fp_copy(f,a[j]);
		        if(j!=0){
		            fp_copy(temp,tabl2[q[i3][0]][s[i3]+t[i3]-j]);
		            for(i=1;i<=r[i3];i++)
		            {
		                fp_sqr(temp,temp);
		            }
		            fp_mul(f,f,temp);
	                }
		        for(i=1;i<j;i++)
		        {
		            fp_mul(f,f,tabl2[q[i3][i]][nw-(j+1-i)]);
		        }
		        if(j==0){
                            fp_prime_back(tmp, f);
                            bn_get_dig(&d, tmp);
                            d=d&0x7f;
                            q[i3][j]=tab3[d]>>(w-l[k3-1]+t[k3-1]*w);
                        }
                        if(j!=0){
                            fp_prime_back(tmp, f);
                            bn_get_dig(&d, tmp);
                            d=d&0x7f;
                            q[i3][j]=tab3[d];
                        }
	        }

	        Qacc = make120(0, q[i3][0]);
	        for (i = 1; i < t[i3] + 1; i++) {
	            term = lshift120(make120(0, q[i3][i]), (l[i3] - (t[i3]+1-i) * w));
	            Qacc = add120(Qacc, term);
	        }
	        ss = lshift120(Qacc, (n - l[i3]));
	    
		tt = add120(ss, tt);
		fp_set_dig(gamma,one);
                for(ii=k3-1;ii>=0;ii--){
                    shifted = rshift120(tt, kk[k3][ii]);
                    qq[ii] = shifted.low & ll[ii];  
                }

                for(j=0;j<k3;j++){
                    fp_set_dig(temp,one);
                    for(jj=0;jj<t[j]+1;jj++){
                        q[j][jj]=qq[j] & and_f;
                        qq[j]=qq[j]>>w;
                    }
                    for(jj=0;jj<t[j]+1;jj++){
                        fp_mul(temp,temp,tabl2[q[j][jj]][(jj+ep[k3][j])]);
                    }
                    for(jj=1;jj<=rho[k3][j];jj++){
                        fp_sqr(temp,temp);
                    }
                    fp_mul(gamma,gamma,temp);
                }
		fp_mul(y,v1,gamma);
		//fp_sqr(y,y);
		//fp_print(y);
		//fp_print(u);
	}

	RLC_CATCH_ANY {
		RLC_THROW(ERR_CAUGHT);
	}
	RLC_FINALLY {
		
		fp_free(v);
		fp_free(v1);
		fp_free(f);
		for(i=0;i<t_max+1;i++){
		        fp_free(a[i]);
	        }
		fp_free(gamma);
		fp_free(alpha);
		fp_free(temp);
	}

}








int main(void) {   
    if (core_init() != RLC_OK) {
          core_clean();
          return 1;
    }

    /* Initialize pairing-friendly curve parameters */
    if (fp_param_set_any_pmers() != RLC_OK) {
        printf("Curve initialization failed\n");
        core_clean();
        return 1;
    }

    printf("Using Primes of NIST-224\n");

    fp_t tabl2[we][nw],g,z,b,h,h1,h2,y;
    uint64_t q[k3][t_max+1],qq[k3];
    int tab3[128]={0};
    bn_t tmp,m,e;

    
    for(i=0;i<we;i++){
        for(j=0;j<nw;j++){
            fp_null(tabl2[i][j]);
            fp_new(tabl2[i][j]);
        }
    }
    
    tab3[ 0x1 ]= 0 ;
    tab3[ 0x55 ]= 1 ;
    tab3[ 0x78 ]= 2 ;
    tab3[ 0x49 ]= 3 ;
    tab3[ 0x68 ]= 4 ;
    tab3[ 0x51 ]= 5 ;
    tab3[ 0x44 ]= 6 ;
    tab3[ 0x52 ]= 7 ;
    tab3[ 0x0 ]= 8 ;
    tab3[ 0x2c ]= 9 ;
    tab3[ 0x9 ]= 10 ;
    tab3[ 0x38 ]= 11 ;
    tab3[ 0x19 ]= 12 ;
    tab3[ 0x30 ]= 13 ;
    tab3[ 0x3d ]= 14 ;
    tab3[ 0x2f ]= 15 ;


    fp_null(b);
    fp_new(b);
    
    fp_null(y);
    fp_new(y);
    
    fp_null(g);
    fp_new(g);
    
    fp_null(z);
    fp_new(z);

    fp_null(h);
    fp_new(h);

    fp_null(h1);
    fp_new(h1);
    
    fp_null(h2);
    fp_new(h2);
    
    bn_null(tmp);
    bn_new(tmp);
    
    bn_null(m);
    bn_new(m);
    
    bn_null(e);
    bn_new(e);
    
    fp_rand(b);
    while(fp_is_sqr(b)!=1){
        fp_rand(b);
    }
    fp_print(b);
    bn_read_str(tmp,"b",1,16);
    fp_prime_conv(z,tmp);
    
    bn_read_str(m,"ffffffffffffffffffffffffffffffff",32,16);
    fp_exp(g,z,m);
    
    bn_read_str(e,"7fffffffffffffffffffffffffffffff",32,16);
    


    precomputation(g,tabl2);
    printf("using tonelli-shank with look up tablle\n");
    printf("W=%d\nK=%d\n",w,k3);

    
    MEASURE(findsqroot(b,y,tabl2,tab3,q,qq,e,tmp);) 
    
    printf("RDTSC_clk_min=%f\n",RDTSC_clk_min);
    printf("RDTSC_clk_median=%f\n",RDTSC_clk_median);
    printf("RDTSC_clk_max=%f\n",RDTSC_clk_max);


    for(i=0;i<we;i++){
        for(j=0;j<nw;j++){
            fp_free(tabl2[i][j]);
        }
    }

    fp_free(b);
    fp_free(y);
    bn_free(tmp);
    bn_free(e);
    bn_free(m);
    fp_free(h1);
    
    core_clean();
    return 0;
}
