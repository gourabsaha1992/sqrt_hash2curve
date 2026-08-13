#include <stdio.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include<immintrin.h>
#include <relic/relic_core.h>
#include "relic/relic_md.h"

#define w 6
#define we 64
#define n 96
#define k3 2
#define nw 16
#define count 1 

int i,j,ii,jj,i3;
int l[k3]= {47, 48} ;
uint64_t ll[k3]={
0x7fffffffffff ,
0xffffffffffff };
int kk[k3+1][k3+1]= {{0, 0, 0}, {1, 0, 0}, {1, 48, 0}} ;
int s[k3]= {8, 8} ;
int t[k3]= {7, 7} ;
int t_max= 7 ;
int r[k3]= {1, 0} ;
int ep[k3+1][k3+1]= {{0, 0, 0}, {0, 0, 0}, {0, 7, 0}} ;
int rho[k3+1][k3+1]= {{0, 0, 0}, {1, 0, 0}, {0, 5, 0}} ;

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
uint64_t and_f=63;
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




void precomputation(fp_t g,fp_t h,fp_t h1,fp_t tabl2[we][nw]){
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


void findsqroot(fp_t u, fp_t y,fp_t tabl2[we][nw],int tab3[1024],uint64_t q[k3][t_max+1],uint64_t qq[k3],bn_t e, bn_t tmp) {
	fp_t v,v1,x[k3],gamma,alpha;
	bn_t s,tt;
	fp_t temp,temp1;
	fp_t f,a[t_max+1];
	fp_null(f);
	
	for(i=0;i<t_max+1;i++){
		fp_null(a[i]);
		fp_new(a[i]);
	}
    
 
        fp_null(temp);
        fp_null(temp1);
     
        
	
	
	fp_null(v);
	fp_null(v1);
	fp_null(gamma);
	fp_null(alpha);
	bn_null(s);
	bn_null(tt);


	RLC_TRY {
		
		
		fp_new(v);
		fp_new(v1);
		fp_new(gamma);
		fp_new(alpha);
		bn_new(s);
		bn_new(tt);
		fp_new(temp);
                fp_new(temp1);

		
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
		bn_set_dig(s,0);
		bn_set_dig(tt,0);
		i3=0;
	        fp_set_dig(gamma,1);
	        fp_copy(a[t[i3]],x[i3]);
		for(i=1;i<t[i3]+1;i++){
	                fp_copy(a[t[i3]-i],a[t[i3]+1-i]);
			for(j=1;j<=w;j++)
			{
			    fp_sqr(a[t[i3]-i],a[t[i3]-i]);
			}
		}
			
		for(j=0;j<t[i3]+1;j++){
			fp_copy(f,a[j]);
			//fp_print(f);
			if(j!=0){
			    fp_copy(temp,tabl2[q[i3][0]][15-j]);
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
	                    d=d&0x3ff;
	                    q[i3][j]=tab3[d]>>(w-l[0]+t[0]*w);
                        }
                        if(j!=0){
                            fp_prime_back(tmp, f);
	                    bn_get_dig(&d, tmp);
	                    d=d&0x3ff;
	                    q[i3][j]=tab3[d];
	                }
		}
                bn_set_dig(Q,q[i3][0]);
                for(i=1;i<t[i3]+1;i++){
                    bn_set_dig(temp2,q[i3][i]);
                    bn_lsh(temp2,temp2,(l[i3]-(t[i3]+1-i)*w));
                    bn_add(Q,Q,temp2);
                }
                bn_lsh(s,Q,(n-l[i3]));
	        
	        
		i3=1;
	        bn_add(tt,s,tt);
	        bn_rsh(tt,tt,l[i3]);
                for(ii=i3-1;ii>=0;ii--){
                    bn_rsh(tmp,tt,kk[i3][ii]);
                    bn_get_dig(&d, tmp);
                    qq[ii] =d & ll[ii];
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
	        //power(gamma,tt,i3,tabl2,q,qq,tmp);
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
		            fp_copy(temp,tabl2[q[i3][0]][15-j]);
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
                            d=d&0x3ff;
                            q[i3][j]=tab3[d]>>(w-l[k3-1]+t[k3-1]*w);
                        }
                        if(j!=0){
                            fp_prime_back(tmp, f);
                            bn_get_dig(&d, tmp);
                            d=d&0x3ff;
                            q[i3][j]=tab3[d];
                        }
	        }
                bn_set_dig(Q,q[i3][0]);
                for(i=1;i<t[i3]+1;i++){
                    bn_set_dig(temp2,q[i3][i]);
                    bn_lsh(temp2,temp2,(l[i3]-(t[i3]+1-i)*w));
                    bn_add(Q,Q,temp2);
                }
                bn_lsh(s,Q,(n-l[i3]));
		
		bn_add(tt,s,tt);
		fp_set_dig(gamma,one);
                for(ii=k3-1;ii>=0;ii--){
                    bn_rsh(tmp,tt,kk[k3][ii]);
                    bn_get_dig(&d, tmp);
                    qq[ii] =d & ll[ii];
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
		fp_free(gamma);
		fp_free(alpha);
		bn_free(s);
		bn_free(tt);
		fp_free(temp);
                fp_free(temp1);
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
    int tab3[1024]={0};
    bn_t tmp,m,e;
    
    
    for(i=0;i<we;i++){
        for(j=0;j<nw;j++){
            fp_null(tabl2[i][j]);
            fp_new(tabl2[i][j]);
        }
    }
    
    tab3[ 0x1 ]= 0;
    tab3[ 0x32e ]= 1;
    tab3[ 0x100 ]= 2;
    tab3[ 0x18f ]= 3;
    tab3[ 0x355 ]= 4;
    tab3[ 0x9c ]= 5;
    tab3[ 0x16b ]= 6;
    tab3[ 0x198 ]= 7;
    tab3[ 0x2f8 ]= 8;
    tab3[ 0x7e ]= 9;
    tab3[ 0x2f7 ]= 10;
    tab3[ 0x19e ]= 11;
    tab3[ 0x349 ]= 12;
    tab3[ 0x17a ]= 13;
    tab3[ 0x3a3 ]= 14;
    tab3[ 0x76 ]= 15;
    tab3[ 0x1e8 ]= 16;
    tab3[ 0x3ff ]= 17;
    tab3[ 0x85 ]= 18;
    tab3[ 0xca ]= 19;
    tab3[ 0x251 ]= 20;
    tab3[ 0x1df ]= 21;
    tab3[ 0x9f ]= 22;
    tab3[ 0x3fc ]= 23;
    tab3[ 0x244 ]= 24;
    tab3[ 0x13c ]= 25;
    tab3[ 0x14e ]= 26;
    tab3[ 0x15c ]= 27;
    tab3[ 0xd2 ]= 28;
    tab3[ 0x21 ]= 29;
    tab3[ 0x12e ]= 30;
    tab3[ 0xb6 ]= 31;
    tab3[ 0x0 ]= 32;
    tab3[ 0xd3 ]= 33;
    tab3[ 0x301 ]= 34;
    tab3[ 0x272 ]= 35;
    tab3[ 0xac ]= 36;
    tab3[ 0x365 ]= 37;
    tab3[ 0x296 ]= 38;
    tab3[ 0x269 ]= 39;
    tab3[ 0x109 ]= 40;
    tab3[ 0x383 ]= 41;
    tab3[ 0x10a ]= 42;
    tab3[ 0x263 ]= 43;
    tab3[ 0xb8 ]= 44;
    tab3[ 0x287 ]= 45;
    tab3[ 0x5e ]= 46;
    tab3[ 0x38b ]= 47;
    tab3[ 0x219 ]= 48;
    tab3[ 0x2 ]= 49;
    tab3[ 0x37c ]= 50;
    tab3[ 0x337 ]= 51;
    tab3[ 0x1b0 ]= 52;
    tab3[ 0x222 ]= 53;
    tab3[ 0x362 ]= 54;
    tab3[ 0x5 ]= 55;
    tab3[ 0x1bd ]= 56;
    tab3[ 0x2c5 ]= 57;
    tab3[ 0x2b3 ]= 58;
    tab3[ 0x2a5 ]= 59;
    tab3[ 0x32f ]= 60;
    tab3[ 0x3e0 ]= 61;
    tab3[ 0x2d3 ]= 62;
    tab3[ 0x34b ]= 63;


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
    
//    bn_read_str(tmp,"4",96,16);
//    fp_prime_conv(b,tmp);
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
    
    //h=g^(2^(n-w))
    bn_t a1;
    bn_new(a1);
    bn_set_dig(a1,1);
    bn_lsh(a1,a1,n-w);
    fp_exp(h,g,a1);
    //h1=g^(2^(n-3))=34
    bn_set_dig(a1,1);
    bn_lsh(a1,a1,n-l[0]+t[0]*w);
    fp_exp(h1,g,a1);
    
    bn_set_dig(a1,1);
    bn_lsh(a1,a1,n-l[0]+t[0]*w);
    fp_exp(h2,g,a1);
    precomputation(g,h,h1,tabl2);
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
//    for(i=0;i<we;i++){
 //       fp_free(tabl3[i]);
 //   }
 
    fp_free(b);
    fp_free(y);
    bn_free(tmp);
    bn_free(e);
    bn_free(m);
    bn_free(a1);
    fp_free(h1);
    
    core_clean();
    return 0;
}

