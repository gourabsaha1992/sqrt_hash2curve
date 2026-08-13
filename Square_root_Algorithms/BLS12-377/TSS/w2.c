#include <stdio.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include<immintrin.h>
#include <relic/relic_core.h>
#include "relic/relic_md.h"


#define w 2
#define we 4
#define wf 2
#define k3 4
#define n 46
#define nw 23


static int i,j,ii,jj;
int l[4]={11,11,11,12};
int ll[4]={2047,2047,2047,4095};
int kk[5][5]={{0,0,0,0,0},{24, 0, 0, 0, 0}, {13, 24, 0, 0, 0}, {1, 12, 23, 0, 0},{1, 12, 23, 34, 0}};
int s[4]={17,17,17,17};
int t[4]={5,5,5,5};
int t_max=5;
int r[4]={1,1,1,0};
int ep[5][5]={{0, 0, 0, 0, 0}, {12, 0, 0, 0, 0}, {6, 12, 0, 0, 0}, {0, 6, 11, 0, 0},{0, 5, 11, 16, 0}};
int rho[5][5]={{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {1, 0, 0, 0, 0}, {1, 0, 1, 0, 0},{0, 1, 0, 1, 0}};



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





void findsqroot(fp_t u, fp_t y,fp_t tabl2[we][nw],int tabl3[4],dig_t q[k3][t_max+1],dig_t qq[k3],bn_t e, bn_t tmp) {
	fp_t v,v1,xx,x[k3],gamma,alpha;
	dig_t s,tt;

	dig_t and_f=3;
        fp_t temp,temp1;
        dig_t d,c1,c,one=1;
        
        fp_t f,a[t_max+1];
	fp_null(f);
	dig_t Q,temp2;
	for(i=0;i<t_max+1;i++){
		fp_null(a[i]);
		fp_new(a[i]);
	}
	
	for(int i=0;i<k3;i++){
	    fp_null(x[i]);
	}
	fp_null(xx);
	fp_null(v);
	fp_null(v1);
	fp_null(gamma);
	fp_null(alpha);

	RLC_TRY {
		for(int i=0;i<k3;i++){
	            fp_new(x[i]);
	        }
		fp_new(xx);
		fp_new(v);
		fp_new(v1);
		fp_new(gamma);
		fp_new(alpha);

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
		s=0;
		tt=0;
		int i3=0;
	        fp_set_dig(gamma,1);
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
			    fp_copy(temp,tabl2[q[i3][0]][22-j]);
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
	                        d=d&0x3;
	                        q[i3][j]=tabl3[d]>>(w-l[0]+t[0]*w);	
			}
			if(j!=0){
				fp_prime_back(tmp, f);
	                        bn_get_dig(&d, tmp);
	                        d=d&0x3;
	                        q[i3][j]=tabl3[d];
			}
            	    
	                
		}
                Q=q[i3][0];
                for(i=1;i<t[i3]+1;i++){
                    temp2= q[i3][i]<<(l[i3]-(t[i3]+1-i)*w);
                    Q=Q+temp2;
                }
                s=Q<<(n-l[i3]);  
	        
	        i3=1;
	        tt=s+tt;
	        tt=tt>>l[i3];
	        
	        fp_set_dig(gamma,1);
                for(ii=i3-1;ii>=0;ii--){
                    qq[ii]=(tt>>kk[i3][ii])& ll[ii];
                }
                
                for(j=0;j<i3;j++){ 
                    fp_set_dig(temp,1);
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
			    fp_copy(temp,tabl2[q[i3][0]][22-j]);
			    for(i=1;i<=r[i3];i++)
			    {
			        fp_sqr(temp,temp);
			    }
			    fp_mul(f,f,temp);
		        }
			for(i=1;i<j;i++)
			{
			    fp_mul(f,f,tabl2[q[i3][i]][23-(j+1-i)]);
			}
			if(j==0){
				fp_prime_back(tmp, f);
	                        bn_get_dig(&d, tmp);
	                        d=d&0x3;
	                        q[i3][j]=tabl3[d]>>(w-l[0]+t[0]*w);	
			}
			if(j!=0){
				fp_prime_back(tmp, f);
	                        bn_get_dig(&d, tmp);
	                        d=d&0x3;
	                        q[i3][j]=tabl3[d];
			}
            	    
	                
		}
                Q=q[i3][0];
                for(i=1;i<t[i3]+1;i++){
                    temp2= q[i3][i]<<(l[i3]-(t[i3]+1-i)*w);
                    Q=Q+temp2;
                }
                s=Q<<(n-l[i3]);
	        
	        i3=2;
	        tt=s+tt;
	        tt=tt>>l[i3];
	        fp_set_dig(gamma,1);
                for(ii=i3-1;ii>=0;ii--){
                    qq[ii]=(tt>>kk[i3][ii])& ll[ii];
                }
                
                for(j=0;j<i3;j++){ 
                    fp_set_dig(temp,1);
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
			    fp_copy(temp,tabl2[q[i3][0]][22-j]);
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
	                        d=d&0x3;
	                        q[i3][j]=tabl3[d]>>(w-l[0]+t[0]*w);	
			}
			if(j!=0){
				fp_prime_back(tmp, f);
	                        bn_get_dig(&d, tmp);
	                        d=d&0x3;
	                        q[i3][j]=tabl3[d];
			}
            	    
	                
		}
                Q=q[i3][0];
                for(i=1;i<t[i3]+1;i++){
                    temp2= q[i3][i]<<(l[i3]-(t[i3]+1-i)*w);
                    Q=Q+temp2;
                }
                s=Q<<(n-l[i3]);  
	        
	        
	        
		
		
		i3=3;
		tt=s+tt;
	        tt=tt>>l[i3];
		fp_set_dig(gamma,1);
                for(ii=i3-1;ii>=0;ii--){
                    qq[ii]=(tt>>kk[i3][ii])& ll[ii];
                }
                
                for(j=0;j<i3;j++){ 
                    fp_set_dig(temp,1);
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
			    fp_copy(temp,tabl2[q[i3][0]][22-j]);
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
	                        d=d&0x3;
	                        q[i3][j]=tabl3[d]>>(w-l[k3-1]+t[k3-1]*w);	
			}
			if(j!=0){
				fp_prime_back(tmp, f);
	                        bn_get_dig(&d, tmp);
	                        d=d&0x3;
	                        q[i3][j]=tabl3[d];
			}
            	    
	                
		}
                Q=q[i3][0];
                for(i=1;i<t[i3]+1;i++){
                    temp2= q[i3][i]<<(l[i3]-(t[i3]+1-i)*w);
                    Q=Q+temp2;
                }
                s=Q<<(n-l[i3]);  
		
		
		
		tt=s+tt;
		
		fp_set_dig(gamma,1);
                for(ii=k3-1;ii>=0;ii--){
                    qq[ii]=(tt>>kk[k3][ii])& ll[ii];
                }
                
                for(j=0;j<k3;j++){ 
                    fp_set_dig(temp,1);
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
		for(int i=0;i<k3;i++){
	            fp_free(x[i]);
	        }
		fp_free(xx);
		fp_free(v);
		fp_free(v1);
		fp_free(gamma);
		fp_free(alpha);
	}

}








int main(void) {   
    if (core_init() != RLC_OK) {
          core_clean();
          return 1;
    }

    /* Initialize pairing-friendly curve parameters */
    if (ep_param_set_any_pairf() != RLC_OK) {
        printf("Curve initialization failed\n");
        core_clean();
        return 1;
    }

    printf("Using Primes of bls24-509\n");

    fp_t tabl2[we][nw],g,z,b,h,h1,y;
    dig_t q[k3][t_max+1],qq[k3];
    int tabl3[4]={0};
    bn_t tmp,m,e;

    
    for(i=0;i<we;i++){
        for(j=0;j<nw;j++){
            fp_null(tabl2[i][j]);
            fp_new(tabl2[i][j]);
        }
    }

    tabl3[ 0x1 ]= 0 ;
    tabl3[ 0x2 ]= 1 ;
    tabl3[ 0x0 ]= 2 ;
    tabl3[ 0x3 ]= 3 ;
    
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
//    bn_read_str(tmp,"4",1,16);
//    fp_prime_conv(b,tmp);
    
    bn_read_str(tmp,"5",1,16);
    fp_prime_conv(z,tmp);
    
    bn_read_str(m,"6b8e9185f1443ab18ec1701b28524ec688b67cc03d44e3c7bcd88bee82520005c2d7510c00000021423",96,16);
    fp_exp(g,z,m);
    
    bn_read_str(e,"35c748c2f8a21d58c760b80d94292763445b3e601ea271e3de6c45f741290002e16ba88600000010a11",96,16);
    
    //h=g^(2^(n-2))
    bn_t a1;
    bn_null(a1);
    bn_new(a1);
    bn_read_str(a1,"100000000000",12,16);
    fp_exp(h,g,a1);
    //h1=g^(2^(43))
    bn_read_str(a1,"200000000000",12,16);
    fp_exp(h1,g,a1);
    precomputation(g,h,h1,tabl2);
    printf("using tonelli-shank with look up tablle\n");

    
    MEASURE(findsqroot(b,y,tabl2,tabl3,q,qq,e,tmp);)
    
    
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
    bn_free(a1);
    fp_free(g);
    fp_free(z);
    fp_free(h);
    fp_free(h1);
    
    core_clean();
    return 0;
}

