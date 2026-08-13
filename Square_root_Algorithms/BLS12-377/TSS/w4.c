#include <stdio.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include<immintrin.h>
#include <relic/relic_core.h>
#include "relic/relic_md.h"


#define w 4
#define we 16
#define wf 8
#define wff 4
#define k3 2
#define n 46
#define nw 12


static int i,j,ii,jj;
int l[2]={22,23};
uint32_t ll[2]={4194303,8388607};
int kk[3][3]={{0, 0, 0}, {1, 0, 0}, {1, 23, 0}};
int s[2]={6, 5};
int t[2]={5, 5};
int t_max=5;
int r[2]={0, 3};
int ep[3][3]={{0, 0, 0}, {0, 0, 0}, {0, 5, 0}};
int rho[3][3]={{0, 0, 0}, {1, 0, 0}, {0, 2, 0}};



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




void precomputation(fp_t g,fp_t h,fp_t h1,fp_t h2,fp_t tabl1[we][t_max+1],fp_t tabl2[we][nw]){
    bn_t temp,one,itemp;
    bn_null(temp);
    bn_new(temp);
    bn_null(one);
    bn_new(one);
    bn_set_dig(one,1);
    bn_null(itemp);
    bn_new(itemp);
    //printf("step1\n");
    for(int vv=0;vv<we;vv++){
            for(int i=2;i<t_max+1;i++){
                bn_set_dig(itemp,vv);
                bn_lsh(temp,one,n-i*w);
                bn_mul(temp,temp,itemp);
                fp_exp(tabl1[vv][i],g,temp);
            }
    }
    for(int vv=0;vv<we;vv++){
        for(int i=0;i<nw;i++){
            bn_set_dig(itemp,vv);
            bn_lsh(temp,one,i*w);
            bn_mul(temp,temp,itemp);
            fp_exp(tabl2[vv][i],g,temp);
        }
    }


}





void findsqroot(fp_t u, fp_t y,fp_t tabl1[we][t_max+1],fp_t tabl2[we][nw],int tab3[64],dig_t q[k3][t_max+1],dig_t qq[k3],bn_t e,bn_t tmp) {
	fp_t v,v1,x[k3],gamma,alpha;
	dig_t s,tt;
	
	dig_t and_f=15;
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
	fp_null(v);
	fp_null(v1);
	fp_null(gamma);
	fp_null(alpha);
	fp_null(temp);
        fp_null(temp1);

	RLC_TRY {
		for(int i=0;i<k3;i++){
	            fp_new(x[i]);
	        }
		fp_new(xx);
		fp_new(v);
		fp_new(v1);
		fp_new(gamma);
		fp_new(alpha);
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
		s=0;
		tt=0;
		int uu=0;
	        fp_set_dig(gamma,1);
	        fp_mul(alpha,x[uu],gamma);
	        
	        fp_copy(a[t[uu]],alpha);
		for(i=1;i<t[uu]+1;i++){
	                fp_copy(a[t[uu]-i],a[t[uu]+1-i]);
			for(j=1;j<=w;j++)
			{
			    fp_sqr(a[t[uu]-i],a[t[uu]-i]);
			}
		}
			
		for(j=0;j<t[uu]+1;j++){
			fp_copy(f,a[j]);
			if(j!=0){
			    fp_copy(temp,tabl2[q[uu][0]][11-j]);
			    for(i=1;i<=r[uu];i++)
			    {
			        fp_sqr(temp,temp);
			    }
			    fp_mul(f,f,temp);
		        }
			for(i=1;i<j;i++)
			{
			    fp_mul(f,f,tabl1[q[uu][i]][j+1-i]);
			}
			//fp_print(f);
			if(j==0){
				fp_prime_back(tmp, f);
	                        dig_t d;
	                        bn_get_dig(&d, tmp);
	                        d=d&0x3f;
	                        q[uu][j]=tab3[d]>>(w-l[0]+t[0]*w);
			}
			if(j!=0){
				fp_prime_back(tmp, f);
	                        bn_get_dig(&d, tmp);
	                        d=d&0x3f;
	                        q[uu][j]=tab3[d];
			}
            	    
	                
		}
                Q=q[uu][0];
                for(i=1;i<t[uu]+1;i++){
                    temp2= q[uu][i]<<(l[uu]-(t[uu]+1-i)*w);
                    Q=Q+temp2;
                }
                s=Q<<(n-l[uu]);
		
		
		uu=k3-1;
		tt=s+tt;
	        tt=tt>>l[uu];
                fp_set_dig(gamma,one);
                for(ii=uu-1;ii>=0;ii--){
                    //printf("%d",ii);
                    qq[ii]=(tt>>kk[uu][ii])& ll[ii];
                }
                
                for(j=0;j<uu;j++){ 
                    fp_set_dig(temp,one);
                    for(jj=0;jj<t[j]+1;jj++){
                        q[j][jj]=qq[j] & and_f;
                        qq[j]=qq[j]>>w;
                    }
                    for(jj=0;jj<t[j]+1;jj++){
                        fp_mul(temp,temp,tabl2[q[j][jj]][(jj+ep[uu][j])]);
                    }
                    for(jj=1;jj<=rho[uu][j];jj++){
                        fp_sqr(temp,temp);
                    }
                    fp_mul(gamma,gamma,temp);
                }
	        
	        fp_mul(alpha,x[uu],gamma);
	        
	        fp_copy(a[t[uu]],alpha);
		for(i=1;i<t[uu]+1;i++){
	                fp_copy(a[t[uu]-i],a[t[uu]+1-i]);
			for(j=1;j<=w;j++)
			{
			    fp_sqr(a[t[uu]-i],a[t[uu]-i]);
			}
		}
		
		for(j=0;j<t[uu]+1;j++){
			fp_copy(f,a[j]);
			if(j!=0){
			    fp_copy(temp,tabl2[q[uu][0]][10-j]);
			    for(i=1;i<=r[uu];i++)
			    {
			        fp_sqr(temp,temp);
			    }
			    fp_mul(f,f,temp);
		        }
			for(i=1;i<j;i++)
			{
			    fp_mul(f,f,tabl1[q[uu][i]][j+1-i]);
			}
			
			if(j==0){
				fp_prime_back(tmp, f);
	                        bn_get_dig(&d, tmp);
	                        d=d&0x3f;
	                        q[uu][j]=tab3[d]>>(w-l[1]+t[1]*w);	
			}
			if(j!=0){
			        fp_prime_back(tmp, f);
	                        bn_get_dig(&d, tmp);
	                        d=d&0x3f;
	                        q[uu][j]=tab3[d];
			}
		}
                Q=q[uu][0];
                for(i=1;i<t[uu]+1;i++){
                    temp2= q[uu][i]<<(l[uu]-(t[uu]+1-i)*w);
                    Q=Q+temp2;
                }
                s=Q<<(n-l[uu]);
		
		
		
		tt=s+tt;
		
		fp_set_dig(gamma,one);
                for(ii=k3-1;ii>=0;ii--){
                    //printf("%d",ii);
                    qq[ii]=(tt>>kk[k3][ii])& ll[ii];
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
		for(int i=0;i<k3;i++){
	            fp_free(x[i]);
	        }
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

    printf("Using Primes of bls12-377\n");

    fp_t tabl1[we][t_max+1],tabl2[we][nw],g,z,b,h,h1,h2,y;
    dig_t q[k3][t_max+1],qq[k3];
    int tab3[64]={0},tab4[4]={0},tab6[32]={0};
    bn_t tmp,m,e;
    for(i=0;i<we;i++){
        for(j=0;j<t_max+1;j++){
            fp_null(tabl1[i][j]);
            fp_new(tabl1[i][j]);
        }
    }
    
    for(i=0;i<we;i++){
        for(j=0;j<nw;j++){
            fp_null(tabl2[i][j]);
            fp_new(tabl2[i][j]);
        }
    }
    tab3[ 0x1 ]= 0 ;
    tab3[ 0x24 ]= 1 ;
    tab3[ 0x11 ]= 2 ;
    tab3[ 0x10 ]= 3 ;
    tab3[ 0x1e ]= 4 ;
    tab3[ 0x3e ]= 5 ;
    tab3[ 0x7 ]= 6 ;
    tab3[ 0xa ]= 7 ;
    tab3[ 0x0 ]= 8 ;
    tab3[ 0x1d ]= 9 ;
    tab3[ 0x30 ]= 10 ;
    tab3[ 0x31 ]= 11 ;
    tab3[ 0x23 ]= 12 ;
    tab3[ 0x3 ]= 13 ;
    tab3[ 0x3a ]= 14 ;
    tab3[ 0x37 ]= 15 ;   


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
//    bn_read_str(tmp,"4",1,16);
//    fp_prime_conv(b,tmp);
    
    bn_read_str(tmp,"5",1,16);
    fp_prime_conv(z,tmp);
    
    bn_read_str(m,"6b8e9185f1443ab18ec1701b28524ec688b67cc03d44e3c7bcd88bee82520005c2d7510c00000021423",96,16);
    fp_exp(g,z,m);
    
    bn_read_str(e,"35c748c2f8a21d58c760b80d94292763445b3e601ea271e3de6c45f741290002e16ba88600000010a11",83,16);
    
    //h=g^(2^(n-w))
    bn_t a1;
    bn_null(a1);
    bn_new(a1);
    bn_read_str(a1,"40000000000",11,16);
    fp_exp(h,g,a1);
    //h1=g^(2^(n-2))
    bn_read_str(a1,"100000000000",12,16);
    fp_exp(h1,g,a1);
    //h1=g^(2^(n-3))
    bn_read_str(a1,"80000000000",11,16);
    fp_exp(h2,g,a1);
    precomputation(g,h,h1,h2,tabl1,tabl2);
    printf("using tonelli-shank with look up tablle\n");

    
    MEASURE(findsqroot(b,y,tabl1,tabl2,tab3,q,qq,e,tmp);)
    
    
    printf("RDTSC_clk_min=%f\n",RDTSC_clk_min);
    printf("RDTSC_clk_median=%f\n",RDTSC_clk_median);
    printf("RDTSC_clk_max=%f\n",RDTSC_clk_max);

    for(i=0;i<we;i++){
        for(j=0;j<t_max+1;j++){
            fp_free(tabl1[i][j]);
        }
    }
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
    fp_free(h2);
    
    core_clean();
    return 0;
}

