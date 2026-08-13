#include <stdio.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include<immintrin.h>
#include <relic/relic_core.h>
#include "relic/relic_md.h"


#define w 6
#define we 64
#define wf 8
#define wff 8
#define k3 1
#define n 46
#define nw 8


static int i,j,ii,jj;
int l[1]={45};
uint64_t ll[1]={0x1fffffffffff};
int kk[2][2]={{0, 0}, {1, 0}};
int s[1]={0};
int t[1]={7};
int t_max=7;
int r[1]={1};
int ep[2][2]={{0,0},{0,0}};
int rho[2][2]={{0,0},{0,0}};



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






void findsqroot(fp_t u, fp_t y,fp_t tabl1[we][t_max+1],fp_t tabl2[we][nw],int tab3[4096],dig_t q[k3][t_max+1],dig_t qq[k3],bn_t e,bn_t tmp) {
	fp_t v,v1,x[k3],gamma,alpha;
	dig_t d,s,tt;
	dig_t and_f=63;
        dig_t c1,c,one=1;
	
	fp_t f,a[t[0]+1],temp,temp1;
	fp_null(f);
	dig_t Q,temp2;

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
		fp_new(v);
		fp_new(v1);
		fp_new(gamma);
		fp_new(alpha);
		fp_new(temp);
	        fp_new(temp1);
		

		fp_exp(v, u, e);
		fp_mul(v1,v,u);
		fp_mul(x[0],v,v1);

		s=0;
		tt=0;
		int i3=0;
	        fp_set_dig(gamma,1);
	        fp_mul(alpha,x[0],gamma);
	        
	        
	        fp_copy(a[t[0]],alpha);
	        for(i=1;i<t[0]+1;i++){
                        fp_copy(a[t[0]-i],a[t[0]+1-i]);
		        for(j=1;j<=w;j++)
		        {
		            fp_sqr(a[t[0]-i],a[t[0]-i]);
		        }
	        }
	        
	        for(j=0;j<t[0]+1;j++){
			fp_copy(f,a[j]);
			if(j!=0){
			    fp_copy(temp,tabl2[q[0][0]][7-j]);
			    for(i=1;i<=r[0];i++)
			    {
			        fp_sqr(temp,temp);
			    }
			    fp_mul(f,f,temp);
		        }
			for(i=1;i<j;i++)
			{
			    fp_mul(f,f,tabl1[q[0][i]][j+1-i]);
			}
			if(j==0){
			        fp_prime_back(tmp, f);
	                        bn_get_dig(&d, tmp);
	                        d=d&0xfff;
	                        q[0][j]=tab3[d]>>(w-l[0]+t[0]*w);
			}
			if(j!=0){
				fp_prime_back(tmp, f);
	                        bn_get_dig(&d, tmp);
	                        d=d&0xfff;
	                        q[0][j]=tab3[d];
			}
            	    
	                
		}

	        Q=q[0][0];
                for(i=1;i<8;i++){
                    temp2= q[0][i]<<(45-(8-i)*w);
                    Q=Q+temp2;
                }
                s=Q<<(n-l[0]);  
	
          	tt=s;
          	
          	fp_set_dig(gamma,1);
                
                qq[0]=(tt>>kk[1][0])& ll[0];
                
                
                for(j=0;j<1;j++){ 
                    fp_set_dig(temp,1);
                    for(jj=0;jj<t[j]+1;jj++){
                        q[j][jj]=qq[j] & and_f;
                        qq[j]=qq[j]>>w;
                    }
                    for(jj=0;jj<t[j]+1;jj++){
                        fp_mul(temp,temp,tabl2[q[j][jj]][(jj+ep[1][j])]);
                    }
                    for(jj=1;jj<=rho[1][j];jj++){
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
    if (ep_param_set_any_pairf() != RLC_OK) {
        printf("Curve initialization failed\n");
        core_clean();
        return 1;
    }

    printf("Using Primes of bls12-377\n");

    fp_t tabl1[we][t_max+1],tabl2[we][nw],tabl4[wff],tabl6[wf],g,z,b,h,h1,h2,y;
    dig_t q[k3][t_max+1],qq[k3];
    int tab3[4096]={0};
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
tab3[ 0x71f ]= 1 ;
tab3[ 0xd1b ]= 2 ;
tab3[ 0x1ac ]= 3 ;
tab3[ 0xea4 ]= 4 ;
tab3[ 0xc00 ]= 5 ;
tab3[ 0xe56 ]= 6 ;
tab3[ 0x52f ]= 7 ;
tab3[ 0xf11 ]= 8 ;
tab3[ 0xd ]= 9 ;
tab3[ 0x247 ]= 10 ;
tab3[ 0x61d ]= 11 ;
tab3[ 0xd50 ]= 12 ;
tab3[ 0xe4 ]= 13 ;
tab3[ 0xbd1 ]= 14 ;
tab3[ 0x685 ]= 15 ;
tab3[ 0x39e ]= 16 ;
tab3[ 0x9f3 ]= 17 ;
tab3[ 0x19b ]= 18 ;
tab3[ 0x476 ]= 19 ;
tab3[ 0xebe ]= 20 ;
tab3[ 0xa8c ]= 21 ;
tab3[ 0xf14 ]= 22 ;
tab3[ 0x9e2 ]= 23 ;
tab3[ 0xa47 ]= 24 ;
tab3[ 0x49a ]= 25 ;
tab3[ 0xad5 ]= 26 ;
tab3[ 0x28b ]= 27 ;
tab3[ 0xdca ]= 28 ;
tab3[ 0xca8 ]= 29 ;
tab3[ 0x14f ]= 30 ;
tab3[ 0x75b ]= 31 ;
tab3[ 0x0 ]= 32 ;
tab3[ 0x8e2 ]= 33 ;
tab3[ 0x2e6 ]= 34 ;
tab3[ 0xe55 ]= 35 ;
tab3[ 0x15d ]= 36 ;
tab3[ 0x401 ]= 37 ;
tab3[ 0x1ab ]= 38 ;
tab3[ 0xad2 ]= 39 ;
tab3[ 0xf0 ]= 40 ;
tab3[ 0xff4 ]= 41 ;
tab3[ 0xdba ]= 42 ;
tab3[ 0x9e4 ]= 43 ;
tab3[ 0x2b1 ]= 44 ;
tab3[ 0xf1d ]= 45 ;
tab3[ 0x430 ]= 46 ;
tab3[ 0x97c ]= 47 ;
tab3[ 0xc63 ]= 48 ;
tab3[ 0x60e ]= 49 ;
tab3[ 0xe66 ]= 50 ;
tab3[ 0xb8b ]= 51 ;
tab3[ 0x143 ]= 52 ;
tab3[ 0x575 ]= 53 ;
tab3[ 0xed ]= 54 ;
tab3[ 0x61f ]= 55 ;
tab3[ 0x5ba ]= 56 ;
tab3[ 0xb67 ]= 57 ;
tab3[ 0x52c ]= 58 ;
tab3[ 0xd76 ]= 59 ;
tab3[ 0x237 ]= 60 ;
tab3[ 0x359 ]= 61 ;
tab3[ 0xeb2 ]= 62 ;
tab3[ 0x8a6 ]= 63 ;
    


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
//    bn_read_str(tmp,"10",2,16);
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
    bn_read_str(a1,"10000000000",11,16);
    fp_exp(h,g,a1);
    //h1=g^(2^(n-3))
    bn_read_str(a1,"80000000000",11,16);
    fp_exp(h1,g,a1);
    //h1=g^(2^(n-5))
    bn_read_str(a1,"200000000000",11,16);
    fp_exp(h2,g,a1);
    precomputation(g,h,h1,h2,tabl1,tabl2);
    printf("using tonelli-shank with look up table\n");

    
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

