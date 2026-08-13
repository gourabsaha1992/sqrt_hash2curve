#include <stdio.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include<immintrin.h>
#include <relic/relic_core.h>
#include "relic/relic_md.h"

#define w 8
#define we 256
#define n 192
#define k3 3
#define nw 24
#define count 1 

int i,j,ii,jj,i3;
int l[k3]= {63, 64, 64} ;
uint64_t ll[k3]={
0x7fffffffffffffff ,
0xffffffffffffffff ,
0xffffffffffffffff };
int kk[k3+1][k3+1]= {{0, 0, 0, 0}, {65, 0, 0, 0}, {1, 64, 0, 0}, {1, 64, 128, 0}} ;
int s[k3]= {16, 16, 16} ;
int t[k3]= {7, 7, 7} ;
int t_max= 7 ;
int r[k3]= {1, 0, 0} ;
int ep[k3+1][k3+1]= {{0, 0, 0, 0}, {8, 0, 0, 0}, {0, 8, 0, 0}, {0, 7, 15, 0}} ;
int rho[k3+1][k3+1]= {{0, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {0, 7, 7, 0}} ;


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
uint64_t and_f=255;
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


void findsqroot(fp_t u, fp_t y,fp_t tabl2[we][nw],int tab3[16384],uint64_t q[k3][t_max+1],uint64_t qq[k3],bn_t e, bn_t tmp) {
	fp_t v,v1,x[k3],gamma,alpha,temp,f,a[t_max+1];
	bn_t ss,tt;

	
	
	fp_null(v);
	fp_null(temp);
	fp_null(v1);
	fp_null(f);
	fp_null(gamma);
	fp_null(alpha);
	bn_null(ss);
	bn_null(tt);
	for(i=0;i<t_max+1;i++){
	    fp_null(a[i]);
	}
	
	

	RLC_TRY {
		
		
		fp_new(v);
		fp_new(temp);
		fp_new(f);
		fp_new(v1);
		fp_new(gamma);
		fp_new(alpha);
		bn_new(ss);
		bn_new(tt);
		for(i=0;i<t_max+1;i++){
	            fp_new(a[i]);
	        }

		
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
		bn_set_dig(ss,0);
		bn_set_dig(tt,0);
		for(i3=0;i3<count;i3++)
		{
		    bn_add(tt,ss,tt);
		    bn_rsh(tt,tt,l[i3]);
		    fp_set_dig(gamma,one);
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
	                        d=d&0x3fff;
	                        q[i3][j]=tab3[d]>>(w-l[0]+t[0]*w);
                                }
                            if(j!=0){
                                fp_prime_back(tmp, f);
                                bn_get_dig(&d, tmp);
                                d=d&0x3fff;
                                q[i3][j]=tab3[d];
                                }
			}
                        bn_set_dig(Q,q[i3][0]);
                        for(i=1;i<t[i3]+1;i++){
                            bn_set_dig(temp2,q[i3][i]);
                            bn_lsh(temp2,temp2,(l[i3]-(t[i3]+1-i)*w));
                            bn_add(Q,Q,temp2);
                        }
                        bn_lsh(ss,Q,(n-l[i3]));
		}
		for(i3=count;i3<k3;i3++)
		{
		    bn_add(tt,ss,tt);
		    bn_rsh(tt,tt,l[i3]);
		    fp_set_dig(gamma,one);
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
	                        d=d&0x3fff;
	                        q[i3][j]=tab3[d]>>(w-l[k3-1]+t[k3-1]*w);
                                }
                            if(j!=0){
                                fp_prime_back(tmp, f);
                                bn_get_dig(&d, tmp);
                                d=d&0x3fff;
                                q[i3][j]=tab3[d];
                                }
			}
                        bn_set_dig(Q,q[i3][0]);
                        for(i=1;i<t[i3]+1;i++){
                            bn_set_dig(temp2,q[i3][i]);
                            bn_lsh(temp2,temp2,(l[i3]-(t[i3]+1-i)*w));
                            bn_add(Q,Q,temp2);
                        }
                        bn_lsh(ss,Q,(n-l[i3]));
		}
		bn_add(tt,ss,tt);
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
		fp_free(temp);
		fp_free(v1);
		fp_free(f);
		fp_free(gamma);
		fp_free(alpha);
		bn_free(ss);
		bn_free(tt);
		for(i=0;i<t_max+1;i++){
	            fp_free(a[i]);
	        }
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

    printf("Using Primes of STARK\n");

    fp_t tabl2[we][nw],g,z,b,h,h1,h2,y;
    uint64_t q[k3][t_max+1],qq[k3];
    int tab3[16384]={0};
    bn_t tmp,m,e;
    
    for(i=0;i<we;i++){
        for(j=0;j<nw;j++){
            fp_null(tabl2[i][j]);
            fp_new(tabl2[i][j]);
        }
    }
    
tab3[ 0x1 ]= 0 ;
tab3[ 0x34f5 ]= 1 ;
tab3[ 0x1652 ]= 2 ;
tab3[ 0x2878 ]= 3 ;
tab3[ 0x26d8 ]= 4 ;
tab3[ 0x318d ]= 5 ;
tab3[ 0x2cd2 ]= 6 ;
tab3[ 0x27e4 ]= 7 ;
tab3[ 0x2f41 ]= 8 ;
tab3[ 0x275e ]= 9 ;
tab3[ 0x2810 ]= 10 ;
tab3[ 0x263c ]= 11 ;
tab3[ 0xa0c ]= 12 ;
tab3[ 0x3048 ]= 13 ;
tab3[ 0x33fa ]= 14 ;
tab3[ 0x27a2 ]= 15 ;
tab3[ 0x2a75 ]= 16 ;
tab3[ 0x2cf9 ]= 17 ;
tab3[ 0x1783 ]= 18 ;
tab3[ 0x447 ]= 19 ;
tab3[ 0x1ab9 ]= 20 ;
tab3[ 0x20b9 ]= 21 ;
tab3[ 0x1812 ]= 22 ;
tab3[ 0x2595 ]= 23 ;
tab3[ 0x8ec ]= 24 ;
tab3[ 0x1362 ]= 25 ;
tab3[ 0x3eef ]= 26 ;
tab3[ 0x36ff ]= 27 ;
tab3[ 0x27d7 ]= 28 ;
tab3[ 0xe2f ]= 29 ;
tab3[ 0x280f ]= 30 ;
tab3[ 0x37df ]= 31 ;
tab3[ 0xa34 ]= 32 ;
tab3[ 0x6d6 ]= 33 ;
tab3[ 0x2de6 ]= 34 ;
tab3[ 0x29c3 ]= 35 ;
tab3[ 0x1f6c ]= 36 ;
tab3[ 0x3f9b ]= 37 ;
tab3[ 0x3e66 ]= 38 ;
tab3[ 0xf18 ]= 39 ;
tab3[ 0x118c ]= 40 ;
tab3[ 0x52c ]= 41 ;
tab3[ 0x3791 ]= 42 ;
tab3[ 0x1ee3 ]= 43 ;
tab3[ 0x2394 ]= 44 ;
tab3[ 0x1a71 ]= 45 ;
tab3[ 0xda4 ]= 46 ;
tab3[ 0x3605 ]= 47 ;
tab3[ 0x730 ]= 48 ;
tab3[ 0x3c6d ]= 49 ;
tab3[ 0x2ebf ]= 50 ;
tab3[ 0x3051 ]= 51 ;
tab3[ 0x1f9e ]= 52 ;
tab3[ 0x271 ]= 53 ;
tab3[ 0x2e94 ]= 54 ;
tab3[ 0x2757 ]= 55 ;
tab3[ 0x11ac ]= 56 ;
tab3[ 0x15dc ]= 57 ;
tab3[ 0x2b8b ]= 58 ;
tab3[ 0x57c ]= 59 ;
tab3[ 0xc79 ]= 60 ;
tab3[ 0x36c4 ]= 61 ;
tab3[ 0x3e2 ]= 62 ;
tab3[ 0x926 ]= 63 ;
tab3[ 0x81e ]= 64 ;
tab3[ 0x317d ]= 65 ;
tab3[ 0xc75 ]= 66 ;
tab3[ 0x29bd ]= 67 ;
tab3[ 0xacd ]= 68 ;
tab3[ 0x3d22 ]= 69 ;
tab3[ 0x3cca ]= 70 ;
tab3[ 0x3854 ]= 71 ;
tab3[ 0x3390 ]= 72 ;
tab3[ 0x1546 ]= 73 ;
tab3[ 0xe66 ]= 74 ;
tab3[ 0x1cbf ]= 75 ;
tab3[ 0x1c4 ]= 76 ;
tab3[ 0x14d5 ]= 77 ;
tab3[ 0x31f2 ]= 78 ;
tab3[ 0x182e ]= 79 ;
tab3[ 0x20af ]= 80 ;
tab3[ 0x1fde ]= 81 ;
tab3[ 0xb9a ]= 82 ;
tab3[ 0x4c1 ]= 83 ;
tab3[ 0x20cf ]= 84 ;
tab3[ 0x2143 ]= 85 ;
tab3[ 0x8d3 ]= 86 ;
tab3[ 0x1483 ]= 87 ;
tab3[ 0x38f5 ]= 88 ;
tab3[ 0x1ed3 ]= 89 ;
tab3[ 0x2598 ]= 90 ;
tab3[ 0x369 ]= 91 ;
tab3[ 0x20b4 ]= 92 ;
tab3[ 0x21c2 ]= 93 ;
tab3[ 0x1758 ]= 94 ;
tab3[ 0xafb ]= 95 ;
tab3[ 0x1146 ]= 96 ;
tab3[ 0x23fb ]= 97 ;
tab3[ 0x199b ]= 98 ;
tab3[ 0x1f8 ]= 99 ;
tab3[ 0x34e3 ]= 100 ;
tab3[ 0x1d8f ]= 101 ;
tab3[ 0x25fa ]= 102 ;
tab3[ 0x300a ]= 103 ;
tab3[ 0x392a ]= 104 ;
tab3[ 0x84e ]= 105 ;
tab3[ 0xd84 ]= 106 ;
tab3[ 0x91b ]= 107 ;
tab3[ 0x3703 ]= 108 ;
tab3[ 0x2053 ]= 109 ;
tab3[ 0x1bcf ]= 110 ;
tab3[ 0x1a35 ]= 111 ;
tab3[ 0x1ac8 ]= 112 ;
tab3[ 0x1c53 ]= 113 ;
tab3[ 0xd73 ]= 114 ;
tab3[ 0x2eae ]= 115 ;
tab3[ 0x2fb7 ]= 116 ;
tab3[ 0x2dfc ]= 117 ;
tab3[ 0x2573 ]= 118 ;
tab3[ 0xf88 ]= 119 ;
tab3[ 0x1756 ]= 120 ;
tab3[ 0x508 ]= 121 ;
tab3[ 0xa3a ]= 122 ;
tab3[ 0x269 ]= 123 ;
tab3[ 0x89c ]= 124 ;
tab3[ 0x2d3e ]= 125 ;
tab3[ 0xdf2 ]= 126 ;
tab3[ 0xcba ]= 127 ;
tab3[ 0x0 ]= 128 ;
tab3[ 0xb0c ]= 129 ;
tab3[ 0x29af ]= 130 ;
tab3[ 0x1789 ]= 131 ;
tab3[ 0x1929 ]= 132 ;
tab3[ 0xe74 ]= 133 ;
tab3[ 0x132f ]= 134 ;
tab3[ 0x181d ]= 135 ;
tab3[ 0x10c0 ]= 136 ;
tab3[ 0x18a3 ]= 137 ;
tab3[ 0x17f1 ]= 138 ;
tab3[ 0x19c5 ]= 139 ;
tab3[ 0x35f5 ]= 140 ;
tab3[ 0xfb9 ]= 141 ;
tab3[ 0xc07 ]= 142 ;
tab3[ 0x185f ]= 143 ;
tab3[ 0x158c ]= 144 ;
tab3[ 0x1308 ]= 145 ;
tab3[ 0x287e ]= 146 ;
tab3[ 0x3bba ]= 147 ;
tab3[ 0x2548 ]= 148 ;
tab3[ 0x1f48 ]= 149 ;
tab3[ 0x27ef ]= 150 ;
tab3[ 0x1a6c ]= 151 ;
tab3[ 0x3715 ]= 152 ;
tab3[ 0x2c9f ]= 153 ;
tab3[ 0x112 ]= 154 ;
tab3[ 0x902 ]= 155 ;
tab3[ 0x182a ]= 156 ;
tab3[ 0x31d2 ]= 157 ;
tab3[ 0x17f2 ]= 158 ;
tab3[ 0x822 ]= 159 ;
tab3[ 0x35cd ]= 160 ;
tab3[ 0x392b ]= 161 ;
tab3[ 0x121b ]= 162 ;
tab3[ 0x163e ]= 163 ;
tab3[ 0x2095 ]= 164 ;
tab3[ 0x66 ]= 165 ;
tab3[ 0x19b ]= 166 ;
tab3[ 0x30e9 ]= 167 ;
tab3[ 0x2e75 ]= 168 ;
tab3[ 0x3ad5 ]= 169 ;
tab3[ 0x870 ]= 170 ;
tab3[ 0x211e ]= 171 ;
tab3[ 0x1c6d ]= 172 ;
tab3[ 0x2590 ]= 173 ;
tab3[ 0x325d ]= 174 ;
tab3[ 0x9fc ]= 175 ;
tab3[ 0x38d1 ]= 176 ;
tab3[ 0x394 ]= 177 ;
tab3[ 0x1142 ]= 178 ;
tab3[ 0xfb0 ]= 179 ;
tab3[ 0x2063 ]= 180 ;
tab3[ 0x3d90 ]= 181 ;
tab3[ 0x116d ]= 182 ;
tab3[ 0x18aa ]= 183 ;
tab3[ 0x2e55 ]= 184 ;
tab3[ 0x2a25 ]= 185 ;
tab3[ 0x1476 ]= 186 ;
tab3[ 0x3a85 ]= 187 ;
tab3[ 0x3388 ]= 188 ;
tab3[ 0x93d ]= 189 ;
tab3[ 0x3c1f ]= 190 ;
tab3[ 0x36db ]= 191 ;
tab3[ 0x37e3 ]= 192 ;
tab3[ 0xe84 ]= 193 ;
tab3[ 0x338c ]= 194 ;
tab3[ 0x1644 ]= 195 ;
tab3[ 0x3534 ]= 196 ;
tab3[ 0x2df ]= 197 ;
tab3[ 0x337 ]= 198 ;
tab3[ 0x7ad ]= 199 ;
tab3[ 0xc71 ]= 200 ;
tab3[ 0x2abb ]= 201 ;
tab3[ 0x319b ]= 202 ;
tab3[ 0x2342 ]= 203 ;
tab3[ 0x3e3d ]= 204 ;
tab3[ 0x2b2c ]= 205 ;
tab3[ 0xe0f ]= 206 ;
tab3[ 0x27d3 ]= 207 ;
tab3[ 0x1f52 ]= 208 ;
tab3[ 0x2023 ]= 209 ;
tab3[ 0x3467 ]= 210 ;
tab3[ 0x3b40 ]= 211 ;
tab3[ 0x1f32 ]= 212 ;
tab3[ 0x1ebe ]= 213 ;
tab3[ 0x372e ]= 214 ;
tab3[ 0x2b7e ]= 215 ;
tab3[ 0x70c ]= 216 ;
tab3[ 0x212e ]= 217 ;
tab3[ 0x1a69 ]= 218 ;
tab3[ 0x3c98 ]= 219 ;
tab3[ 0x1f4d ]= 220 ;
tab3[ 0x1e3f ]= 221 ;
tab3[ 0x28a9 ]= 222 ;
tab3[ 0x3506 ]= 223 ;
tab3[ 0x2ebb ]= 224 ;
tab3[ 0x1c06 ]= 225 ;
tab3[ 0x2666 ]= 226 ;
tab3[ 0x3e09 ]= 227 ;
tab3[ 0xb1e ]= 228 ;
tab3[ 0x2272 ]= 229 ;
tab3[ 0x1a07 ]= 230 ;
tab3[ 0xff7 ]= 231 ;
tab3[ 0x6d7 ]= 232 ;
tab3[ 0x37b3 ]= 233 ;
tab3[ 0x327d ]= 234 ;
tab3[ 0x36e6 ]= 235 ;
tab3[ 0x8fe ]= 236 ;
tab3[ 0x1fae ]= 237 ;
tab3[ 0x2432 ]= 238 ;
tab3[ 0x25cc ]= 239 ;
tab3[ 0x2539 ]= 240 ;
tab3[ 0x23ae ]= 241 ;
tab3[ 0x328e ]= 242 ;
tab3[ 0x1153 ]= 243 ;
tab3[ 0x104a ]= 244 ;
tab3[ 0x1205 ]= 245 ;
tab3[ 0x1a8e ]= 246 ;
tab3[ 0x3079 ]= 247 ;
tab3[ 0x28ab ]= 248 ;
tab3[ 0x3af9 ]= 249 ;
tab3[ 0x35c7 ]= 250 ;
tab3[ 0x3d98 ]= 251 ;
tab3[ 0x3765 ]= 252 ;
tab3[ 0x12c3 ]= 253 ;
tab3[ 0x320f ]= 254 ;
tab3[ 0x3347 ]= 255 ;


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
    bn_read_str(tmp,"3",1,16);
    fp_prime_conv(z,tmp);
    
    bn_read_str(m,"800000000000011",32,16);
    fp_exp(g,z,m);
    //bn_print(m);
    
    bn_read_str(e,"400000000000008",32,16);
    
   
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

