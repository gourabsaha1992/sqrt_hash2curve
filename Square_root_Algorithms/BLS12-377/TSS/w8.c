#include <stdio.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include<immintrin.h>
#include <relic/relic_core.h>
#include "relic/relic_md.h"


#define w 8
#define we 256
#define wf 1
#define wff 32
#define k3 1
#define n 46
#define nw 46

dig_t d;
static int i,j,ii,jj;
int l[1]={45};
uint64_t ll[1]={0x1fffffffffff};
int kk[2][2]={{0, 0}, {1, 0}};
int s[1]={0};
int t[1]={5};
int t_max=5;
int r[1]={1};
int ep[2][2]={{0, 0},{0, 0}};
int rho[2][2]={{0, 0},{0, 0}};



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




void findsqroot(fp_t u, fp_t y,fp_t tabl1[we][t_max+1],fp_t tabl2[we][nw],int tab3[65536],int tabl4[4096],dig_t q[k3][t_max+1],dig_t qq[k3],bn_t e,bn_t tmp) {
	fp_t v,v1,xx,x[k3],gamma,alpha;
	dig_t s,tt;
	dig_t and_f=255;
        dig_t c1,c,one=1;
	
	fp_t f,a[t[0]+1],temp,temp1;
	fp_null(f);
	dig_t Q,temp2;
	
	for(i=0;i<t[0]+1;i++){
	    fp_null(a[i]);
	    fp_new(a[i]); 
	}
	fp_null(temp);
	fp_null(temp1);

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
		fp_new(v);
		fp_new(v1);
		fp_new(gamma);
		fp_new(alpha);
		fp_new(f);
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
	        
	        for(j=0;j<t[i3]+1;j++){
			fp_copy(f,a[j]);
			if(j!=0){
			    fp_copy(temp,tabl2[q[i3][0]][5-j]);
			    for(i=1;i<=r[i3];i++)
			    {
			        fp_sqr(temp,temp);
			    }
			    fp_mul(f,f,temp);
		        }
			for(i=1;i<j;i++)
			{
			    fp_mul(f,f,tabl1[q[i3][i]][j+1-i]);
			}
			if(j==0){
				fp_prime_back(tmp, f);
	                        bn_get_dig(&d, tmp);
	                        d=d&0xffff;
	                        q[i3][j]=tab3[d]>>(w-l[0]+t[0]*w);
			}
			if(j!=0){
				fp_prime_back(tmp, f);
	                        bn_get_dig(&d, tmp);
	                        d=d&0xffff;
	                        q[i3][j]=tab3[d];
			}
            	    
	                
		}
	        Q=q[0][0];
                for(i=1;i<6;i++){
                    temp2= q[0][i]<<(45-(6-i)*w);
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
		fp_free(xx);
		fp_free(v);
		fp_free(v1);
		fp_free(gamma);
		fp_free(alpha);
		fp_free(f);
                fp_free(temp);
                fp_free(temp1);
                for(i=0;i<t[0]+1;i++){
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
    if (ep_param_set_any_pairf() != RLC_OK) {
        printf("Curve initialization failed\n");
        core_clean();
        return 1;
    }

    printf("Using Primes of bls12-377\n");

    fp_t tabl1[we][t_max+1],tabl2[we][nw],g,z,b,h,h1,h2,y;
    dig_t q[k3][t_max+1],qq[k3];
    int tabl4[4096]={0},tab3[65536]={0};
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
tab3[ 0x9fad ]= 1 ;
tab3[ 0x7eac ]= 2 ;
tab3[ 0x9ef5 ]= 3 ;
tab3[ 0x871f ]= 4 ;
tab3[ 0x3ce4 ]= 5 ;
tab3[ 0xb018 ]= 6 ;
tab3[ 0xd9b6 ]= 7 ;
tab3[ 0xdd1b ]= 8 ;
tab3[ 0xca1f ]= 9 ;
tab3[ 0x44e8 ]= 10 ;
tab3[ 0xb5c3 ]= 11 ;
tab3[ 0xa1ac ]= 12 ;
tab3[ 0xdce2 ]= 13 ;
tab3[ 0x70fe ]= 14 ;
tab3[ 0x5ba ]= 15 ;
tab3[ 0xea4 ]= 16 ;
tab3[ 0x1de5 ]= 17 ;
tab3[ 0xbfc0 ]= 18 ;
tab3[ 0x1b62 ]= 19 ;
tab3[ 0xcc00 ]= 20 ;
tab3[ 0xc672 ]= 21 ;
tab3[ 0x9497 ]= 22 ;
tab3[ 0x2003 ]= 23 ;
tab3[ 0x2e56 ]= 24 ;
tab3[ 0xa0e5 ]= 25 ;
tab3[ 0x3af3 ]= 26 ;
tab3[ 0xbb17 ]= 27 ;
tab3[ 0x652f ]= 28 ;
tab3[ 0xc1a6 ]= 29 ;
tab3[ 0xcd5f ]= 30 ;
tab3[ 0xc210 ]= 31 ;
tab3[ 0x7f11 ]= 32 ;
tab3[ 0xe2bb ]= 33 ;
tab3[ 0x9997 ]= 34 ;
tab3[ 0xa8a7 ]= 35 ;
tab3[ 0x400d ]= 36 ;
tab3[ 0x15d0 ]= 37 ;
tab3[ 0xcb44 ]= 38 ;
tab3[ 0x94e9 ]= 39 ;
tab3[ 0x247 ]= 40 ;
tab3[ 0x6a7a ]= 41 ;
tab3[ 0xed30 ]= 42 ;
tab3[ 0x20a1 ]= 43 ;
tab3[ 0x61d ]= 44 ;
tab3[ 0xa8b8 ]= 45 ;
tab3[ 0x9f20 ]= 46 ;
tab3[ 0x4f35 ]= 47 ;
tab3[ 0xed50 ]= 48 ;
tab3[ 0x6c3c ]= 49 ;
tab3[ 0x1354 ]= 50 ;
tab3[ 0x3982 ]= 51 ;
tab3[ 0xc0e4 ]= 52 ;
tab3[ 0xf754 ]= 53 ;
tab3[ 0xd0bd ]= 54 ;
tab3[ 0xca01 ]= 55 ;
tab3[ 0xfbd1 ]= 56 ;
tab3[ 0xd1c0 ]= 57 ;
tab3[ 0xe78a ]= 58 ;
tab3[ 0xcd10 ]= 59 ;
tab3[ 0x4685 ]= 60 ;
tab3[ 0xafde ]= 61 ;
tab3[ 0x2909 ]= 62 ;
tab3[ 0x2358 ]= 63 ;
tab3[ 0xe39e ]= 64 ;
tab3[ 0xc3e0 ]= 65 ;
tab3[ 0xe933 ]= 66 ;
tab3[ 0xc885 ]= 67 ;
tab3[ 0xa9f3 ]= 68 ;
tab3[ 0x15bd ]= 69 ;
tab3[ 0x545 ]= 70 ;
tab3[ 0xa174 ]= 71 ;
tab3[ 0xa19b ]= 72 ;
tab3[ 0xb579 ]= 73 ;
tab3[ 0xbe33 ]= 74 ;
tab3[ 0xa764 ]= 75 ;
tab3[ 0xe476 ]= 76 ;
tab3[ 0xa3b1 ]= 77 ;
tab3[ 0xf16e ]= 78 ;
tab3[ 0x3f04 ]= 79 ;
tab3[ 0x2ebe ]= 80 ;
tab3[ 0xf6ee ]= 81 ;
tab3[ 0xbfea ]= 82 ;
tab3[ 0xc1e ]= 83 ;
tab3[ 0x4a8c ]= 84 ;
tab3[ 0x1045 ]= 85 ;
tab3[ 0xff55 ]= 86 ;
tab3[ 0x649c ]= 87 ;
tab3[ 0x3f14 ]= 88 ;
tab3[ 0x8d2a ]= 89 ;
tab3[ 0xd68a ]= 90 ;
tab3[ 0x737d ]= 91 ;
tab3[ 0x39e2 ]= 92 ;
tab3[ 0x749a ]= 93 ;
tab3[ 0x8664 ]= 94 ;
tab3[ 0x3adc ]= 95 ;
tab3[ 0x7a47 ]= 96 ;
tab3[ 0xf94e ]= 97 ;
tab3[ 0xa772 ]= 98 ;
tab3[ 0xd8a6 ]= 99 ;
tab3[ 0xe49a ]= 100 ;
tab3[ 0x5313 ]= 101 ;
tab3[ 0x91d9 ]= 102 ;
tab3[ 0x3c29 ]= 103 ;
tab3[ 0x2ad5 ]= 104 ;
tab3[ 0xa48a ]= 105 ;
tab3[ 0xeaa2 ]= 106 ;
tab3[ 0xc5ca ]= 107 ;
tab3[ 0x628b ]= 108 ;
tab3[ 0x8d60 ]= 109 ;
tab3[ 0xe1cf ]= 110 ;
tab3[ 0xa717 ]= 111 ;
tab3[ 0x4dca ]= 112 ;
tab3[ 0xc8e7 ]= 113 ;
tab3[ 0xdc ]= 114 ;
tab3[ 0x9d47 ]= 115 ;
tab3[ 0xca8 ]= 116 ;
tab3[ 0xaa7e ]= 117 ;
tab3[ 0x27ec ]= 118 ;
tab3[ 0x3645 ]= 119 ;
tab3[ 0xf14f ]= 120 ;
tab3[ 0x35b1 ]= 121 ;
tab3[ 0x4cb3 ]= 122 ;
tab3[ 0x79a8 ]= 123 ;
tab3[ 0x575b ]= 124 ;
tab3[ 0x8b07 ]= 125 ;
tab3[ 0x5d4b ]= 126 ;
tab3[ 0xf56a ]= 127 ;
tab3[ 0x0 ]= 128 ;
tab3[ 0x6054 ]= 129 ;
tab3[ 0x8155 ]= 130 ;
tab3[ 0x610c ]= 131 ;
tab3[ 0x78e2 ]= 132 ;
tab3[ 0xc31d ]= 133 ;
tab3[ 0x4fe9 ]= 134 ;
tab3[ 0x264b ]= 135 ;
tab3[ 0x22e6 ]= 136 ;
tab3[ 0x35e2 ]= 137 ;
tab3[ 0xbb19 ]= 138 ;
tab3[ 0x4a3e ]= 139 ;
tab3[ 0x5e55 ]= 140 ;
tab3[ 0x231f ]= 141 ;
tab3[ 0x8f03 ]= 142 ;
tab3[ 0xfa47 ]= 143 ;
tab3[ 0xf15d ]= 144 ;
tab3[ 0xe21c ]= 145 ;
tab3[ 0x4041 ]= 146 ;
tab3[ 0xe49f ]= 147 ;
tab3[ 0x3401 ]= 148 ;
tab3[ 0x398f ]= 149 ;
tab3[ 0x6b6a ]= 150 ;
tab3[ 0xdffe ]= 151 ;
tab3[ 0xd1ab ]= 152 ;
tab3[ 0x5f1c ]= 153 ;
tab3[ 0xc50e ]= 154 ;
tab3[ 0x44ea ]= 155 ;
tab3[ 0x9ad2 ]= 156 ;
tab3[ 0x3e5b ]= 157 ;
tab3[ 0x32a2 ]= 158 ;
tab3[ 0x3df1 ]= 159 ;
tab3[ 0x80f0 ]= 160 ;
tab3[ 0x1d46 ]= 161 ;
tab3[ 0x666a ]= 162 ;
tab3[ 0x575a ]= 163 ;
tab3[ 0xbff4 ]= 164 ;
tab3[ 0xea31 ]= 165 ;
tab3[ 0x34bd ]= 166 ;
tab3[ 0x6b18 ]= 167 ;
tab3[ 0xfdba ]= 168 ;
tab3[ 0x9587 ]= 169 ;
tab3[ 0x12d1 ]= 170 ;
tab3[ 0xdf60 ]= 171 ;
tab3[ 0xf9e4 ]= 172 ;
tab3[ 0x5749 ]= 173 ;
tab3[ 0x60e1 ]= 174 ;
tab3[ 0xb0cc ]= 175 ;
tab3[ 0x12b1 ]= 176 ;
tab3[ 0x93c5 ]= 177 ;
tab3[ 0xecad ]= 178 ;
tab3[ 0xc67f ]= 179 ;
tab3[ 0x3f1d ]= 180 ;
tab3[ 0x8ad ]= 181 ;
tab3[ 0x2f44 ]= 182 ;
tab3[ 0x3600 ]= 183 ;
tab3[ 0x430 ]= 184 ;
tab3[ 0x2e41 ]= 185 ;
tab3[ 0x1877 ]= 186 ;
tab3[ 0x32f1 ]= 187 ;
tab3[ 0xb97c ]= 188 ;
tab3[ 0x5023 ]= 189 ;
tab3[ 0xd6f8 ]= 190 ;
tab3[ 0xdca9 ]= 191 ;
tab3[ 0x1c63 ]= 192 ;
tab3[ 0x3c21 ]= 193 ;
tab3[ 0x16ce ]= 194 ;
tab3[ 0x377c ]= 195 ;
tab3[ 0x560e ]= 196 ;
tab3[ 0xea44 ]= 197 ;
tab3[ 0xfabc ]= 198 ;
tab3[ 0x5e8d ]= 199 ;
tab3[ 0x5e66 ]= 200 ;
tab3[ 0x4a88 ]= 201 ;
tab3[ 0x41ce ]= 202 ;
tab3[ 0x589d ]= 203 ;
tab3[ 0x1b8b ]= 204 ;
tab3[ 0x5c50 ]= 205 ;
tab3[ 0xe93 ]= 206 ;
tab3[ 0xc0fd ]= 207 ;
tab3[ 0xd143 ]= 208 ;
tab3[ 0x913 ]= 209 ;
tab3[ 0x4017 ]= 210 ;
tab3[ 0xf3e3 ]= 211 ;
tab3[ 0xb575 ]= 212 ;
tab3[ 0xefbc ]= 213 ;
tab3[ 0xac ]= 214 ;
tab3[ 0x9b65 ]= 215 ;
tab3[ 0xc0ed ]= 216 ;
tab3[ 0x72d7 ]= 217 ;
tab3[ 0x2977 ]= 218 ;
tab3[ 0x8c84 ]= 219 ;
tab3[ 0xc61f ]= 220 ;
tab3[ 0x8b67 ]= 221 ;
tab3[ 0x799d ]= 222 ;
tab3[ 0xc525 ]= 223 ;
tab3[ 0x85ba ]= 224 ;
tab3[ 0x6b3 ]= 225 ;
tab3[ 0x588f ]= 226 ;
tab3[ 0x275b ]= 227 ;
tab3[ 0x1b67 ]= 228 ;
tab3[ 0xacee ]= 229 ;
tab3[ 0x6e28 ]= 230 ;
tab3[ 0xc3d8 ]= 231 ;
tab3[ 0xd52c ]= 232 ;
tab3[ 0x5b77 ]= 233 ;
tab3[ 0x155f ]= 234 ;
tab3[ 0x3a37 ]= 235 ;
tab3[ 0x9d76 ]= 236 ;
tab3[ 0x72a1 ]= 237 ;
tab3[ 0x1e32 ]= 238 ;
tab3[ 0x58ea ]= 239 ;
tab3[ 0xb237 ]= 240 ;
tab3[ 0x371a ]= 241 ;
tab3[ 0xff25 ]= 242 ;
tab3[ 0x62ba ]= 243 ;
tab3[ 0xf359 ]= 244 ;
tab3[ 0x5583 ]= 245 ;
tab3[ 0xd815 ]= 246 ;
tab3[ 0xc9bc ]= 247 ;
tab3[ 0xeb2 ]= 248 ;
tab3[ 0xca50 ]= 249 ;
tab3[ 0xb34e ]= 250 ;
tab3[ 0x8659 ]= 251 ;
tab3[ 0xa8a6 ]= 252 ;
tab3[ 0x74fa ]= 253 ;
tab3[ 0xa2b6 ]= 254 ;
tab3[ 0xa97 ]= 255 ; 

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
    bn_read_str(a1,"4000000000",10,16);
    fp_exp(h,g,a1);
    //h1=g^(2^(n-5))
    bn_read_str(a1,"20000000000",11,16);
    fp_exp(h1,g,a1);
    //h1=g^(2^(n-7))
    bn_read_str(a1,"8000000000",10,16);
    fp_exp(h2,g,a1);
    precomputation(g,h,h1,h2,tabl1,tabl2);
    printf("using tonelli-shank with look up tablle\n");

    
    MEASURE(findsqroot(b,y,tabl1,tabl2,tab3,tabl4,q,qq,e,tmp);)
    
    
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

