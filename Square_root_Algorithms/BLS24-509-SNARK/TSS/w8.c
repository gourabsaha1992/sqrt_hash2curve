#include <stdio.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include<immintrin.h>
#include <relic/relic_core.h>
#include "relic/relic_md.h"

#define w 8
#define we 256
#define n 37
#define k3 1
#define nw 5
#define count 0 

int i,j,ii,jj,i3;
int l[k3]= {36} ;
uint64_t ll[k3]={
0xfffffffff };
int kk[k3+1][k3+1]= {{0, 0}, {1, 0}} ;
int s[k3]= {0} ;
int t[k3]= {4} ;
int t_max= 4 ;
int r[k3]= {1} ;
int ep[k3+1][k3+1]= {{0, 0}, {0, 0}} ;
int rho[k3+1][k3+1]= {{0, 0}, {0, 0}} ;

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
uint64_t Q,temp2;

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




void precomputation(fp_t g,fp_t h,fp_t h1,fp_t tabl1[we][t_max+1],fp_t tabl2[we][nw]){
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



void findsqroot(fp_t u, fp_t y,fp_t tabl1[we][t_max+1],fp_t tabl2[we][nw],int tab3[32768],uint64_t q[k3][t_max+1],uint64_t qq[k3],bn_t e, bn_t tmp) {
	fp_t v,v1,x[k3],gamma,alpha;
	uint64_t s,tt;
	
	dig_t and_f=255;
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
		
		
		fp_new(v);
		fp_new(v1);
		fp_new(gamma);
		fp_new(alpha);
		fp_new(temp);
		fp_new(temp1);

		
		fp_exp(v, u, e);
		fp_mul(v1,v,u);
		fp_mul(x[k3-1],v,v1);
		
		
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
		            fp_copy(temp,tabl2[q[i3][0]][4-j]);
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
                            d=d&0x7fff;
                            q[i3][j]=tab3[d]>>(w-l[k3-1]+t[k3-1]*w);
                            }
                        if(j!=0){
                            fp_prime_back(tmp, f);
                            bn_get_dig(&d, tmp);
                            d=d&0x7fff;
                            q[i3][j]=tab3[d];
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
		
		fp_free(xx);
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

    printf("Using Primes of bls24-509\n");

    fp_t tabl1[we][t_max+1],tabl2[we][nw],g,z,b,h,h1,h2,y;
    uint64_t q[k3][t_max+1],qq[k3];
    int tab3[32768]={0};
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
tab3[ 0x7918 ]= 1 ;
tab3[ 0x3d60 ]= 2 ;
tab3[ 0x5f9 ]= 3 ;
tab3[ 0x549f ]= 4 ;
tab3[ 0x704f ]= 5 ;
tab3[ 0x45b4 ]= 6 ;
tab3[ 0x377a ]= 7 ;
tab3[ 0x68e8 ]= 8 ;
tab3[ 0x5f82 ]= 9 ;
tab3[ 0x45f4 ]= 10 ;
tab3[ 0x53a1 ]= 11 ;
tab3[ 0x37f1 ]= 12 ;
tab3[ 0x5a9e ]= 13 ;
tab3[ 0x7bd8 ]= 14 ;
tab3[ 0x6cd ]= 15 ;
tab3[ 0x524b ]= 16 ;
tab3[ 0x2b83 ]= 17 ;
tab3[ 0x75e7 ]= 18 ;
tab3[ 0x3c30 ]= 19 ;
tab3[ 0x323c ]= 20 ;
tab3[ 0x5838 ]= 21 ;
tab3[ 0x437 ]= 22 ;
tab3[ 0x7f40 ]= 23 ;
tab3[ 0x5c3b ]= 24 ;
tab3[ 0x4964 ]= 25 ;
tab3[ 0x23b4 ]= 26 ;
tab3[ 0x5fd6 ]= 27 ;
tab3[ 0x4b4f ]= 28 ;
tab3[ 0x5995 ]= 29 ;
tab3[ 0x5aff ]= 30 ;
tab3[ 0xaa0 ]= 31 ;
tab3[ 0x6a15 ]= 32 ;
tab3[ 0x6355 ]= 33 ;
tab3[ 0x41bf ]= 34 ;
tab3[ 0x1f1 ]= 35 ;
tab3[ 0x10e7 ]= 36 ;
tab3[ 0x462 ]= 37 ;
tab3[ 0x3918 ]= 38 ;
tab3[ 0x7797 ]= 39 ;
tab3[ 0x47a6 ]= 40 ;
tab3[ 0x222f ]= 41 ;
tab3[ 0xeb3 ]= 42 ;
tab3[ 0x2d0f ]= 43 ;
tab3[ 0x55be ]= 44 ;
tab3[ 0xf5f ]= 45 ;
tab3[ 0x77a5 ]= 46 ;
tab3[ 0x47f ]= 47 ;
tab3[ 0x1be ]= 48 ;
tab3[ 0x24f5 ]= 49 ;
tab3[ 0x5e0c ]= 50 ;
tab3[ 0x5be3 ]= 51 ;
tab3[ 0x503a ]= 52 ;
tab3[ 0x613 ]= 53 ;
tab3[ 0x7c09 ]= 54 ;
tab3[ 0x45a7 ]= 55 ;
tab3[ 0x2579 ]= 56 ;
tab3[ 0x7769 ]= 57 ;
tab3[ 0x24db ]= 58 ;
tab3[ 0x1705 ]= 59 ;
tab3[ 0x20ff ]= 60 ;
tab3[ 0x4821 ]= 61 ;
tab3[ 0x3133 ]= 62 ;
tab3[ 0x6fe3 ]= 63 ;
tab3[ 0x271e ]= 64 ;
tab3[ 0x187d ]= 65 ;
tab3[ 0x573c ]= 66 ;
tab3[ 0x10c1 ]= 67 ;
tab3[ 0x4151 ]= 68 ;
tab3[ 0x780d ]= 69 ;
tab3[ 0x77c7 ]= 70 ;
tab3[ 0x67af ]= 71 ;
tab3[ 0x398a ]= 72 ;
tab3[ 0x123a ]= 73 ;
tab3[ 0x556a ]= 74 ;
tab3[ 0x4ec1 ]= 75 ;
tab3[ 0x4130 ]= 76 ;
tab3[ 0x4a0b ]= 77 ;
tab3[ 0x86b ]= 78 ;
tab3[ 0xf61 ]= 79 ;
tab3[ 0x4cdf ]= 80 ;
tab3[ 0x4e62 ]= 81 ;
tab3[ 0x37a2 ]= 82 ;
tab3[ 0x7a43 ]= 83 ;
tab3[ 0x2a16 ]= 84 ;
tab3[ 0x6889 ]= 85 ;
tab3[ 0xa43 ]= 86 ;
tab3[ 0x40c9 ]= 87 ;
tab3[ 0x2978 ]= 88 ;
tab3[ 0x4a56 ]= 89 ;
tab3[ 0x6636 ]= 90 ;
tab3[ 0x7181 ]= 91 ;
tab3[ 0x3e74 ]= 92 ;
tab3[ 0xa55 ]= 93 ;
tab3[ 0x4ae8 ]= 94 ;
tab3[ 0xd26 ]= 95 ;
tab3[ 0xbe4 ]= 96 ;
tab3[ 0x63ff ]= 97 ;
tab3[ 0x3487 ]= 98 ;
tab3[ 0x56dc ]= 99 ;
tab3[ 0x6bdd ]= 100 ;
tab3[ 0x403d ]= 101 ;
tab3[ 0xf56 ]= 102 ;
tab3[ 0x24b3 ]= 103 ;
tab3[ 0x3dd ]= 104 ;
tab3[ 0x11ea ]= 105 ;
tab3[ 0x6ca2 ]= 106 ;
tab3[ 0x30f0 ]= 107 ;
tab3[ 0x4c1d ]= 108 ;
tab3[ 0x23fa ]= 109 ;
tab3[ 0x30e2 ]= 110 ;
tab3[ 0x744a ]= 111 ;
tab3[ 0x73c ]= 112 ;
tab3[ 0x2fa8 ]= 113 ;
tab3[ 0x7ff ]= 114 ;
tab3[ 0x7a37 ]= 115 ;
tab3[ 0x6f94 ]= 116 ;
tab3[ 0x7ab3 ]= 117 ;
tab3[ 0x3a2c ]= 118 ;
tab3[ 0x2b6d ]= 119 ;
tab3[ 0x681d ]= 120 ;
tab3[ 0xfac ]= 121 ;
tab3[ 0x4221 ]= 122 ;
tab3[ 0x397e ]= 123 ;
tab3[ 0x7d5b ]= 124 ;
tab3[ 0x6974 ]= 125 ;
tab3[ 0x6ff6 ]= 126 ;
tab3[ 0x5c27 ]= 127 ;
tab3[ 0x0 ]= 128 ;
tab3[ 0x6e9 ]= 129 ;
tab3[ 0x42a1 ]= 130 ;
tab3[ 0x7a08 ]= 131 ;
tab3[ 0x2b62 ]= 132 ;
tab3[ 0xfb2 ]= 133 ;
tab3[ 0x3a4d ]= 134 ;
tab3[ 0x4887 ]= 135 ;
tab3[ 0x1719 ]= 136 ;
tab3[ 0x207f ]= 137 ;
tab3[ 0x3a0d ]= 138 ;
tab3[ 0x2c60 ]= 139 ;
tab3[ 0x4810 ]= 140 ;
tab3[ 0x2563 ]= 141 ;
tab3[ 0x429 ]= 142 ;
tab3[ 0x7934 ]= 143 ;
tab3[ 0x2db6 ]= 144 ;
tab3[ 0x547e ]= 145 ;
tab3[ 0xa1a ]= 146 ;
tab3[ 0x43d1 ]= 147 ;
tab3[ 0x4dc5 ]= 148 ;
tab3[ 0x27c9 ]= 149 ;
tab3[ 0x7bca ]= 150 ;
tab3[ 0xc1 ]= 151 ;
tab3[ 0x23c6 ]= 152 ;
tab3[ 0x369d ]= 153 ;
tab3[ 0x5c4d ]= 154 ;
tab3[ 0x202b ]= 155 ;
tab3[ 0x34b2 ]= 156 ;
tab3[ 0x266c ]= 157 ;
tab3[ 0x2502 ]= 158 ;
tab3[ 0x7561 ]= 159 ;
tab3[ 0x15ec ]= 160 ;
tab3[ 0x1cac ]= 161 ;
tab3[ 0x3e42 ]= 162 ;
tab3[ 0x7e10 ]= 163 ;
tab3[ 0x6f1a ]= 164 ;
tab3[ 0x7b9f ]= 165 ;
tab3[ 0x46e9 ]= 166 ;
tab3[ 0x86a ]= 167 ;
tab3[ 0x385b ]= 168 ;
tab3[ 0x5dd2 ]= 169 ;
tab3[ 0x714e ]= 170 ;
tab3[ 0x52f2 ]= 171 ;
tab3[ 0x2a43 ]= 172 ;
tab3[ 0x70a2 ]= 173 ;
tab3[ 0x85c ]= 174 ;
tab3[ 0x7b82 ]= 175 ;
tab3[ 0x7e43 ]= 176 ;
tab3[ 0x5b0c ]= 177 ;
tab3[ 0x21f5 ]= 178 ;
tab3[ 0x241e ]= 179 ;
tab3[ 0x2fc7 ]= 180 ;
tab3[ 0x79ee ]= 181 ;
tab3[ 0x3f8 ]= 182 ;
tab3[ 0x3a5a ]= 183 ;
tab3[ 0x5a88 ]= 184 ;
tab3[ 0x898 ]= 185 ;
tab3[ 0x5b26 ]= 186 ;
tab3[ 0x68fc ]= 187 ;
tab3[ 0x5f02 ]= 188 ;
tab3[ 0x37e0 ]= 189 ;
tab3[ 0x4ece ]= 190 ;
tab3[ 0x101e ]= 191 ;
tab3[ 0x58e3 ]= 192 ;
tab3[ 0x6784 ]= 193 ;
tab3[ 0x28c5 ]= 194 ;
tab3[ 0x6f40 ]= 195 ;
tab3[ 0x3eb0 ]= 196 ;
tab3[ 0x7f4 ]= 197 ;
tab3[ 0x83a ]= 198 ;
tab3[ 0x1852 ]= 199 ;
tab3[ 0x4677 ]= 200 ;
tab3[ 0x6dc7 ]= 201 ;
tab3[ 0x2a97 ]= 202 ;
tab3[ 0x3140 ]= 203 ;
tab3[ 0x3ed1 ]= 204 ;
tab3[ 0x35f6 ]= 205 ;
tab3[ 0x7796 ]= 206 ;
tab3[ 0x70a0 ]= 207 ;
tab3[ 0x3322 ]= 208 ;
tab3[ 0x319f ]= 209 ;
tab3[ 0x485f ]= 210 ;
tab3[ 0x5be ]= 211 ;
tab3[ 0x55eb ]= 212 ;
tab3[ 0x1778 ]= 213 ;
tab3[ 0x75be ]= 214 ;
tab3[ 0x3f38 ]= 215 ;
tab3[ 0x5689 ]= 216 ;
tab3[ 0x35ab ]= 217 ;
tab3[ 0x19cb ]= 218 ;
tab3[ 0xe80 ]= 219 ;
tab3[ 0x418d ]= 220 ;
tab3[ 0x75ac ]= 221 ;
tab3[ 0x3519 ]= 222 ;
tab3[ 0x72db ]= 223 ;
tab3[ 0x741d ]= 224 ;
tab3[ 0x1c02 ]= 225 ;
tab3[ 0x4b7a ]= 226 ;
tab3[ 0x2925 ]= 227 ;
tab3[ 0x1424 ]= 228 ;
tab3[ 0x3fc4 ]= 229 ;
tab3[ 0x70ab ]= 230 ;
tab3[ 0x5b4e ]= 231 ;
tab3[ 0x7c24 ]= 232 ;
tab3[ 0x6e17 ]= 233 ;
tab3[ 0x135f ]= 234 ;
tab3[ 0x4f11 ]= 235 ;
tab3[ 0x33e4 ]= 236 ;
tab3[ 0x5c07 ]= 237 ;
tab3[ 0x4f1f ]= 238 ;
tab3[ 0xbb7 ]= 239 ;
tab3[ 0x78c5 ]= 240 ;
tab3[ 0x5059 ]= 241 ;
tab3[ 0x7802 ]= 242 ;
tab3[ 0x5ca ]= 243 ;
tab3[ 0x106d ]= 244 ;
tab3[ 0x54e ]= 245 ;
tab3[ 0x45d5 ]= 246 ;
tab3[ 0x5494 ]= 247 ;
tab3[ 0x17e4 ]= 248 ;
tab3[ 0x7055 ]= 249 ;
tab3[ 0x3de0 ]= 250 ;
tab3[ 0x4683 ]= 251 ;
tab3[ 0x2a6 ]= 252 ;
tab3[ 0x168d ]= 253 ;
tab3[ 0x100b ]= 254 ;
tab3[ 0x23da ]= 255 ;
//    for(i=0;i<we;i++){
 //       fp_null(tabl3[i]);
 //       fp_new(tabl3[i]);
 //   }


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
    
    bn_read_str(tmp,"b",96,16);
    fp_prime_conv(z,tmp);
    
    bn_read_str(m,"ba29500be5eeb41528150f09d4eb38e93cac77669619e1d1b6156f110ecdcab5f527a4d95af73ebeb95690032ee595c74e5000ada8a9600000408f",118,16);
    fp_exp(g,z,m);
    
    bn_read_str(e,"5d14a805f2f75a0a940a8784ea759c749e563bb34b0cf0e8db0ab7888766e55afa93d26cad7b9f5f5cab48019772cae3a7280056d454b000002047",118,16);
    
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
    precomputation(g,h,h1,tabl1,tabl2);
    printf("using tonelli-shank with look up tablle\n");
    printf("W=%d\nK=%d\n",w,k3);

    
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

