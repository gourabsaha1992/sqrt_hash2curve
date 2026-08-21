

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <immintrin.h>


#define LIMB_BITS 60
#define LIMB_MASK 0x0FFFFFFFFFFFFFFFull
typedef struct { uint64_t low; uint64_t high; } uint120_t;
static inline uint120_t make120(uint64_t h,uint64_t l)
{ return (uint120_t){.low=l&LIMB_MASK,.high=h&LIMB_MASK}; }
static inline uint120_t add120(uint120_t a,uint120_t b){
    uint64_t r=a.low+b.low,c=r>>LIMB_BITS;
    return make120((a.high+b.high+c)&LIMB_MASK,r&LIMB_MASK);}
static inline uint120_t sub120(uint120_t a,uint120_t b){
    uint64_t r=(1ull<<LIMB_BITS)+a.low-b.low,bw=1-(r>>LIMB_BITS);
    return make120((a.high-b.high-bw)&LIMB_MASK,r&LIMB_MASK);}
static inline uint120_t rshift120(uint120_t a,int n){
    if(n<=0)return a; if(n>=120)return make120(0,0);
    if(n<LIMB_BITS)return make120((a.high>>n)&LIMB_MASK,
        ((a.low>>n)|(a.high<<(LIMB_BITS-n)))&LIMB_MASK);
    return make120(0,(a.high>>(n-LIMB_BITS))&LIMB_MASK);}
#define POW96  make120(1ull<<36,0)
#define MASK48 0x0000FFFFFFFFFFFFull


#define CHUNK96(e,k) (((k)<30)?(uint8_t)((e).low >>((k)*2))&0x3 \
                               :(uint8_t)((e).high>>(((k)-30)*2))&0x3)

#define REPEAT 100000
#define WARMUP (REPEAT/4)
double RDTSC_clk[REPEAT],RDTSC_clk_min,RDTSC_clk_median,RDTSC_clk_max;
int iii,jjj,ttt; double min_v,l1;
unsigned long long RDTSC_start_clk,RDTSC_end_clk;
static inline unsigned long long get_Clks(void){
    unsigned int lo,hi;
    __asm__ volatile("cpuid\n\t""rdtsc\n\t":"=a"(lo),"=d"(hi)::"rbx","rcx","memory");
    return((unsigned long long)hi<<32)|lo;}
#define MEASURE(x) do{ \
    for(int _w=0;_w<WARMUP;_w++){x} \
    for(int _r=0;_r<REPEAT;_r++){ \
        RDTSC_start_clk=get_Clks();{x}RDTSC_end_clk=get_Clks(); \
        RDTSC_clk[_r]=(double)(RDTSC_end_clk-RDTSC_start_clk);} \
    for(iii=0;iii<REPEAT;iii++){min_v=RDTSC_clk[iii];ttt=iii; \
        for(jjj=iii+1;jjj<REPEAT;jjj++) \
            if(min_v>RDTSC_clk[jjj]){min_v=RDTSC_clk[jjj];ttt=jjj;} \
        l1=RDTSC_clk[ttt];RDTSC_clk[ttt]=RDTSC_clk[iii];RDTSC_clk[iii]=l1;} \
    RDTSC_clk_min=RDTSC_clk[0]; \
    RDTSC_clk_median=RDTSC_clk[REPEAT/2]; \
    RDTSC_clk_max=RDTSC_clk[REPEAT-1];}while(0)
dig_t d;
int _tmp;
#define W  2
#define WE 4
#define N  192
#define NW 96
int countM=0,countS=0;

static void SELECT(fp_t a0,fp_t a1,bool ctl,fp_t out){
    if(ctl)fp_copy(out,a1);else fp_copy(out,a0);}
static int rll_lookup(fp_t x,fp_t rll[WE]){
    for(int k=0;k<WE;k++)if(fp_cmp(rll[k],x)==RLC_EQ)return k;
    return 0;}


static void GPOW_189_1(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=(1ull<<1)-1; e<<=1;
    fp_copy(t,gw[94][e&0x3]);}


static void GPOW_186_3(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=(1ull<<3)-1;
    fp_copy(t,gw[93][e&0x3]);
    e>>=2;fp_mul(t,t,gw[94][e&0x3]);//countM++;
    }


static void GPOW_180_6(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=(1ull<<6)-1;
    fp_copy(t,gw[90][e&0x3]);
    e>>=2;fp_mul(t,t,gw[91][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[92][e&0x3]);//countM++;
    }


static void GPOW_168_12(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=(1ull<<12)-1;
    fp_copy(t,gw[84][e&0x3]);
    e>>=2;fp_mul(t,t,gw[85][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[86][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[87][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[88][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[89][e&0x3]);//countM++;
    }


static void GPOW_144_24(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=(1ull<<24)-1;
    fp_copy(t,gw[72][e&0x3]);
    e>>=2;fp_mul(t,t,gw[73][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[74][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[75][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[76][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[77][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[78][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[79][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[80][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[81][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[82][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[83][e&0x3]);//countM++;
    }


static void GPOW_96_48(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=MASK48;
    fp_copy(t,gw[48][e&0x3]);
    e>>=2;fp_mul(t,t,gw[49][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[50][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[51][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[52][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[53][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[54][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[55][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[56][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[57][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[58][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[59][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[60][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[61][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[62][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[63][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[64][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[65][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[66][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[67][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[68][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[69][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[70][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[71][e&0x3]);//countM++;
    }


static void GPOW_0_95(fp_t t,uint120_t e,fp_t gw[NW][WE]){
    e.high&=~(1ull<<35); /* mask to 95 bits */
    fp_copy(t,gw[0][CHUNK96(e,0)]);
    fp_mul(t,t,gw[1][CHUNK96(e,1)]);//countM++;
    fp_mul(t,t,gw[2][CHUNK96(e,2)]);//countM++;
    fp_mul(t,t,gw[3][CHUNK96(e,3)]);//countM++;
    fp_mul(t,t,gw[4][CHUNK96(e,4)]);//countM++;
    fp_mul(t,t,gw[5][CHUNK96(e,5)]);//countM++;
    fp_mul(t,t,gw[6][CHUNK96(e,6)]);//countM++;
    fp_mul(t,t,gw[7][CHUNK96(e,7)]);//countM++;
    fp_mul(t,t,gw[8][CHUNK96(e,8)]);//countM++;
    fp_mul(t,t,gw[9][CHUNK96(e,9)]);//countM++;
    fp_mul(t,t,gw[10][CHUNK96(e,10)]);//countM++;
    fp_mul(t,t,gw[11][CHUNK96(e,11)]);//countM++;
    fp_mul(t,t,gw[12][CHUNK96(e,12)]);//countM++;
    fp_mul(t,t,gw[13][CHUNK96(e,13)]);//countM++;
    fp_mul(t,t,gw[14][CHUNK96(e,14)]);//countM++;
    fp_mul(t,t,gw[15][CHUNK96(e,15)]);//countM++;
    fp_mul(t,t,gw[16][CHUNK96(e,16)]);//countM++;
    fp_mul(t,t,gw[17][CHUNK96(e,17)]);//countM++;
    fp_mul(t,t,gw[18][CHUNK96(e,18)]);//countM++;
    fp_mul(t,t,gw[19][CHUNK96(e,19)]);//countM++;
    fp_mul(t,t,gw[20][CHUNK96(e,20)]);//countM++;
    fp_mul(t,t,gw[21][CHUNK96(e,21)]);//countM++;
    fp_mul(t,t,gw[22][CHUNK96(e,22)]);//countM++;
    fp_mul(t,t,gw[23][CHUNK96(e,23)]);//countM++;
    fp_mul(t,t,gw[24][CHUNK96(e,24)]);//countM++;
    fp_mul(t,t,gw[25][CHUNK96(e,25)]);//countM++;
    fp_mul(t,t,gw[26][CHUNK96(e,26)]);//countM++;
    fp_mul(t,t,gw[27][CHUNK96(e,27)]);//countM++;
    fp_mul(t,t,gw[28][CHUNK96(e,28)]);//countM++;
    fp_mul(t,t,gw[29][CHUNK96(e,29)]);//countM++;
    fp_mul(t,t,gw[30][CHUNK96(e,30)]);//countM++;
    fp_mul(t,t,gw[31][CHUNK96(e,31)]);//countM++;
    fp_mul(t,t,gw[32][CHUNK96(e,32)]);//countM++;
    fp_mul(t,t,gw[33][CHUNK96(e,33)]);//countM++;
    fp_mul(t,t,gw[34][CHUNK96(e,34)]);//countM++;
    fp_mul(t,t,gw[35][CHUNK96(e,35)]);//countM++;
    fp_mul(t,t,gw[36][CHUNK96(e,36)]);//countM++;
    fp_mul(t,t,gw[37][CHUNK96(e,37)]);//countM++;
    fp_mul(t,t,gw[38][CHUNK96(e,38)]);//countM++;
    fp_mul(t,t,gw[39][CHUNK96(e,39)]);//countM++;
    fp_mul(t,t,gw[40][CHUNK96(e,40)]);//countM++;
    fp_mul(t,t,gw[41][CHUNK96(e,41)]);//countM++;
    fp_mul(t,t,gw[42][CHUNK96(e,42)]);//countM++;
    fp_mul(t,t,gw[43][CHUNK96(e,43)]);//countM++;
    fp_mul(t,t,gw[44][CHUNK96(e,44)]);//countM++;
    fp_mul(t,t,gw[45][CHUNK96(e,45)]);//countM++;
    fp_mul(t,t,gw[46][CHUNK96(e,46)]);//countM++;
    fp_mul(t,t,gw[47][CHUNK96(e,47)]);//countM++;
    }


static void GPOW_95_48(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=MASK48; e<<=1;
    fp_copy(t,gw[47][e&0x3]);
    e>>=2;fp_mul(t,t,gw[48][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[49][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[50][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[51][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[52][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[53][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[54][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[55][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[56][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[57][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[58][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[59][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[60][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[61][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[62][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[63][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[64][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[65][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[66][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[67][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[68][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[69][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[70][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[71][e&0x3]);//countM++;
    }


static void GPOW_143_24(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=(1ull<<24)-1; e<<=1;
    fp_copy(t,gw[71][e&0x3]);
    e>>=2;fp_mul(t,t,gw[72][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[73][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[74][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[75][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[76][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[77][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[78][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[79][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[80][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[81][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[82][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[83][e&0x3]);//countM++;
    }


static void GPOW_167_12(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=(1ull<<12)-1; e<<=1;
    fp_copy(t,gw[83][e&0x3]);
    e>>=2;fp_mul(t,t,gw[84][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[85][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[86][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[87][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[88][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[89][e&0x3]);//countM++;
    }


static void GPOW_179_6(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=(1ull<<6)-1; e<<=1;
    fp_copy(t,gw[89][e&0x3]);
    e>>=2;fp_mul(t,t,gw[90][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[91][e&0x3]);//countM++;
    e>>=2;fp_mul(t,t,gw[92][e&0x3]);//countM++;
    }


static void GPOW_185_3(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=(1ull<<3)-1; e<<=1;
    fp_copy(t,gw[92][e&0x3]);
    e>>=2;fp_mul(t,t,gw[93][e&0x3]);//countM++;
    }


static void GPOW_188_1(fp_t t,uint64_t e,fp_t gw[NW][WE]){
    e&=(1ull<<1)-1;
    fp_copy(t,gw[94][e&0x3]);}


void precomputation(fp_t g,fp_t h,fp_t hh,
                    fp_t gw[NW][WE],fp_t rll[WE],fp_t fll[WE],fp_t gpp[N]){
    bn_t temp,one;bn_null(temp);bn_new(temp);bn_null(one);bn_new(one);bn_set_dig(one,1);
    for(int v=0;v<WE;v++){bn_set_dig(temp,v);fp_exp(rll[v],h,temp);fp_exp(fll[v],hh,temp);}
    for(int i=0;i<NW;i++)for(int j=0;j<WE;j++){
        bn_lsh(temp,one,i*W);bn_mul_dig(temp,temp,j);fp_exp(gw[i][j],g,temp);}
    fp_copy(gpp[0],g);
    for(int i=1;i<N;i++)fp_sqr(gpp[i],gpp[i-1]);
    bn_free(temp);bn_free(one);}

void solve_dlp_pow2_flat(fp_t u_A,fp_t out_d,
                         fp_t gw[NW][WE],fp_t rll[WE],fp_t fll[WE],fp_t gpp[N], uint8_t rlll[4])
{
#define FA(v) fp_null(v);fp_new(v)
    
    fp_t h0_A,h0_PA,h0_PAH,h0_PAHH;
    bn_t tmp_bn;
    bn_null(tmp_bn);
    bn_new(tmp_bn);
    FA(h0_A);FA(h0_PA);FA(h0_PAH);FA(h0_PAHH);


    fp_t sq_H,h1_tmp,u2_tmp;
    FA(sq_H);FA(h1_tmp);FA(u2_tmp);


    fp_t h0_PAHHH;   
    FA(h0_PAHHH);
    
    fp_t u2_temp_2; 
    FA(u2_temp_2);
    
    fp_t u2_temp_3;  
    FA(u2_temp_3);
    
    fp_t h0_PAHHHH; 
    FA(h0_PAHHHH);
    
    fp_t h0_PAHLH;
    FA(h0_PAHLH);
    
    
    
    fp_t h0_PAHHHHH;
    FA(h0_PAHHHHH);

    
    fp_t h1_PAHH,u2_PAHH; FA(h1_PAHH);FA(u2_PAHH);
    fp_t h1_PAH, u2_PAH;  FA(h1_PAH); FA(u2_PAH);
    fp_t h1_PA,  u2_PA;   FA(h1_PA);  FA(u2_PA);
    fp_t f_A,    u_X0;    FA(f_A);    FA(u_X0);

    
    fp_t h0_X0,h1_X0,u_X1; FA(h0_X0);FA(h1_X0);FA(u_X1);
    fp_t h0_X1,h1_X1,u_X2; FA(h0_X1);FA(h1_X1);FA(u_X2);
    fp_t h0_X2,h1_X2,u_X3; FA(h0_X2);FA(h1_X2);FA(u_X3);
    fp_t h0_X3,h1_X3,u_X4; FA(h0_X3);FA(h1_X3);FA(u_X4);
    fp_t h0_X4,h1_X4,u_X5; FA(h0_X4);FA(h1_X4);FA(u_X5);
    fp_t h0_X5,h1_X5,u_X6; FA(h0_X5);FA(h1_X5);FA(u_X6);

    
    fp_t t_X5,t_X4,t_X3,t_X2,t_X1,t_X0,t_A;
    FA(t_X5);FA(t_X4);FA(t_X3);FA(t_X2);FA(t_X1);FA(t_X0);FA(t_A);
#undef FA
   
    fp_copy(h0_A,u_A);
    for(int j=0;j<96;j++){fp_sqr(h0_A,h0_A);//countS++;
    }

    fp_copy(h0_PA,h0_A);
    for(int j=0;j<48;j++){fp_sqr(h0_PA,h0_PA);//countS++;
    }
    
    fp_copy(h0_PAH,h0_PA);
    for(int j=0;j<24;j++){fp_sqr(h0_PAH,h0_PAH);//countS++;
    }
   

    fp_copy(h0_PAHH,h0_PAH);
    for(int j=0;j<12;j++){fp_sqr(h0_PAHH,h0_PAHH);//countS++;
    }
   
    {
    
    fp_copy(h0_PAHHH,h0_PAHH);
    for(int j=0;j<6;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }
   
    fp_copy(sq_H,h0_PAHHH);
    
    fp_copy(h0_PAHHHH,h0_PAHHH);
    for(int j=0;j<3;j++){fp_sqr(h0_PAHHHH,h0_PAHHHH);//countS++;
    }
   
    
    fp_copy(h0_PAHHHHH,h0_PAHHHH);
    for(int j=0;j<2;j++){fp_sqr(h0_PAHHHHH,h0_PAHHHHH);//countS++;
    }
   
    fp_prime_back(tmp_bn, h0_PAHHHHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    
    uint64_t c_PAHHHH_H=rlll[d]>>1;
   
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_PAHHHH_H,gw);SELECT(h1_tmp,gpp[190],c_PAHHHH_H==0,h1_tmp);//countM++;
  
    fp_mul(u2_tmp,h0_PAHHHH,h1_tmp);
  
    
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
 
    uint64_t c_PAHHHH_L=rlll[d];  
    uint64_t c_PAHHHH=c_PAHHHH_H+((c_PAHHHH_L-1)&0x3)*2;
    GPOW_186_3(h1_tmp,(1u<<3)-c_PAHHHH,gw);

    SELECT(h1_tmp,gpp[189],c_PAHHHH==0,h1_tmp);//countM++;
  
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);  




   
    fp_copy(u2_temp_2,u2_tmp);
    for(int j=0;j<2;j++){fp_sqr(u2_temp_2,u2_temp_2);//countS++;
    }
   
    fp_prime_back(tmp_bn, u2_temp_2);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    
    uint64_t c_PAHHHL_H=rlll[d]>>1;
    
    
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_PAHHHL_H,gw);SELECT(h1_tmp,gpp[190],c_PAHHHL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAHH,u2_tmp,h1_tmp); 

   fp_prime_back(tmp_bn, h1_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_PAHHHL_L=rlll[d];

    uint64_t c_PAHHHL=c_PAHHHL_H+((c_PAHHHL_L-1)&3u)*2;

    uint64_t c_PAHHH=c_PAHHHH+((c_PAHHHL-1)&7u)*8;  


    
    GPOW_180_6(h1_PAHH,(1u<<6)-c_PAHHH,gw);
    SELECT(h1_PAHH,gpp[186],c_PAHHH==0,h1_PAHH);//countM++;
    fp_mul(u2_PAHH,h0_PAHH,h1_PAHH);

    

    
    fp_copy(h0_PAHHH,u2_PAHH);
    for(int j=0;j<3;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }
   
    fp_copy(h0_PAHLH,h0_PAHHH);
    for(int j=0;j<2;j++){fp_sqr(h0_PAHLH,h0_PAHLH);//countS++;
    }
   
    fp_prime_back(tmp_bn, h0_PAHLH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    
    uint64_t c_PAHHLH_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_PAHHLH_H,gw);SELECT(h1_tmp,gpp[190],c_PAHHLH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
   
    
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    uint64_t c_PAHHLH_L=rlll[d];
    
    uint64_t c_PAHHLH=c_PAHHLH_H+((c_PAHHLH_L-1)&3u)*2;
   
    GPOW_186_3(h1_tmp,(1u<<3)-c_PAHHLH,gw);SELECT(h1_tmp,gpp[189],c_PAHHLH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,u2_PAHH,h1_tmp); 
   

   
    fp_copy(sq_H,u2_tmp);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
   
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    uint64_t c_PAHHLL_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_PAHHLL_H,gw);SELECT(h1_tmp,gpp[190],c_PAHHLL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAH,u2_tmp,h1_tmp);
   fp_prime_back(tmp_bn, h1_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_PAHHLL_L=rlll[d];
   
    uint64_t c_PAHHLL=c_PAHHLL_H+((c_PAHHLL_L-1)&3u)*2;
   
    uint64_t c_PAHHL=c_PAHHLH+((c_PAHHLL-1)&7u)*8;
  
    uint64_t c_PAHH=c_PAHHH+((c_PAHHL-1)&63u)*64; 
   
    

   
    GPOW_168_12(h1_PAH,(1ull<<12)-c_PAHH,gw);SELECT(h1_PAH,gpp[180],c_PAHH==0,h1_PAH);//countM++;
   
    fp_mul(u2_PAH,h0_PAH,h1_PAH);
   

    
    fp_copy(h0_PAHH,u2_PAH);
    for(int j=0;j<6;j++){fp_sqr(h0_PAHH,h0_PAHH);//countS++;
    }
   

    fp_copy(h0_PAHHH,h0_PAHH);
    for(int j=0;j<3;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }
  
    
    fp_copy(u2_temp_3,h0_PAHHH);
    for(int j=0;j<2;j++){fp_sqr(u2_temp_3,u2_temp_3);//countS++;
    }
   
    fp_prime_back(tmp_bn, u2_temp_3);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
 
    uint64_t c_PAHLHH_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_PAHLHH_H,gw);SELECT(h1_tmp,gpp[190],c_PAHLHH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
 
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_PAHLHH_L=rlll[d];
    
    uint64_t c_PAHLHH=c_PAHLHH_H+((c_PAHLHH_L-1)&3u)*2;
 
    
    
    
    GPOW_186_3(h1_tmp,(1u<<3)-c_PAHLHH,gw);SELECT(h1_tmp,gpp[189],c_PAHLHH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHH,h1_tmp);

    
    
    fp_copy(sq_H,u2_tmp);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_PAHLHL_H=rlll[d]>>1;

    
    
    
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_PAHLHL_H,gw);SELECT(h1_tmp,gpp[190],c_PAHLHL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAHH,u2_tmp,h1_tmp);
    fp_prime_back(tmp_bn, h1_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_PAHLHL_L=rlll[d];
  
    uint64_t c_PAHLHL=c_PAHLHL_H+((c_PAHLHL_L-1)&3u)*2;
    uint64_t c_PAHLH=c_PAHLHH+((c_PAHLHL-1)&7u)*8;

    








    GPOW_180_6(h1_PAHH,(1u<<6)-c_PAHLH,gw);SELECT(h1_PAHH,gpp[186],c_PAHLH==0,h1_PAHH);//countM++;
    fp_mul(u2_PAHH,u2_PAH,h1_PAHH);  /* u2_PAHL */

    fp_copy(h0_PAHHH,u2_PAHH);
    for(int j=0;j<3;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }
    fp_copy(sq_H,h0_PAHHH);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_PAHLLH_H=rlll[d]>>1;
  
    
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_PAHLLH_H,gw);SELECT(h1_tmp,gpp[190],c_PAHLLH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_PAHLLH_L=rlll[d];

    uint64_t c_PAHLLH=c_PAHLLH_H+((c_PAHLLH_L-1)&3u)*2;

    
    
    GPOW_186_3(h1_tmp,(1u<<3)-c_PAHLLH,gw);SELECT(h1_tmp,gpp[189],c_PAHLLH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,u2_PAHH,h1_tmp);
    fp_copy(sq_H,u2_tmp);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_PAHLLL_H=rlll[d]>>1;
  
    
    
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_PAHLLL_H,gw);SELECT(h1_tmp,gpp[190],c_PAHLLL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAH,u2_tmp,h1_tmp);
    fp_prime_back(tmp_bn, h1_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_PAHLLL_L=rlll[d];
    
    uint64_t c_PAHLLL=c_PAHLLL_H+((c_PAHLLL_L-1)&3u)*2;
    uint64_t c_PAHLL=c_PAHLLH+((c_PAHLLL-1)&7u)*8;
    uint64_t c_PAHL=c_PAHLH+((c_PAHLL-1)&63u)*64;  
    uint64_t c_PAH=c_PAHH+((c_PAHL-1)&0xFFFull)*(1ull<<12);  
   




    
    GPOW_144_24(h1_PA,((1ull<<24)-c_PAH)&0xffffff,gw);
    SELECT(h1_PA,gpp[168],c_PAH==0,h1_PA);//countM++;

    fp_mul(u2_PA,h0_PA,h1_PA);


    
    fp_copy(h0_PAH,u2_PA);
    for(int j=0;j<12;j++){fp_sqr(h0_PAH,h0_PAH);//countS++;
    }
   

    fp_copy(h0_PAHH,h0_PAH);
    for(int j=0;j<6;j++){fp_sqr(h0_PAHH,h0_PAHH);//countS++;
    }


    fp_copy(h0_PAHHH,h0_PAHH);
    for(int j=0;j<3;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }

    
    
    fp_copy(sq_H,h0_PAHHH);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;

    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_PALHHH_H=rlll[d]>>1;

    
    
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_PALHHH_H,gw);SELECT(h1_tmp,gpp[190],c_PALHHH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_PALHHH_L=rlll[d];
    uint64_t c_PALHHH=c_PALHHH_H+((c_PALHHH_L-1)&3u)*2;
 
    GPOW_186_3(h1_tmp,(1u<<3)-c_PALHHH,gw);SELECT(h1_tmp,gpp[189],c_PALHHH==0,h1_tmp);//countM++;
   
    fp_mul(u2_tmp,h0_PAHH,h1_tmp);
    fp_copy(sq_H,u2_tmp);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
 
    uint64_t c_PALHHL_H=rlll[d]>>1;
    GPOW_189_1(h1_tmp,(1u<<1)-c_PALHHL_H,gw);SELECT(h1_tmp,gpp[190],c_PALHHL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAHH,u2_tmp,h1_tmp);
    fp_prime_back(tmp_bn, h1_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_PALHHL_L=rlll[d];
   
    uint64_t c_PALHHL=c_PALHHL_H+((c_PALHHL_L-1)&3u)*2;
    uint64_t c_PALHH=c_PALHHH+((c_PALHHL-1)&7u)*8;
   
    



    GPOW_180_6(h1_PAHH,(1u<<6)-c_PALHH,gw);SELECT(h1_PAHH,gpp[186],c_PALHH==0,h1_PAHH);//countM++;
    fp_mul(u2_PAHH,h0_PAH,h1_PAHH);  /* u2_PALH */
  

    
    fp_copy(h0_PAHHH,u2_PAHH);
    for(int j=0;j<3;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }
    fp_copy(sq_H,h0_PAHHH);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
   
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_PALHLH_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_PALHLH_H,gw);SELECT(h1_tmp,gpp[190],c_PALHLH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
 
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_PALHLH_L=rlll[d];
    
    uint64_t c_PALHLH=c_PALHLH_H+((c_PALHLH_L-1)&3u)*2;
    GPOW_186_3(h1_tmp,(1u<<3)-c_PALHLH,gw);SELECT(h1_tmp,gpp[189],c_PALHLH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,u2_PAHH,h1_tmp);
    fp_copy(sq_H,u2_tmp);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
   
   fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
 
    uint64_t c_PALHLL_H=rlll[d]>>1;
   
    GPOW_189_1(h1_tmp,(1u<<1)-c_PALHLL_H,gw);SELECT(h1_tmp,gpp[190],c_PALHLL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAH,u2_tmp,h1_tmp);

    fp_prime_back(tmp_bn, h1_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_PALHLL_L=rlll[d];

    uint64_t c_PALHLL=c_PALHLL_H+((c_PALHLL_L-1)&3u)*2;
    uint64_t c_PALHL=c_PALHLH+((c_PALHLL-1)&7u)*8;
    uint64_t c_PALH=c_PALHH+((c_PALHL-1)&63u)*64;  /* 12-bit */

    
    
    

    GPOW_168_12(h1_PAH,(1ull<<12)-c_PALH,gw);SELECT(h1_PAH,gpp[180],c_PALH==0,h1_PAH);//countM++;
    fp_mul(u2_PAH,u2_PA,h1_PAH);  /* u2_PAL */
  
    fp_copy(h0_PAHH,u2_PAH);
    for(int j=0;j<6;j++){fp_sqr(h0_PAHH,h0_PAHH);//countS++;
    }

 
    fp_copy(h0_PAHHH,h0_PAHH);
    for(int j=0;j<3;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }
    fp_copy(sq_H,h0_PAHHH);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_PALLHH_H=rlll[d]>>1;
  
    GPOW_189_1(h1_tmp,(1u<<1)-c_PALLHH_H,gw);SELECT(h1_tmp,gpp[190],c_PALLHH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_PALLHH_L=rlll[d];

    uint64_t c_PALLHH=c_PALLHH_H+((c_PALLHH_L-1)&3u)*2;

    
    
    
    GPOW_186_3(h1_tmp,(1u<<3)-c_PALLHH,gw);SELECT(h1_tmp,gpp[189],c_PALLHH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHH,h1_tmp);

    fp_copy(sq_H,u2_tmp);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_PALLHL_H=rlll[d]>>1;

    GPOW_189_1(h1_tmp,(1u<<1)-c_PALLHL_H,gw);SELECT(h1_tmp,gpp[190],c_PALLHL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAHH,u2_tmp,h1_tmp);
    fp_prime_back(tmp_bn, h1_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_PALLHL_L=rlll[d];

    uint64_t c_PALLHL=c_PALLHL_H+((c_PALLHL_L-1)&3u)*2;
    uint64_t c_PALLH=c_PALLHH+((c_PALLHL-1)&7u)*8;





    GPOW_180_6(h1_PAHH,(1u<<6)-c_PALLH,gw);SELECT(h1_PAHH,gpp[186],c_PALLH==0,h1_PAHH);//countM++;
    fp_mul(u2_PAHH,u2_PAH,h1_PAHH);  /* u2_PALL */


    fp_copy(h0_PAHHH,u2_PAHH);
    for(int j=0;j<3;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }
    fp_copy(sq_H,h0_PAHHH);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_PALLLH_H=rlll[d]>>1;
 
    GPOW_189_1(h1_tmp,(1u<<1)-c_PALLLH_H,gw);SELECT(h1_tmp,gpp[190],c_PALLLH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_PALLLH_L=rlll[d];
  
    uint64_t c_PALLLH=c_PALLLH_H+((c_PALLLH_L-1)&3u)*2;
    GPOW_186_3(h1_tmp,(1u<<3)-c_PALLLH,gw);SELECT(h1_tmp,gpp[189],c_PALLLH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,u2_PAHH,h1_tmp);
    fp_copy(sq_H,u2_tmp);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
 
    uint64_t c_PALLLL_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_PALLLL_H,gw);SELECT(h1_tmp,gpp[190],c_PALLLL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAH,u2_tmp,h1_tmp);
    fp_prime_back(tmp_bn, h1_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_PALLLL_L=rlll[d];
    
    uint64_t c_PALLLL=c_PALLLL_H+((c_PALLLL_L-1)&3u)*2;
  
    uint64_t c_PALLL=c_PALLLH+((c_PALLLL-1)&7u)*8;
  
    uint64_t c_PALL=c_PALLH+((c_PALLL-1)&63u)*64;  /* 12-bit */
  
    uint64_t c_PAL=c_PALH+((c_PALL-1)&0xFFFull)*(1ull<<12);  /* 48-bit DLP144 result */
  
    uint64_t c_PA=c_PAH+((c_PAL-1)&0xFFFFFFull)*(1ull<<24);  /* 48-bit DLP144 result */
   
    
    
    
    GPOW_96_48(f_A, ((1ull<<48)-c_PA)&MASK48, gw);
   
    SELECT(f_A, gpp[144], c_PA==0, f_A);
    
    fp_mul(u_X0, f_A, h0_A);   // h0_A = sq96(u) = Sage's h0_TOP
   
    


    fp_copy(h0_X0,u_X0);
    for(int j=0;j<24;j++){fp_sqr(h0_X0,h0_X0);//countS++;
    }

    
    fp_copy(h0_PAH,h0_X0);
    for(int j=0;j<12;j++){fp_sqr(h0_PAH,h0_PAH);//countS++;
    }
    fp_copy(h0_PAHH,h0_PAH);
    for(int j=0;j<6;j++){fp_sqr(h0_PAHH,h0_PAHH);//countS++;
    }
    fp_copy(h0_PAHHH,h0_PAHH);
    for(int j=0;j<3;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }
    fp_copy(sq_H,h0_PAHHH);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t cX0PHHH_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-cX0PHHH_H,gw);SELECT(h1_tmp,gpp[190],cX0PHHH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t cX0PHHH_L=rlll[d];
    
    uint64_t cX0PHHH=cX0PHHH_H+((cX0PHHH_L-1)&3u)*2;
    GPOW_186_3(h1_tmp,(1u<<3)-cX0PHHH,gw);SELECT(h1_tmp,gpp[189],cX0PHHH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHH,h1_tmp);
    fp_copy(sq_H,u2_tmp);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t cX0PHHL_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-cX0PHHL_H,gw);SELECT(h1_tmp,gpp[190],cX0PHHL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAHH,u2_tmp,h1_tmp);
    fp_prime_back(tmp_bn, h1_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t cX0PHHL_L=rlll[d];
    
    uint64_t cX0PHHL=cX0PHHL_H+((cX0PHHL_L-1)&3u)*2;
    uint64_t cX0PHH=cX0PHHH+((cX0PHHL-1)&7u)*8;
   
    
    
    GPOW_180_6(h1_PAHH,(1u<<6)-cX0PHH,gw);SELECT(h1_PAHH,gpp[186],cX0PHH==0,h1_PAHH);//countM++;
    fp_mul(u2_PAHH,h0_PAH,h1_PAHH);
    fp_copy(h0_PAHHH,u2_PAHH);
    for(int j=0;j<3;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }
    fp_copy(sq_H,h0_PAHHH);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t cX0PHLH_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-cX0PHLH_H,gw);SELECT(h1_tmp,gpp[190],cX0PHLH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t cX0PHLH_L=rlll[d];
    
    uint64_t cX0PHLH=cX0PHLH_H+((cX0PHLH_L-1)&3u)*2;
    GPOW_186_3(h1_tmp,(1u<<3)-cX0PHLH,gw);SELECT(h1_tmp,gpp[189],cX0PHLH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,u2_PAHH,h1_tmp);
    fp_copy(sq_H,u2_tmp);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t cX0PHLL_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-cX0PHLL_H,gw);SELECT(h1_tmp,gpp[190],cX0PHLL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAH,u2_tmp,h1_tmp);
    fp_prime_back(tmp_bn, h1_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t cX0PHLL_L=rlll[d];
    
    uint64_t cX0PHLL=cX0PHLL_H+((cX0PHLL_L-1)&3u)*2;
    uint64_t cX0PHL=cX0PHLH+((cX0PHLL-1)&7u)*8;
    uint64_t cX0PH=cX0PHH+((cX0PHL-1)&63u)*64;  /* 12-bit DLP168 H result */
   
    
    
    GPOW_168_12(h1_PAH,(1ull<<12)-cX0PH,gw);SELECT(h1_PAH,gpp[180],cX0PH==0,h1_PAH);//countM++;
    fp_mul(u2_PAH,h0_X0,h1_PAH);
    
    fp_copy(h0_PAHH,u2_PAH);
    for(int j=0;j<6;j++){fp_sqr(h0_PAHH,h0_PAHH);//countS++;
    }
    fp_copy(h0_PAHHH,h0_PAHH);
    for(int j=0;j<3;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }
    fp_copy(sq_H,h0_PAHHH);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t cX0PLHH_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-cX0PLHH_H,gw);SELECT(h1_tmp,gpp[190],cX0PLHH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
  
    uint64_t cX0PLHH_L=rlll[d];
    
    uint64_t cX0PLHH=cX0PLHH_H+((cX0PLHH_L-1)&3u)*2;
    GPOW_186_3(h1_tmp,(1u<<3)-cX0PLHH,gw);SELECT(h1_tmp,gpp[189],cX0PLHH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHH,h1_tmp);
    fp_copy(sq_H,u2_tmp);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
 
    uint64_t cX0PLHL_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-cX0PLHL_H,gw);SELECT(h1_tmp,gpp[190],cX0PLHL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAHH,u2_tmp,h1_tmp);
    fp_prime_back(tmp_bn, h1_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t cX0PLHL_L=rlll[d];
    
    uint64_t cX0PLHL=cX0PLHL_H+((cX0PLHL_L-1)&3u)*2;
    uint64_t cX0PLH=cX0PLHH+((cX0PLHL-1)&7u)*8;
    
 
    
    GPOW_180_6(h1_PAHH,(1u<<6)-cX0PLH,gw);SELECT(h1_PAHH,gpp[186],cX0PLH==0,h1_PAHH);//countM++;
    fp_mul(u2_PAHH,u2_PAH,h1_PAHH);
    fp_copy(h0_PAHHH,u2_PAHH);
    for(int j=0;j<3;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }
    fp_copy(sq_H,h0_PAHHH);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
 
    uint64_t cX0PLLH_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-cX0PLLH_H,gw);SELECT(h1_tmp,gpp[190],cX0PLLH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t cX0PLLH_L=rlll[d];
    
    uint64_t cX0PLLH=cX0PLLH_H+((cX0PLLH_L-1)&3u)*2;
    GPOW_186_3(h1_tmp,(1u<<3)-cX0PLLH,gw);SELECT(h1_tmp,gpp[189],cX0PLLH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,u2_PAHH,h1_tmp);
    fp_copy(sq_H,u2_tmp);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t cX0PLLL_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-cX0PLLL_H,gw);SELECT(h1_tmp,gpp[190],cX0PLLL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAH,u2_tmp,h1_tmp);
    fp_prime_back(tmp_bn, h1_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t cX0PLLL_L=rlll[d];

    uint64_t cX0PLLL=cX0PLLL_H+((cX0PLLL_L-1)&3u)*2;
    uint64_t cX0PLL=cX0PLLH+((cX0PLLL-1)&7u)*8;
    uint64_t cX0PL=cX0PLH+((cX0PLL-1)&63u)*64;  /* 12-bit */
    uint64_t c_X0=cX0PH+((cX0PL-1)&0xFFFull)*(1ull<<12);  /* 48-bit DLP144 EXT0 result */
    

  
    GPOW_144_24(h1_X0,((1ull<<24)-c_X0)&0xffffff,gw);SELECT(h1_X0,gpp[168],c_X0==0,h1_X0);//countM++;
    fp_mul(u_X1,u_X0,h1_X0);

    
    fp_copy(h0_X1,u_X1);
    for(int j=0;j<12;j++){fp_sqr(h0_X1,h0_X1);//countS++;
    }
   
    fp_copy(h0_PAHH,h0_X1);
    for(int j=0;j<6;j++){fp_sqr(h0_PAHH,h0_PAHH);//countS++;
    }
    fp_copy(h0_PAHHH,h0_PAHH);
    for(int j=0;j<3;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }
    fp_copy(sq_H,h0_PAHHH);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
 
    uint64_t cX1HHH_H=rlll[d]>>1;

    GPOW_189_1(h1_tmp,(1u<<1)-cX1HHH_H,gw);SELECT(h1_tmp,gpp[190],cX1HHH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t cX1HHH_L=rlll[d];
    
    uint64_t cX1HHH=cX1HHH_H+((cX1HHH_L-1)&3u)*2;
    GPOW_186_3(h1_tmp,(1u<<3)-cX1HHH,gw);SELECT(h1_tmp,gpp[189],cX1HHH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHH,h1_tmp);
    fp_copy(sq_H,u2_tmp);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t cX1HHL_H=rlll[d]>>1;
   
    GPOW_189_1(h1_tmp,(1u<<1)-cX1HHL_H,gw);SELECT(h1_tmp,gpp[190],cX1HHL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAHH,u2_tmp,h1_tmp);
    fp_prime_back(tmp_bn, h1_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t cX1HHL_L=rlll[d];
    
    uint64_t cX1HHL=cX1HHL_H+((cX1HHL_L-1)&3u)*2;
    uint64_t cX1HH=cX1HHH+((cX1HHL-1)&7u)*8;
   
    
    
    
    
    GPOW_180_6(h1_PAHH,(1u<<6)-cX1HH,gw);SELECT(h1_PAHH,gpp[186],cX1HH==0,h1_PAHH);//countM++;
    fp_mul(u2_PAHH,h0_X1,h1_PAHH);
    fp_copy(h0_PAHHH,u2_PAHH);
    for(int j=0;j<3;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }
    fp_copy(sq_H,h0_PAHHH);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
  
    uint64_t cX1HLH_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-cX1HLH_H,gw);SELECT(h1_tmp,gpp[190],cX1HLH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t cX1HLH_L=rlll[d];
    
    uint64_t cX1HLH=cX1HLH_H+((cX1HLH_L-1)&3u)*2;
    GPOW_186_3(h1_tmp,(1u<<3)-cX1HLH,gw);SELECT(h1_tmp,gpp[189],cX1HLH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,u2_PAHH,h1_tmp);
    fp_copy(sq_H,u2_tmp);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t cX1HLL_H=rlll[d]>>1;
   
    GPOW_189_1(h1_tmp,(1u<<1)-cX1HLL_H,gw);SELECT(h1_tmp,gpp[190],cX1HLL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAH,u2_tmp,h1_tmp);
    fp_prime_back(tmp_bn, h1_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
 
    uint64_t cX1HLL_L=rlll[d];
    
    uint64_t cX1HLL=cX1HLL_H+((cX1HLL_L-1)&3u)*2;
    uint64_t cX1HL=cX1HLH+((cX1HLL-1)&7u)*8;
    uint64_t c_X1=cX1HH+((cX1HL-1)&0x3f)*64;  /* 24-bit DLP168 EXT1 result */
    
 

    GPOW_168_12(h1_X1,(1ull<<12)-c_X1,gw);SELECT(h1_X1,gpp[180],c_X1==0,h1_X1);//countM++;
    fp_mul(u_X2,u_X1,h1_X1);

    fp_copy(h0_X2,u_X2);
    for(int j=0;j<6;j++){fp_sqr(h0_X2,h0_X2);//countS++;
    }
    fp_copy(h0_PAHHH,h0_X2);
    for(int j=0;j<3;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }
    fp_copy(sq_H,h0_PAHHH);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t cX2HH_H=rlll[d]>>1;
  
    GPOW_189_1(h1_tmp,(1u<<1)-cX2HH_H,gw);SELECT(h1_tmp,gpp[190],cX2HH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t cX2HH_L=rlll[d];
   
    uint64_t cX2HH=cX2HH_H+((cX2HH_L-1)&3u)*2;
    GPOW_186_3(h1_tmp,(1u<<3)-cX2HH,gw);SELECT(h1_tmp,gpp[189],cX2HH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_X2,h1_tmp);
    fp_copy(sq_H,u2_tmp);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t cX2HL_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-cX2HL_H,gw);SELECT(h1_tmp,gpp[190],cX2HL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAHH,u2_tmp,h1_tmp);
    fp_prime_back(tmp_bn, h1_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t cX2HL_L=rlll[d];
    
    uint64_t cX2HL=cX2HL_H+((cX2HL_L-1)&0x3)*2;
    uint64_t c_X2=cX2HH+((cX2HL-1)&0x7)*8;  /* 6-bit */
   
    
    

    GPOW_180_6(h1_X2,(1u<<6)-c_X2,gw);SELECT(h1_X2,gpp[186],c_X2==0,h1_X2);//countM++;
    fp_mul(u_X3,u_X2,h1_X2);
    fp_copy(h0_X3,u_X3);
    for(int j=0;j<3;j++){fp_sqr(h0_X3,h0_X3);//countS++;
    }
    fp_copy(sq_H,h0_X3);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t cX3H_H=rlll[d]>>1;

    GPOW_189_1(h1_tmp,(1u<<1)-cX3H_H,gw);SELECT(h1_tmp,gpp[190],cX3H_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_X3,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t cX3H_L=rlll[d];

    uint64_t c_X3=cX3H_H+((cX3H_L-1)&3u)*2;  /* 3-bit: DLP186 result = 6-bit? */

    

    GPOW_186_3(h1_tmp,(1u<<3)-c_X3,gw);

    SELECT(h1_tmp,gpp[189],c_X3==0,h1_tmp);//countM++;

    fp_mul(u2_tmp,u_X3,h1_tmp);

    fp_copy(sq_H,u2_tmp);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_prime_back(tmp_bn, sq_H);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
  
    uint64_t cX3L_H=rlll[d]>>1;
 
    GPOW_189_1(h1_tmp,(1u<<1)-cX3L_H,gw);SELECT(h1_tmp,gpp[190],cX3L_H==0,h1_tmp);//countM++;
    fp_mul(h1_X3,u2_tmp,h1_tmp);
    fp_prime_back(tmp_bn, h1_X3);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t cX3L_L=rlll[d];

    uint64_t cX3L=cX3L_H+((cX3L_L-1)&3u)*2;

    uint64_t c_X3_6=c_X3+((cX3L-1)&7u)*8;  

    uint64_t cX0PHL12=c_X2+((c_X3_6-1)&63u)*64; 

    uint64_t cX0PL24=c_X1+((cX0PHL12-1)&0xFFFull)*4096;  /* 24-bit DLP168 L result */
 
    uint64_t c_PAL_1=c_X0+((cX0PL24-1)&0xFFFFFFull)*(1ull<<24);  /* 48-bit DLP144 EXT0 */
   
    
    uint64_t dp1=(c_PAL_1-1)&MASK48;  /* (c_TOP_H_L - 1) mod 2^48 */
    uint120_t c_A;

     
    c_A.low  = (c_PA | ((dp1 & 0xFFFull) << 48)) & LIMB_MASK;
    c_A.high = (dp1 >> 12) & LIMB_MASK;
   
    uint120_t exp_fA=rshift120(add120(sub120(POW96,c_A),make120(0,1)),1);
    GPOW_0_95(f_A,exp_fA,gw);
    bool cA0=(c_A.low==0&&c_A.high==0);

    SELECT(f_A,gpp[95],cA0,f_A);

    //countM++;
    fp_sqr(u_X0,f_A);//countS++;
    fp_mul(u_X0,u_X0,u_A);
  
    
    fp_copy(h0_X0,u_X0);
    for(int j=0;j<48;j++){fp_sqr(h0_X0,h0_X0);//countS++;
    }

    fp_copy(h0_PAH,h0_X0);
    for(int j=0;j<24;j++){fp_sqr(h0_PAH,h0_PAH);//countS++;
    }
    fp_copy(h0_PAHH,h0_PAH);
    for(int j=0;j<12;j++){fp_sqr(h0_PAHH,h0_PAHH);//countS++;
    }

   
    fp_copy(h0_PAHHH,h0_PAHH);
    for(int j=0;j<6;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }

  
    fp_copy(sq_H,h0_PAHHH);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_copy(u2_tmp,sq_H);fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_X0HHHH_H=rlll[d]>>1;
    
    
    
    
    
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X0HHHH_H,gw);SELECT(h1_tmp,gpp[190],c_X0HHHH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,sq_H,h1_tmp);
  
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_X0HHHH_L=rlll[d];
    
    uint64_t c_X0HHHH=c_X0HHHH_H+((c_X0HHHH_L-1)&3u)*2;
  
    
    

    
    GPOW_186_3(h1_tmp,(1u<<3)-c_X0HHHH,gw);SELECT(h1_tmp,gpp[189],c_X0HHHH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
   

    
    fp_copy(h1_PAHH,u2_tmp);fp_sqr(h1_PAHH,h1_PAHH);//countS++;
    fp_sqr(h1_PAHH,h1_PAHH);//countS++;
   //fp_print(h1_PAHH);
   fp_prime_back(tmp_bn, h1_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_X0HHHL_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X0HHHL_H,gw);SELECT(h1_tmp,gpp[190],c_X0HHHL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAHH,u2_tmp,h1_tmp);
   
    fp_prime_back(tmp_bn, h1_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_X0HHHL_L=rlll[d];
    
    uint64_t c_X0HHHL=c_X0HHHL_H+((c_X0HHHL_L-1)&3u)*2;
   
    uint64_t c_X0HHH=c_X0HHHH+((c_X0HHHL-1)&7u)*8;  /* 6-bit DLP186 */
   
    
    
    
    
    
    
    
    GPOW_180_6(h1_PAHH,(1u<<6)-c_X0HHH,gw);SELECT(h1_PAHH,gpp[186],c_X0HHH==0,h1_PAHH);//countM++;
    fp_mul(u2_PAHH,h0_PAHH,h1_PAHH);
   

    
    fp_copy(h0_PAHHH,u2_PAHH);
    for(int j=0;j<3;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }
    
    fp_copy(u2_tmp,h0_PAHHH);fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    uint64_t c_X0HHLH_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X0HHLH_H,gw);SELECT(h1_tmp,gpp[190],c_X0HHLH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
    
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_X0HHLH_L=rlll[d];
    
    uint64_t c_X0HHLH=c_X0HHLH_H+((c_X0HHLH_L-1)&3u)*2;
   
    GPOW_186_3(h1_tmp,(1u<<3)-c_X0HHLH,gw);SELECT(h1_tmp,gpp[189],c_X0HHLH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,u2_PAHH,h1_tmp);
    
    
    fp_copy(h1_PAH,u2_tmp);fp_sqr(h1_PAH,h1_PAH);//countS++;
    fp_sqr(h1_PAH,h1_PAH);//countS++;
    fp_prime_back(tmp_bn, h1_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_X0HHLL_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X0HHLL_H,gw);SELECT(h1_tmp,gpp[190],c_X0HHLL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAH,u2_tmp,h1_tmp);
   
    fp_prime_back(tmp_bn, h1_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_X0HHLL_L=rlll[d];

    uint64_t c_X0HHLL=c_X0HHLL_H+((c_X0HHLL_L-1)&3u)*2;

    uint64_t c_X0HHL=c_X0HHLH+((c_X0HHLL-1)&7u)*8; 

    uint64_t c_X0HH=c_X0HHH+((c_X0HHL-1)&63u)*64; 
   
    
    
    
    

    
    GPOW_168_12(h1_PAH,(1ull<<12)-c_X0HH,gw);SELECT(h1_PAH,gpp[180],c_X0HH==0,h1_PAH);//countM++;
   
    fp_mul(u2_PAH,h0_PAH,h1_PAH);
   
    fp_copy(h0_PAHHH,u2_PAH);
    for(int j=0;j<6;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }

    
    fp_copy(sq_H,h0_PAHHH);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_copy(u2_tmp,sq_H);fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_X0HLHH_H=rlll[d]>>1;
   
    GPOW_189_1(h1_tmp,(1u<<1)-c_X0HLHH_H,gw);SELECT(h1_tmp,gpp[190],c_X0HLHH_H==0,h1_tmp);//countM++;
   
    fp_mul(u2_tmp,sq_H,h1_tmp);
   
   fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_X0HLHH_L=rlll[d];
    
    uint64_t c_X0HLHH=c_X0HLHH_H+((c_X0HLHH_L-1)&3u)*2;
    GPOW_186_3(h1_tmp,(1u<<3)-c_X0HLHH,gw);SELECT(h1_tmp,gpp[189],c_X0HLHH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
  

    
    fp_copy(h1_PAHH,u2_tmp);fp_sqr(h1_PAHH,h1_PAHH);//countS++;
    fp_sqr(h1_PAHH,h1_PAHH);//countS++;
    fp_prime_back(tmp_bn, h1_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_X0HLHL_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X0HLHL_H,gw);SELECT(h1_tmp,gpp[190],c_X0HLHL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAHH,u2_tmp,h1_tmp);
    fp_prime_back(tmp_bn, h1_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_X0HLHL_L=rlll[d];
   
    uint64_t c_X0HLHL=c_X0HLHL_H+((c_X0HLHL_L-1)&3u)*2;
   
    uint64_t c_X0HLH=c_X0HLHH+((c_X0HLHL-1)&7u)*8;  
   
    

    
    GPOW_180_6(h1_PAHH,(1u<<6)-c_X0HLH,gw);SELECT(h1_PAHH,gpp[186],c_X0HLH==0,h1_PAHH);//countM++;
    fp_mul(u2_PAHH,u2_PAH,h1_PAHH);

    fp_copy(h0_PAHHH,u2_PAHH);
    for(int j=0;j<3;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }
    fp_copy(u2_tmp,h0_PAHHH);fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_X0HLLH_H=rlll[d]>>1;

    GPOW_189_1(h1_tmp,(1u<<1)-c_X0HLLH_H,gw);SELECT(h1_tmp,gpp[190],c_X0HLLH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);

   fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_X0HLLH_L=rlll[d];
 
    uint64_t c_X0HLLH=c_X0HLLH_H+((c_X0HLLH_L-1)&3u)*2;

    
    
    
    GPOW_186_3(h1_tmp,(1u<<3)-c_X0HLLH,gw);SELECT(h1_tmp,gpp[189],c_X0HLLH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,u2_PAHH,h1_tmp);
 
    fp_copy(h1_PAH,u2_tmp);fp_sqr(h1_PAH,h1_PAH);//countS++;
    fp_sqr(h1_PAH,h1_PAH);//countS++;
    fp_prime_back(tmp_bn, h1_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_X0HLLL_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X0HLLL_H,gw);SELECT(h1_tmp,gpp[190],c_X0HLLL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAH,u2_tmp,h1_tmp);
   fp_prime_back(tmp_bn, h1_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_X0HLLL_L=rlll[d];
    
    uint64_t c_X0HLLL=c_X0HLLL_H+((c_X0HLLL_L-1)&3u)*2;
    
    uint64_t c_X0HLL=c_X0HLLH+((c_X0HLLL-1)&7u)*8; 
    
    uint64_t c_X0HL=c_X0HLH+((c_X0HLL-1)&63u)*64;  
    
    uint64_t c_X0H=c_X0HH+((c_X0HL-1)&0xFFFull)*4096;  
   
    
    
    
    
    GPOW_144_24(h1_PAH,(1ull<<24)-c_X0H,gw);SELECT(h1_PAH,gpp[168],c_X0H==0,h1_PAH);//countM++;
    fp_mul(u2_PAH,h0_X0,h1_PAH);

    
    fp_copy(h0_PAHH,u2_PAH);
    for(int j=0;j<12;j++){fp_sqr(h0_PAHH,h0_PAHH);//countS++;
    }

    
    fp_copy(h0_PAHHH,h0_PAHH);
    for(int j=0;j<6;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }

    fp_copy(sq_H,h0_PAHHH);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_copy(u2_tmp,sq_H);fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_X0LHHH_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X0LHHH_H,gw);SELECT(h1_tmp,gpp[190],c_X0LHHH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,sq_H,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_X0LHHH_L=rlll[d];
    
    uint64_t c_X0LHHH=c_X0LHHH_H+((c_X0LHHH_L-1)&3u)*2;
   
    
    
    
    GPOW_186_3(h1_tmp,(1u<<3)-c_X0LHHH,gw);SELECT(h1_tmp,gpp[189],c_X0LHHH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);

    
    fp_copy(h1_PAHH,u2_tmp);fp_sqr(h1_PAHH,h1_PAHH);//countS++;
    fp_sqr(h1_PAHH,h1_PAHH);//countS++;
    fp_prime_back(tmp_bn, h1_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_X0LHHL_H=rlll[d]>>1;
    
    
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X0LHHL_H,gw);SELECT(h1_tmp,gpp[190],c_X0LHHL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAHH,u2_tmp,h1_tmp);
    fp_prime_back(tmp_bn, h1_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    uint64_t c_X0LHHL_L=rlll[d];
    
    uint64_t c_X0LHHL=c_X0LHHL_H+((c_X0LHHL_L-1)&3u)*2;
    uint64_t c_X0LHH=c_X0LHHH+((c_X0LHHL-1)&7u)*8;  /* 6-bit */
   
    
    
    GPOW_180_6(h1_PAHH,(1u<<6)-c_X0LHH,gw);SELECT(h1_PAHH,gpp[186],c_X0LHH==0,h1_PAHH);//countM++;
    fp_mul(u2_PAHH,h0_PAHH,h1_PAHH);

    fp_copy(h0_PAHHH,u2_PAHH);
    for(int j=0;j<3;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }

    fp_copy(u2_tmp,h0_PAHHH);fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    uint64_t c_X0LHLH_H=rlll[d]>>1;
    GPOW_189_1(h1_tmp,(1u<<1)-c_X0LHLH_H,gw);SELECT(h1_tmp,gpp[190],c_X0LHLH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_X0LHLH_L=rlll[d];
    uint64_t c_X0LHLH=c_X0LHLH_H+((c_X0LHLH_L-1)&3u)*2;
    GPOW_186_3(h1_tmp,(1u<<3)-c_X0LHLH,gw);SELECT(h1_tmp,gpp[189],c_X0LHLH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,u2_PAHH,h1_tmp);

    
    fp_copy(h1_PAH,u2_tmp);fp_sqr(h1_PAH,h1_PAH);//countS++;
    fp_sqr(h1_PAH,h1_PAH);//countS++;
    fp_prime_back(tmp_bn, h1_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    uint64_t c_X0LHLL_H=rlll[d]>>1;
    GPOW_189_1(h1_tmp,(1u<<1)-c_X0LHLL_H,gw);SELECT(h1_tmp,gpp[190],c_X0LHLL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAH,u2_tmp,h1_tmp);
   
    fp_prime_back(tmp_bn, h1_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    uint64_t c_X0LHLL_L=rlll[d];
    uint64_t c_X0LHLL=c_X0LHLL_H+((c_X0LHLL_L-1)&3u)*2;
    uint64_t c_X0LHL=c_X0LHLH+((c_X0LHLL-1)&7u)*8;
    uint64_t c_X0LH=c_X0LHH+((c_X0LHL-1)&63u)*64;
   
    

    GPOW_168_12(h1_PAH,(1ull<<12)-c_X0LH,gw);SELECT(h1_PAH,gpp[180],c_X0LH==0,h1_PAH);//countM++;
    fp_mul(u2_PAH,u2_PAH,h1_PAH);

    
    fp_copy(h0_PAHHH,u2_PAH);
    for(int j=0;j<6;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }

    fp_copy(sq_H,h0_PAHHH);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_copy(u2_tmp,sq_H);fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    uint64_t c_X0LLHH_H=rlll[d]>>1;
    
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X0LLHH_H,gw);SELECT(h1_tmp,gpp[190],c_X0LLHH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,sq_H,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    uint64_t c_X0LLHH_L=rlll[d];
    uint64_t c_X0LLHH=c_X0LLHH_H+((c_X0LLHH_L-1)&3u)*2;
   
    
    
    GPOW_186_3(h1_tmp,(1u<<3)-c_X0LLHH,gw);SELECT(h1_tmp,gpp[189],c_X0LLHH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);

    
    fp_copy(h1_PAHH,u2_tmp);fp_sqr(h1_PAHH,h1_PAHH);//countS++;
    fp_sqr(h1_PAHH,h1_PAHH);//countS++;
    fp_prime_back(tmp_bn, h1_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    uint64_t c_X0LLHL_H=rlll[d]>>1;
    
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X0LLHL_H,gw);SELECT(h1_tmp,gpp[190],c_X0LLHL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAHH,u2_tmp,h1_tmp);
    fp_prime_back(tmp_bn, h1_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    uint64_t c_X0LLHL_L=rlll[d];
    uint64_t c_X0LLHL=c_X0LLHL_H+((c_X0LLHL_L-1)&3u)*2;
    uint64_t c_X0LLH=c_X0LLHH+((c_X0LLHL-1)&7u)*8;  

    GPOW_180_6(h1_PAHH,(1u<<6)-c_X0LLH,gw);SELECT(h1_PAHH,gpp[186],c_X0LLH==0,h1_PAHH);//countM++;
    fp_mul(u2_PAHH,u2_PAH,h1_PAHH);

    fp_copy(h0_PAHHH,u2_PAHH);
    for(int j=0;j<3;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }

    
    fp_copy(u2_tmp,h0_PAHHH);fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    uint64_t c_X0LLLH_H=rlll[d]>>1;
    
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X0LLLH_H,gw);SELECT(h1_tmp,gpp[190],c_X0LLLH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    uint64_t c_X0LLLH_L=rlll[d];
    uint64_t c_X0LLLH=c_X0LLLH_H+((c_X0LLLH_L-1)&3u)*2;
   
    
    
    GPOW_186_3(h1_tmp,(1u<<3)-c_X0LLLH,gw);SELECT(h1_tmp,gpp[189],c_X0LLLH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,u2_PAHH,h1_tmp);
    fp_copy(h1_PAH,u2_tmp);fp_sqr(h1_PAH,h1_PAH);//countS++;
    fp_sqr(h1_PAH,h1_PAH);//countS++;
    fp_prime_back(tmp_bn, h1_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_X0LLLL_H=rlll[d]>>1;
  
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X0LLLL_H,gw);SELECT(h1_tmp,gpp[190],c_X0LLLL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAH,u2_tmp,h1_tmp);
    fp_prime_back(tmp_bn, h1_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_X0LLLL_L=rlll[d];
   
    uint64_t c_X0LLLL=c_X0LLLL_H+((c_X0LLLL_L-1)&3u)*2;
    uint64_t c_X0LLL=c_X0LLLH+((c_X0LLLL-1)&7u)*8;  /* 6-bit */
    
    uint64_t c_X0LL=c_X0LLH+((c_X0LLL-1)&63u)*64;  /* 12-bit DLP180 L of DLP168 L */
    
    uint64_t c_X0L=c_X0LH+((c_X0LL-1)&0xFFFull)*4096;  /* 24-bit DLP168 L result */
    
    
    c_X0=c_X0H+((c_X0L-1)&0xFFFFFFull)*(1ull<<24);  /* 48-bit EXT0 result */
    
    
    

    GPOW_95_48(h1_X0,((1ull<<48)-c_X0)&MASK48,gw);SELECT(h1_X0,gpp[143],c_X0==0,h1_X0);//countM++;
    fp_sqr(u_X1,h1_X0);//countS++;
    fp_mul(u_X1,u_X0,u_X1);
    fp_copy(h0_X1,u_X1);
    for(int j=0;j<24;j++){fp_sqr(h0_X1,h0_X1);//countS++;
    }
    fp_copy(h0_PAHH,h0_X1);
    for(int j=0;j<12;j++){fp_sqr(h0_PAHH,h0_PAHH);//countS++;
    }
    fp_copy(h0_PAHHH,h0_PAHH);
    for(int j=0;j<6;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }

    fp_copy(sq_H,h0_PAHHH);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_copy(u2_tmp,sq_H);fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_X1HHHH_H=rlll[d]>>1;
    
    
    
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X1HHHH_H,gw);SELECT(h1_tmp,gpp[190],c_X1HHHH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,sq_H,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    
    uint64_t c_X1HHHH_L=rlll[d];
    
    uint64_t c_X1HHHH=c_X1HHHH_H+((c_X1HHHH_L-1)&3u)*2;
    
    
    
    GPOW_186_3(h1_tmp,(1u<<3)-c_X1HHHH,gw);SELECT(h1_tmp,gpp[189],c_X1HHHH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
    fp_copy(h1_PAHH,u2_tmp);fp_sqr(h1_PAHH,h1_PAHH);//countS++;
    fp_sqr(h1_PAHH,h1_PAHH);//countS++;
    uint64_t c_X1HHHL_H=(uint64_t)rll_lookup(h1_PAHH,rll)>>1;
    GPOW_189_1(h1_tmp,(1u<<1)-c_X1HHHL_H,gw);SELECT(h1_tmp,gpp[190],c_X1HHHL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAHH,u2_tmp,h1_tmp);
    uint64_t c_X1HHHL_L=(uint64_t)rll_lookup(h1_PAHH,rll);
    uint64_t c_X1HHHL=c_X1HHHL_H+((c_X1HHHL_L-1)&3u)*2;
    uint64_t c_X1HHH=c_X1HHHH+((c_X1HHHL-1)&7u)*8;  /* 6-bit */
    

    GPOW_180_6(h1_PAHH,(1u<<6)-c_X1HHH,gw);SELECT(h1_PAHH,gpp[186],c_X1HHH==0,h1_PAHH);//countM++;
    fp_mul(u2_PAHH,h0_PAHH,h1_PAHH);
    fp_copy(h0_PAHHH,u2_PAHH);
    for(int j=0;j<3;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }
    fp_copy(u2_tmp,h0_PAHHH);fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    
    uint64_t c_X1HHLH_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X1HHLH_H,gw);SELECT(h1_tmp,gpp[190],c_X1HHLH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    
    uint64_t c_X1HHLH_L=rlll[d];
    
    uint64_t c_X1HHLH=c_X1HHLH_H+((c_X1HHLH_L-1)&3u)*2;
    
    
    
    GPOW_186_3(h1_tmp,(1u<<3)-c_X1HHLH,gw);SELECT(h1_tmp,gpp[189],c_X1HHLH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,u2_PAHH,h1_tmp);
    fp_copy(h1_PAH,u2_tmp);fp_sqr(h1_PAH,h1_PAH);//countS++;
    fp_sqr(h1_PAH,h1_PAH);//countS++;
    uint64_t c_X1HHLL_H=(uint64_t)rll_lookup(h1_PAH,rll)>>1;
    GPOW_189_1(h1_tmp,(1u<<1)-c_X1HHLL_H,gw);SELECT(h1_tmp,gpp[190],c_X1HHLL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAH,u2_tmp,h1_tmp);
    uint64_t c_X1HHLL_L=(uint64_t)rll_lookup(h1_PAH,rll);
    uint64_t c_X1HHLL=c_X1HHLL_H+((c_X1HHLL_L-1)&3u)*2;
    uint64_t c_X1HHL=c_X1HHLH+((c_X1HHLL-1)&7u)*8;
    uint64_t c_X1HH=c_X1HHH+((c_X1HHL-1)&63u)*64;
    
    
    
    GPOW_168_12(h1_PAH,(1ull<<12)-c_X1HH,gw);SELECT(h1_PAH,gpp[180],c_X1HH==0,h1_PAH);//countM++;
    fp_mul(u2_PAH,h0_X1,h1_PAH);
    fp_copy(h0_PAHHH,u2_PAH);
    for(int j=0;j<6;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }
    fp_copy(sq_H,h0_PAHHH);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_copy(u2_tmp,sq_H);fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_X1LHHH_H=rlll[d]>>1;
 
    GPOW_189_1(h1_tmp,(1u<<1)-c_X1LHHH_H,gw);SELECT(h1_tmp,gpp[190],c_X1LHHH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,sq_H,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;

    uint64_t c_X1LHHH_L=rlll[d];
  
    uint64_t c_X1LHHH=c_X1LHHH_H+((c_X1LHHH_L-1)&3u)*2;

    
    
    
    GPOW_186_3(h1_tmp,(1u<<3)-c_X1LHHH,gw);SELECT(h1_tmp,gpp[189],c_X1LHHH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
    fp_copy(h1_PAHH,u2_tmp);fp_sqr(h1_PAHH,h1_PAHH);//countS++;
    fp_sqr(h1_PAHH,h1_PAHH);//countS++;
    fp_prime_back(tmp_bn, h1_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_X1LHHL_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X1LHHL_H,gw);SELECT(h1_tmp,gpp[190],c_X1LHHL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAHH,u2_tmp,h1_tmp);
    fp_prime_back(tmp_bn, h1_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_X1LHHL_L=rlll[d];
    
    uint64_t c_X1LHHL=c_X1LHHL_H+((c_X1LHHL_L-1)&3u)*2;
    uint64_t c_X1LHH=c_X1LHHH+((c_X1LHHL-1)&7u)*8;
   





    GPOW_180_6(h1_PAHH,(1u<<6)-c_X1LHH,gw);SELECT(h1_PAHH,gpp[186],c_X1LHH==0,h1_PAHH);//countM++;
    fp_mul(u2_PAHH,u2_PAH,h1_PAHH);
    fp_copy(h0_PAHHH,u2_PAHH);
    for(int j=0;j<3;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }
    fp_copy(u2_tmp,h0_PAHHH);fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_X1LLHH_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X1LLHH_H,gw);SELECT(h1_tmp,gpp[190],c_X1LLHH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_X1LLHH_L=rlll[d];
    
    uint64_t c_X1LLHH=c_X1LLHH_H+((c_X1LLHH_L-1)&3u)*2;
  
    
    
    GPOW_186_3(h1_tmp,(1u<<3)-c_X1LLHH,gw);SELECT(h1_tmp,gpp[189],c_X1LLHH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,u2_PAHH,h1_tmp);
    fp_copy(h1_PAH,u2_tmp);fp_sqr(h1_PAH,h1_PAH);//countS++;
    fp_sqr(h1_PAH,h1_PAH);//countS++;
    fp_prime_back(tmp_bn,h1_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_X1LLHL_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X1LLHL_H,gw);SELECT(h1_tmp,gpp[190],c_X1LLHL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAH,u2_tmp,h1_tmp);
    fp_prime_back(tmp_bn, h1_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    
    uint64_t c_X1LLHL_L=rlll[d];
    
    uint64_t c_X1LLHL=c_X1LLHL_H+((c_X1LLHL_L-1)&3u)*2;
    uint64_t c_X1LHL=c_X1LLHH+((c_X1LLHL-1)&7u)*8;
    uint64_t c_X1LL=c_X1LHH+((c_X1LHL-1)&63u)*64;
    c_X1=c_X1HH+((c_X1LL-1)&0xFFFull)*4096;
    

  
    GPOW_143_24(h1_X1,(1ull<<24)-c_X1,gw);SELECT(h1_X1,gpp[167],c_X1==0,h1_X1);//countM++;
    fp_sqr(u_X2,h1_X1);//countS++;
    fp_mul(u_X2,u_X1,u_X2);
    fp_copy(h0_X2,u_X2);
    for(int j=0;j<12;j++){fp_sqr(h0_X2,h0_X2);//countS++;
    }
    fp_copy(h0_PAHHH,h0_X2);
    for(int j=0;j<6;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }

    fp_copy(sq_H,h0_PAHHH);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_copy(u2_tmp,sq_H);fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    
    uint64_t c_X2HHH_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X2HHH_H,gw);SELECT(h1_tmp,gpp[190],c_X2HHH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,sq_H,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    
    uint64_t c_X2HHH_L=rlll[d];
    
    uint64_t c_X2HHH=c_X2HHH_H+((c_X2HHH_L-1)&3u)*2;
    
    
    
    GPOW_186_3(h1_tmp,(1u<<3)-c_X2HHH,gw);SELECT(h1_tmp,gpp[189],c_X2HHH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
    fp_copy(h1_PAHH,u2_tmp);fp_sqr(h1_PAHH,h1_PAHH);//countS++;
    fp_sqr(h1_PAHH,h1_PAHH);//countS++;
    fp_prime_back(tmp_bn, h1_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    
    uint64_t c_X2HHL_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X2HHL_H,gw);SELECT(h1_tmp,gpp[190],c_X2HHL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAHH,u2_tmp,h1_tmp);
    fp_prime_back(tmp_bn, h1_PAHH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    
    uint64_t c_X2HHL_L=rlll[d];
    
    uint64_t c_X2HHL=c_X2HHL_H+((c_X2HHL_L-1)&3u)*2;
    uint64_t c_X2HH=c_X2HHH+((c_X2HHL-1)&7u)*8;
    
    
    
    GPOW_180_6(h1_PAHH,(1u<<6)-c_X2HH,gw);SELECT(h1_PAHH,gpp[186],c_X2HH==0,h1_PAHH);//countM++;
    fp_mul(u2_PAHH,h0_X2,h1_PAHH);
    fp_copy(h0_PAHHH,u2_PAHH);
    for(int j=0;j<3;j++){fp_sqr(h0_PAHHH,h0_PAHHH);//countS++;
    }
    fp_copy(u2_tmp,h0_PAHHH);fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    uint64_t c_X2LHH_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X2LHH_H,gw);SELECT(h1_tmp,gpp[190],c_X2LHH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_PAHHH,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    uint64_t c_X2LHH_L=rlll[d];
    
    uint64_t c_X2LHH=c_X2LHH_H+((c_X2LHH_L-1)&3u)*2;
   
    
    
    GPOW_186_3(h1_tmp,(1u<<3)-c_X2LHH,gw);SELECT(h1_tmp,gpp[189],c_X2LHH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,u2_PAHH,h1_tmp);
    fp_copy(h1_PAH,u2_tmp);fp_sqr(h1_PAH,h1_PAH);//countS++;
    fp_sqr(h1_PAH,h1_PAH);//countS++;
    fp_prime_back(tmp_bn, h1_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    uint64_t c_X2LHL_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X2LHL_H,gw);SELECT(h1_tmp,gpp[190],c_X2LHL_H==0,h1_tmp);//countM++;
    fp_mul(h1_PAH,u2_tmp,h1_tmp);
    fp_prime_back(tmp_bn, h1_PAH);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    uint64_t c_X2LHL_L=rlll[d];
    
    uint64_t c_X2LHL=c_X2LHL_H+((c_X2LHL_L-1)&3u)*2;
    
    uint64_t c_X2LH=c_X2LHH+((c_X2LHL-1)&7u)*8;  /* 6-bit DLP186 L result */
    
    c_X2=c_X2HH+((c_X2LH-1)&63u)*64;  /* 12-bit EXT2 result */
   
    


    GPOW_167_12(h1_X2,(1u<<12)-c_X2,gw);SELECT(h1_X2,gpp[179],c_X2==0,h1_X2);//countM++;
    fp_sqr(u_X3,h1_X2);//countS++;
    fp_mul(u_X3,u_X2,u_X3);
    fp_copy(h0_X3,u_X3);
    for(int j=0;j<6;j++){fp_sqr(h0_X3,h0_X3);//countS++;
    }
    fp_copy(sq_H,h0_X3);
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_sqr(sq_H,sq_H);//countS++;
    fp_copy(u2_tmp,sq_H);fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_sqr(u2_tmp,u2_tmp);//countS++;
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_X3HH_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X3HH_H,gw);SELECT(h1_tmp,gpp[190],c_X3HH_H==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,sq_H,h1_tmp);
    fp_prime_back(tmp_bn, u2_tmp);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_X3HH_L=rlll[d];
    
    uint64_t c_X3HH=c_X3HH_H+((c_X3HH_L-1)&3u)*2; 
   

    GPOW_186_3(h1_tmp,(1u<<3)-c_X3HH,gw);SELECT(h1_tmp,gpp[189],c_X3HH==0,h1_tmp);//countM++;
    fp_mul(u2_tmp,h0_X3,h1_tmp);
    fp_copy(h1_X3,u2_tmp);fp_sqr(h1_X3,h1_X3);//countS++;
    fp_sqr(h1_X3,h1_X3);//countS++;
    fp_prime_back(tmp_bn, h1_X3);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_X3HL_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X3HL_H,gw);SELECT(h1_tmp,gpp[190],c_X3HL_H==0,h1_tmp);//countM++;
    fp_mul(h1_X3,u2_tmp,h1_tmp);
    fp_prime_back(tmp_bn, h1_X3);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t c_X3HL_L=rlll[d];
    
    uint64_t c_X3HL=c_X3HL_H+((c_X3HL_L-1)&3u)*2;
    c_X3=c_X3HH+((c_X3HL-1)&7u)*8;
   
    
    GPOW_179_6(h1_X3,(1u<<6)-c_X3,gw);SELECT(h1_X3,gpp[185],c_X3==0,h1_X3);//countM++;
    fp_sqr(u_X4,h1_X3);//countS++;
    fp_mul(u_X4,u_X3,u_X4);
    fp_copy(h0_X4,u_X4);
    fp_sqr(h0_X4,h0_X4);//countS++;
    fp_sqr(h0_X4,h0_X4);//countS++;
    fp_sqr(h0_X4,h0_X4);//countS++;
    fp_copy(h1_X4,h0_X4);fp_sqr(h1_X4,h1_X4);//countS++;
    fp_sqr(h1_X4,h1_X4);//countS++;
    fp_prime_back(tmp_bn, h1_X4);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    uint64_t c_X4H_H=rlll[d]>>1;
    
    GPOW_189_1(h1_tmp,(1u<<1)-c_X4H_H,gw);SELECT(h1_tmp,gpp[190],c_X4H_H==0,h1_tmp);//countM++;
    fp_mul(h1_X4,h0_X4,h1_tmp);
    
    uint64_t c_X4H_L=(uint64_t)rll_lookup(h1_X4,rll);
    
    uint64_t c_X4=c_X4H_H+((c_X4H_L-1)&3u)*2;  /* 3-bit EXT4 result */
   

    
    GPOW_185_3(h1_X4,(1u<<3)-c_X4,gw);SELECT(h1_X4,gpp[188],c_X4==0,h1_X4);//countM++;
    fp_sqr(u_X5,h1_X4);//countS++;
    fp_mul(u_X5,u_X4,u_X5);
    fp_copy(h0_X5,u_X5);
    fp_sqr(h0_X5,h0_X5);//countS++;
    fp_sqr(h0_X5,h0_X5);//countS++;
    fp_prime_back(tmp_bn, h0_X5);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
    uint64_t c_X5H=rlll[d]>>1;
    
    GPOW_188_1(h1_X5,(1u<<1)-c_X5H,gw);SELECT(h1_X5,gpp[189],c_X5H==0,h1_X5);//countM++;
    fp_sqr(u_X6,h1_X5);//countS++;
    fp_mul(u_X6,u_X5,u_X6);
    fp_prime_back(tmp_bn, u_X6);
    bn_get_dig(&d, tmp_bn);
    d=d&0x3;
   
    uint64_t e_base=rlll[d];
    
    fp_copy(t_X5,fll[e_base]);
    fp_mul(t_X5,h1_X5,t_X5);//countM++;
    fp_mul(t_X4,h1_X4,t_X5);//countM++;
    fp_mul(t_X3,h1_X3,t_X4);//countM++;
    fp_mul(t_X2,h1_X2,t_X3);//countM++;
    fp_mul(t_X1,h1_X1,t_X2);//countM++;
    fp_mul(t_X0,h1_X0,t_X1);//countM++;
    fp_mul(out_d,f_A,t_X0);//countM++;
    }
bn_free(tmp_bn);
#define FF(v) fp_free(v);
    FF(h0_A);FF(h0_PA);FF(h0_PAH);FF(h0_PAHH);
    FF(sq_H);FF(h1_tmp);FF(u2_tmp);FF(h0_PAHHH);
    FF(h1_PAHH);FF(u2_PAHH);FF(h1_PAH);FF(u2_PAH);FF(h1_PA);FF(u2_PA);
    FF(f_A);FF(u_X0);
    FF(h0_X0);FF(h1_X0);FF(u_X1);
    FF(h0_X1);FF(h1_X1);FF(u_X2);
    FF(h0_X2);FF(h1_X2);FF(u_X3);
    FF(h0_X3);FF(h1_X3);FF(u_X4);
    FF(h0_X4);FF(h1_X4);FF(u_X5);
    FF(h0_X5);FF(h1_X5);FF(u_X6);
    FF(t_X5);FF(t_X4);FF(t_X3);FF(t_X2);FF(t_X1);FF(t_X0);FF(t_A);
#undef FF
}

/* ── SQRT ────────────────────────────────────────────────────────────────── */
void SQRT(fp_t x,fp_t y,bn_t e,
          fp_t gw[NW][WE],fp_t rll[WE],fp_t fll[WE],fp_t gpp[N], uint8_t rlll[4]){
    fp_t u,v,w_,d;fp_null(u);fp_null(v);fp_null(w_);fp_null(d);
    fp_new(u);fp_new(v);fp_new(w_);fp_new(d);
    fp_exp(v,x,e);
    fp_mul(w_,x,v);//countM++;
    fp_mul(u,w_,v);//countM++;
    solve_dlp_pow2_flat(u,d,gw,rll,fll,gpp,rlll);
    fp_mul(y,w_,d);//countM++;
    //fp_sqr(y,y);
    //fp_print(y);
    //fp_print(x);
    fp_free(u);fp_free(v);fp_free(w_);fp_free(d);}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(void){
    if(core_init()!=RLC_OK){core_clean();return 1;}
    if(fp_param_set_any_pmers()!=RLC_OK){printf("Curve init failed\n");core_clean();return 1;}
   //printf("n=192, w=2\n");
    fp_t gw[NW][WE],rll[WE],fll[WE],gpp[N],g,z,h,hh,b,y;
    for(int i=0;i<NW;i++)for(int j=0;j<WE;j++){fp_null(gw[i][j]);fp_new(gw[i][j]);}
    for(int i=0;i<WE;i++){fp_null(rll[i]);fp_new(rll[i]);}
    for(int i=0;i<WE;i++){fp_null(fll[i]);fp_new(fll[i]);}
    for(int i=0;i<N; i++){fp_null(gpp[i]);fp_new(gpp[i]);}
    uint8_t rlll[4]={0};
    rlll[ 0x1 ]= 0 ;
    rlll[ 0x3 ]= 1 ;
    rlll[ 0x0 ]= 2 ;
    rlll[ 0x2 ]= 3 ;
    
    
    fp_null(b);fp_new(b);fp_null(y);fp_new(y);
    fp_null(g);fp_new(g);fp_null(z);fp_new(z);fp_null(h);fp_new(h);fp_null(hh);fp_new(hh);
    fp_rand(b);while(!fp_is_sqr(b))fp_rand(b);
    //fp_set_dig(b,1024);
    //bn_read_str(tmp,"07D478EB6943DF8DF45934DCB0F2FF09C1644AE8F0601C0BB98F5C6919987E4F",64,16);fp_prime_conv(b,tmp);
    bn_t tmp,m,e,a1;
    bn_null(e);bn_new(e);bn_null(m);bn_new(m);bn_null(tmp);bn_new(tmp);bn_null(a1);bn_new(a1);
    bn_read_str(tmp,"3",1,16);fp_prime_conv(z,tmp);
    //bn_read_str(tmp,"06060747D4579DC3E8EFACBEFF7CECA42936C792F8934DACA5BD1EC566B5C665",64,16);fp_prime_conv(b,tmp);
    bn_read_str(m,"800000000000011",15,16);fp_exp(g,z,m);
    /* e = (m-1)/2 = 0x400000000000008 */
    bn_read_str(e,"400000000000008",15,16);
    /* h = g^(2^190): n-w=190, so 2^190 in hex is 1 followed by 190/4=47.5 nibbles
     * 2^190 = 0x400000000000000000000000000000000000000000000000 (48 hex digits) */
    bn_read_str(a1,"400000000000000000000000000000000000000000000000",48,16);
    fp_exp(h,g,a1);fp_srt(hh,h);fp_inv(hh,hh);
    precomputation(g,h,hh,gw,rll,fll,gpp);
    MEASURE(SQRT(b,y,e,gw,rll,fll,gpp,rlll););
    printf("RDTSC_clk_min    = %f\n",RDTSC_clk_min);
    printf("RDTSC_clk_median = %f\n",RDTSC_clk_median);
    printf("RDTSC_clk_max    = %f\n",RDTSC_clk_max);
    printf("mult_count = %d\n",countM);
    printf("sqr_count  = %d\n",countS);
    
    fp_print(b);
    /* verify */
    fp_t chk; fp_null(chk); fp_new(chk);
    fp_sqr(chk,y);
    if(fp_cmp(chk,b)==RLC_EQ)printf("sqrt correct: y^2==b\n");
    else                      printf("sqrt WRONG\n");
    fp_free(chk);
    for(int i=0;i<NW;i++)for(int j=0;j<WE;j++)fp_free(gw[i][j]);
    for(int i=0;i<WE;i++)fp_free(rll[i]);
    for(int i=0;i<WE;i++)fp_free(fll[i]);
    for(int i=0;i<N; i++)fp_free(gpp[i]);
    fp_free(b);fp_free(y);fp_free(g);fp_free(z);fp_free(h);fp_free(hh);
    bn_free(e);bn_free(m);bn_free(a1);bn_free(tmp);
    core_clean();return 0;}
