#include <stdio.h>
#include <relic/relic.h>
#include <relic/relic_fp.h>
#include <relic/relic_ep.h>
#include<immintrin.h>

#include <relic/relic_core.h>
#include "relic/relic_md.h"


void findsqroot(fp_t y, fp_t u) {
    fp_t b,I,tmp3;
    fp_t bs;
    bn_t tmp;
    
    
    bn_null(tmp);
    fp_null(I);
    fp_null(b);
    fp_null(bs);
    fp_null(tmp3);
    
    RLC_TRY {
        bn_new(tmp);
        fp_new(b);
        fp_new(I);
        fp_new(bs);
        fp_new(tmp3);
        
        bn_read_str(tmp,
            "2FD2C10DA03FAC68144538C5BA29D4E72EAF66BC66CF1D8992666BAF984D6A1ED0ACC24E225240726B4741FEB723385D27B442F2F542EDD0A4E13CD7FFF92AF4D403A32F7FB552",
            142, 16);

        fp_prime_conv(I, tmp);
        
        bn_read_str(tmp,
            "F63ABEA75727B41BF73E42AC1656E84D33E336622E3B83E9E5FD34F82F91BF9B4777BF28E54DFF228DEA8A493E970AE745D13724429EF89C01221F7F97E4550381F0C00A778516",
            142, 16);
        
        fp_exp(b,u,tmp);
        fp_sqr(bs,b);
        fp_mul(tmp3,b,I);
        if(fp_cmp(bs,u) == RLC_EQ){
            fp_copy(y,b);
        }
        else{
            fp_copy(y,tmp3);
        }
        
    }

    RLC_CATCH_ANY {
	    RLC_THROW(ERR_CAUGHT);
    }
    RLC_FINALLY {
            fp_free(b);
            fp_free(I);
            fp_free(bs);
            fp_free(tmp3);
            bn_free(tmp);
    }
}






#define TMPL_MAP_HORNER(PFX, IN)											\
	static void PFX##_eval(PFX##_t c, PFX##_t a, IN *coeffs, int deg) {		\
		PFX##_copy(c, coeffs[deg]);											\
		for (int i = deg; i > 0; --i) {										\
			PFX##_mul(c, c, a);												\
			PFX##_add(c, c, coeffs[i - 1]);									\
		}																	\
	}

/* conditionally normalize result of isogeny map when not using projective coords */
#if EP_ADD == JACOB

#define TMPL_MAP_ISOMAP_NORM(PFX)											\
	do {																	\
		/* Y = Ny * Dx * Z^2. */											\
		PFX##_mul(q->y, p->y, t1);											\
		PFX##_mul(q->y, q->y, t3);											\
		/* Z = Dx * Dy, t1 = Z^2. */										\
		PFX##_mul(q->z, t2, t3);											\
		PFX##_sqr(t1, q->z);												\
		PFX##_mul(q->y, q->y, t1);											\
		/* X = Nx * Dy * Z. */												\
		PFX##_mul(q->x, t0, t2);											\
		PFX##_mul(q->x, q->x, q->z);										\
		q->coord = JACOB;													\
	} while (0)																\

#elif EP_ADD == PROJC

#define TMPL_MAP_ISOMAP_NORM(PFX)											\
		/* Z = Dx * Dy. */													\
		PFX##_mul(q->z, t2, t3);											\
		/* X = Nx * Dy. */													\
		PFX##_mul(q->x, t0, t2);											\
		/* Y = Ny * Dx. */													\
		PFX##_mul(q->y, p->y, t1);											\
		PFX##_mul(q->y, q->y, t3);											\
		q->coord = PROJC;													\

#else

#define TMPL_MAP_ISOMAP_NORM(PFX)											\
	do {																	\
		/* when working with affine coordinates, clear denominator */		\
		PFX##_mul(q->z, t2, t3);											\
		PFX##_inv(q->z, q->z);												\
		/* y coord */														\
		PFX##_mul(q->y, p->y, q->z);										\
		PFX##_mul(q->y, q->y, t3);											\
		PFX##_mul(q->y, q->y, t1);											\
		/* x coord */														\
		PFX##_mul(q->x, t2, q->z);											\
		PFX##_mul(q->x, q->x, t0);											\
		/* z coord == 1 */													\
		PFX##_set_dig(q->z, 1);												\
		q->coord = BASIC;													\
	} while (0)																\

#endif

/**
 * Generic isogeny map evaluation for use with SSWU map.
 */
#define TMPL_MAP_ISOGENY_MAP(CUR, PFX, ISO)									\
	/* declaring this function inline suppresses unused warnings */			\
	static inline void CUR##_iso(CUR##_t q, CUR##_t p) {					\
		PFX##_t t0, t1, t2, t3;												\
																			\
		if (!CUR##_curve_is_ctmap()) {										\
			CUR##_copy(q, p);												\
			return;															\
		}																	\
		/* XXX need to add real support for input projective points */		\
		if (p->coord != BASIC) {											\
			CUR##_norm(p, p);												\
		}																	\
																			\
		PFX##_null(t0);														\
		PFX##_null(t1);														\
		PFX##_null(t2);														\
		PFX##_null(t3);														\
                                                                    		\
		RLC_TRY {															\
			PFX##_new(t0);													\
			PFX##_new(t1);													\
			PFX##_new(t2);													\
			PFX##_new(t3);													\
																			\
			ISO##_t coeffs = CUR##_curve_get_iso();							\
																			\
			/* numerators */												\
			PFX##_eval(t0, p->x, coeffs->xn, coeffs->deg_xn);				\
			PFX##_eval(t1, p->x, coeffs->yn, coeffs->deg_yn);				\
			/* denominators */												\
			PFX##_eval(t2, p->x, coeffs->yd, coeffs->deg_yd);				\
			PFX##_eval(t3, p->x, coeffs->xd, coeffs->deg_xd);				\
																			\
			/* normalize if necessary */									\
			TMPL_MAP_ISOMAP_NORM(PFX);										\
		}																	\
		RLC_CATCH_ANY { RLC_THROW(ERR_CAUGHT); }							\
		RLC_FINALLY {														\
			PFX##_free(t0);													\
			PFX##_free(t1);													\
			PFX##_free(t2);													\
			PFX##_free(t3);													\
		}																	\
	}																		\

/* Conditionally call isogeny mapping function depending on whether EP_CTMAP is defined */
#ifdef EP_CTMAP
#define TMPL_MAP_CALL_ISOMAP(CUR, PT)										\
	do {																	\
		if (CUR##_curve_is_ctmap()) {										\
			CUR##_iso(PT, PT);												\
		}																	\
	} while (0)																\

#else

#define TMPL_MAP_CALL_ISOMAP(CUR, PT) /* No isogeny map call in this case. */

#endif

/**
 * Simplified SWU mapping from Section 4 of
 * "Fast and simple constant-time hashing to the BLS12-381 Elliptic Curve"
 */
#define TMPL_MAP_SSWU(CUR, PFX, PTR_TY)										\
	static void CUR##_map_sswu(CUR##_t p, const PFX##_t t) {				\
		PFX##_t t0, t1, t2, t3;												\
		ctx_t *ctx = core_get();											\
		PTR_TY *mBoverA = ctx->CUR##_map_c[0];								\
		PTR_TY *a = ctx->CUR##_map_c[2];									\
		PTR_TY *b = ctx->CUR##_map_c[3];									\
		PTR_TY *u = ctx->CUR##_map_u;										\
                                                                            \
		PFX##_null(t0);														\
		PFX##_null(t1);														\
		PFX##_null(t2);														\
		PFX##_null(t3);														\
																			\
		RLC_TRY {															\
			PFX##_new(t0);													\
			PFX##_new(t1);													\
			PFX##_new(t2);													\
			PFX##_new(t3);													\
																			\
			/* start computing the map */									\
			PFX##_sqr(t0, t);												\
			PFX##_mul(t0, t0, u);  /* t0 = u * t^2 */						\
			PFX##_sqr(t1, t0);     /* t1 = u^2 * t^4 */						\
			PFX##_add(t2, t1, t0); /* t2 = u^2 * t^4 + u * t^2 */			\
																			\
			/* handle the exceptional cases */								\
			/* XXX(rsw) should be done projectively */						\
			{																\
				const int e1 = PFX##_is_zero(t2);							\
				PFX##_neg(t3, u);           /* t3 = -u */					\
				/* exception: -u instead of u^2t^4 + ut^2 */				\
				PFX##_copy_sec(t2, t3, e1);      							\
				/* t2 = -1/u or 1/(u^2 * t^4 + u*t^2) */					\
				PFX##_inv(t2, t2);          								\
				PFX##_add_dig(t3, t2, 1);   /* t3 = 1 + t2 */				\
				PFX##_copy_sec(t2, t3, e1 == 0); /* add 1 if t2 != -1/u */	\
			}																\
			/* e1 goes out of scope */										\
			/* compute x1, g(x1) */											\
			/* -B / A * (1 + 1 / (u^2 * t^4 + u * t^2)) */					\
			PFX##_mul(p->x, t2, mBoverA);									\
			PFX##_sqr(p->y, p->x);        /* x^2 */							\
			PFX##_add(p->y, p->y, a);     /* x^2 + a */						\
			PFX##_mul(p->y, p->y, p->x);  /* x^3 + a x */					\
			PFX##_add(p->y, p->y, b);     /* x^3 + a x + b */				\
                                                                            \
			/* compute x2, g(x2) */											\
			PFX##_mul(t2, t0, p->x); /* t2 = u * t^2 * x1 */				\
			PFX##_mul(t1, t0, t1);   /* t1 = u^3 * t^6 */					\
			PFX##_mul(t3, t1, p->y); /* t5 = g(t2) = u^3 * t^6 * g(p->x) */	\
			{																\
				/* try x2, g(x2) */											\
				const int e1 = PFX##_is_sqr(p->y);							\
				PFX##_copy_sec(p->x, t2, e1 == 0);							\
				PFX##_copy_sec(p->y, t3, e1 == 0);							\
			}\
			findsqroot(p->y,p->y);										\
			PFX##_set_dig(p->z, 1);											\
			p->coord = BASIC;												\
		}																	\
		RLC_CATCH_ANY { RLC_THROW(ERR_CAUGHT); }							\
		RLC_FINALLY {														\
			PFX##_free(t0);													\
			PFX##_free(t1);													\
			PFX##_free(t2);													\
			PFX##_free(t3);													\
		}																	\
	}																		\

/**
 * Shallue--van de Woestijne map, based on the definition from
 * draft-irtf-cfrg-hash-to-curve-06, Section 6.6.1
 */
#define TMPL_MAP_SVDW(CUR, PFX, PTR_TY)										\
	static void CUR##_map_svdw(CUR##_t p, const PFX##_t t) {				\
		PFX##_t t1, t2, t3, t4;												\
		ctx_t *ctx = core_get();											\
		PTR_TY *gU = ctx->CUR##_map_c[0];									\
		PTR_TY *mUover2 = ctx->CUR##_map_c[1];								\
		PTR_TY *c3 = ctx->CUR##_map_c[2];									\
		PTR_TY *c4 = ctx->CUR##_map_c[3];									\
		PTR_TY *u = ctx->CUR##_map_u;										\
																			\
		PFX##_null(t1);														\
		PFX##_null(t2);														\
		PFX##_null(t3);														\
		PFX##_null(t4);														\
                                                                            \
		RLC_TRY {															\
			PFX##_new(t1);													\
			PFX##_new(t2);													\
			PFX##_new(t3);													\
			PFX##_new(t4);													\
																			\
			/* start computing the map */									\
			PFX##_sqr(t1, t);												\
			PFX##_mul(t1, t1, gU);											\
			PFX##_add_dig(t2, t1, 1); /* 1 + t^2 * g(u) */					\
			PFX##_sub_dig(t1, t1, 1);										\
			PFX##_neg(t1, t1);     /* 1 - t^2 * g(u) */						\
			PFX##_mul(t3, t1, t2); /* (1+t^2*g(u)) * (1-t^2*g(u)) */		\
                                                                            \
			/* handle exceptional case */									\
			{																\
				/* compute inv0(t3), i.e., 0 if t3 == 0, 1/t3 otherwise */	\
				const int e0 = PFX##_is_zero(t3);							\
				PFX##_copy_sec(t3, gU, e0); /* g(u) is nonzero */			\
				PFX##_inv(t3, t3);											\
				PFX##_zero(t4);												\
				PFX##_copy_sec(t3, t4, e0);									\
			}																\
			/* e0 goes out of scope */										\
			PFX##_mul(t4, t, t1);											\
			PFX##_mul(t4, t4, t3);											\
			PFX##_mul(t4, t4, c3);											\
																			\
			/* compute x1 and g(x1) */										\
			PFX##_sub(p->x, mUover2, t4);									\
			CUR##_rhs(p->y, p->x);											\
			{																\
				const int e0 = PFX##_is_sqr(p->y);							\
				/* compute x2 and g(x2) */									\
				PFX##_add(t4, mUover2, t4);									\
				PFX##_copy_sec(p->x, t4, e0 == 0);							\
				CUR##_rhs(t1, p->x);										\
				PFX##_copy_sec(p->y, t1, e0 == 0);							\
			}																\
			{																\
				const int e1 = PFX##_is_sqr(p->y);							\
				/* compute x3 and g(x3) */									\
				PFX##_sqr(t1, t2);											\
				PFX##_mul(t1, t1, t3);										\
				PFX##_sqr(t1, t1);											\
				PFX##_mul(t1, t1, c4);										\
				PFX##_add(t1, t1, u);										\
				PFX##_copy_sec(p->x, t1, e1 == 0);							\
				CUR##_rhs(t2, p->x);										\
				PFX##_copy_sec(p->y, t2, e1 == 0);							\
			}																\
			if (!PFX##_srt(p->y, p->y)) {									\
				RLC_THROW(ERR_NO_VALID);									\
			}																\
			PFX##_set_dig(p->z, 1);											\
			p->coord = BASIC;												\
		}																	\
		RLC_CATCH_ANY { RLC_THROW(ERR_CAUGHT); }							\
		RLC_FINALLY {														\
			PFX##_free(t1);													\
			PFX##_free(t2);													\
			PFX##_free(t3);													\
			PFX##_free(t4);													\
		}																	\
	}																		\

/*============================================================================*/
/* Private definitions                                                        */
/*============================================================================*/

/**
 * Domain separation tag.
 */
#ifndef RLC_DSTAG
#define RLC_DSTAG		RLC_STRING
#endif

#ifdef EP_CTMAP

/**
 * Evaluate a polynomial represented by its coefficients over a using Horner's
 * rule. Might promove to an API if needed elsewhere in the future.
 *
 * @param[out] c		- the result.
 * @param[in] a			- the input value.
 * @param[in] coeffs	- the vector of coefficients in the polynomial.
 * @param[in] deg 		- the degree of the polynomial.
 */
TMPL_MAP_HORNER(fp, fp_st);

/**
 * Generic isogeny map evaluation for use with SSWU map.
 */
TMPL_MAP_ISOGENY_MAP(ep, fp, iso);

#endif /* EP_CTMAP */

/**
 * Simplified SWU mapping from Section 4 of
 * "Fast and simple constant-time hashing to the BLS12-381 Elliptic Curve"
 */
TMPL_MAP_SSWU(ep, fp, dig_t);

/**
 * Shallue--van de Woestijne map, based on the definition from
 * draft-irtf-cfrg-hash-to-curve-06, Section 6.6.1
 */
TMPL_MAP_SVDW(ep, fp, dig_t);

static void ep_map_basic_impl(ep_t p, const uint8_t *bytes, size_t len) {
	bn_t x;
	fp_t t0;

	bn_null(x);
	fp_null(t0);

	RLC_TRY {
		bn_new(x);
		fp_new(t0);

		bn_read_bin(x, bytes, len);
		fp_prime_conv(p->x, x);
		fp_set_dig(p->z, 1);

		while (1) {
			ep_rhs(t0, p->x);

			if (fp_smb(t0) == 1) {
				fp_srt(p->y, t0);
				p->coord = BASIC;
				break;
			}

			fp_add_dig(p->x, p->x, 1);
		}

		ep_mul_cof(p, p);
	}
	RLC_CATCH_ANY {
		RLC_THROW(ERR_CAUGHT);
	}
	RLC_FINALLY {
		bn_free(x);
		fp_free(t0);
	}
}

/**
 * Maps an array of uniformly random bytes to a point in a prime elliptic
 * curve.
 * That array is expected to have a length suitable for two field elements plus
 * extra bytes for uniformity.
  *
 * @param[out] p			- the result.
 * @param[in] uniform_bytes	- the array of uniform bytes to map.
 * @param[in] len			- the array length in bytes.
 * @param[in] map_fn		- the mapping function.
 */
static void ep_map_sswum_impl2(ep_t p, const uint8_t *bytes, size_t len,
		void (*const map_fn)(ep_t, const fp_t)) {
	bn_t k;
	fp_t t;
	ep_t q;
	int neg;
	/* enough space for two field elements plus extra bytes for uniformity */
	const size_t elm = (FP_PRIME + ep_param_level() + 7) / 8;

	bn_null(k);
	fp_null(t);
	ep_null(q);

	RLC_TRY {
		bn_new(k);
		fp_new(t);
		ep_new(q);

#define EP_MAP_CONVERT_BYTES(IDX)											\
    do {																	\
		bn_read_bin(k, bytes + IDX * elm, elm);						\
		fp_prime_conv(t, k);												\
    } while (0)

#define EP_MAP_APPLY_MAP(PT)												\
    do {																	\
		/* check sign of t */												\
		neg = fp_is_even(t);												\
		/* convert */														\
		map_fn(PT, t);														\
		/* compare sign of y and sign of t; fix if necessary */				\
		neg = neg != fp_is_even(PT->y);										\
		fp_neg(t, PT->y);													\
		dv_copy_sec(PT->y, t, RLC_FP_DIGS, neg);							\
    } while (0)

		/* first map invocation */
		EP_MAP_CONVERT_BYTES(0);
		EP_MAP_APPLY_MAP(p);
		TMPL_MAP_CALL_ISOMAP(ep, p);

		/* second map invocation */
		EP_MAP_CONVERT_BYTES(1);
		EP_MAP_APPLY_MAP(q);
		TMPL_MAP_CALL_ISOMAP(ep, q);

		/* XXX(rsw) could add p and q and then apply isomap,
		 * but need ep_add to support addition on isogeny curves */

#undef EP_MAP_CONVERT_BYTES
#undef EP_MAP_APPLY_MAP

		/* sum the result */
		ep_add(p, p, q);
		ep_norm(p, p);
		ep_mul_cof(p, p);
	}
	RLC_CATCH_ANY {
		RLC_THROW(ERR_CAUGHT);
	}
	RLC_FINALLY {
		bn_free(k);
		fp_free(t);
		ep_free(q);
	}
}

static void ep_map_swift_impl2(ep_t p, const uint8_t *random, size_t len) {
	fp_t h[8], t1, t2, v, w, y, x1, x2, x3, d[3];
	ctx_t *ctx = core_get();
	bn_t k;
	uint8_t s;

	bn_null(k);
	fp_null(v);
	fp_null(w);
	fp_null(y);
	fp_null(t1);
	fp_null(t2);
	fp_null(x1);
	fp_null(x2);
	fp_null(x3);
	fp_null(d[0]);
	fp_null(d[1]);
	fp_null(d[2]);

	RLC_TRY {
		bn_new(k);
		fp_new(v);
		fp_new(w);
		fp_new(y);
		fp_new(t1);
		fp_new(t2);
		fp_new(x1);
		fp_new(x2);
		fp_new(x3);
		fp_new(d[0]);
		fp_new(d[1]);
		fp_new(d[2]);
		for (size_t i = 0; i < 8; i++) {
			fp_null(h[i]);
			fp_new(h[i]);
		}

		bn_read_bin(k, random, len / 2);
		fp_prime_conv(t1, k);
		bn_read_bin(k, random + len / 2, len / 2);
		fp_prime_conv(t2, k);
		s = random[len - 1] & 1;

		if (ep_curve_opt_b() == RLC_ZERO) {
			/* h0 = t1^2, h1 = h0^2, h2 = 4a, h3 = h2^3, h4 = h0 * h1 + h3. */
			fp_sqr(h[0], t1);
			fp_sqr(h[1], h[0]);
			fp_mul(h[4], h[0], h[1]);
			if (ep_curve_opt_a() == RLC_ONE) {
				fp_add_dig(h[4], h[4], 64);
			} else {
				fp_dbl(h[2], ep_curve_get_a());
				fp_dbl(h[2], h[2]);
				fp_sqr(h[3], h[2]);
				fp_mul(h[3], h[3], h[2]);
				fp_add(h[4], h[4], h[3]);
			}
			/* h5 = t2^2, h6 = \tau * t1, h7 = 24*h0*h5*h6. */
			fp_sqr(h[5], t2);
			fp_mul(h[6], ctx->ep_map_c[4], t1);
			fp_mul(h[7], h[0], h[5]);
			fp_mul(h[7], h[7], h[6]);
			fp_mul_dig(h[7], h[7], 24);
			/* \tau = (\omega - 1)/2. */
			fp_sub_dig(p->x, ctx->ep_map_c[4], 1);
			fp_hlv(p->x, p->x);
			/* w = h9 = h1^2, v = h10 = \omega(h2h4 + h0h7). */
			fp_sqr(w, h[1]);
			fp_mul(v, h[0], h[7]);
			if (ep_curve_opt_a() == RLC_ONE) {
				fp_dbl(t1, h[4]);
				fp_dbl(t1, t1);
			} else {
				fp_mul(t1, h[2], h[4]);
			}
			fp_add(v, v, t1);
			fp_mul(v, v, p->x);

			/* d0 = -2h6\omega(h4 + h7), d1 = d0\omega. */
			fp_add(d[0], h[4], h[7]);
			fp_mul(d[0], d[0], h[6]);
			fp_mul(d[0], d[0], p->x);
			fp_dbl(d[0], d[0]);
			fp_neg(d[0], d[0]);
			fp_mul(d[1], d[0], p->x);
			if (ep_curve_opt_a() == RLC_ONE) {
				fp_sub_dig(d[2], h[0], 4);
			} else {
				fp_sub(d[2], h[0], h[2]);
			}
			/* d2 = -432*h1h5(h0 - h2)^2. */
			fp_sqr(d[2], d[2]);
			fp_mul_dig(d[2], d[2], 216);
			fp_dbl(d[2], d[2]);
			fp_neg(d[2], d[2]);
			fp_mul(d[2], d[2], h[1]);
			fp_mul(d[2], d[2], h[5]);

			if (fp_is_zero(d[0]) || fp_is_zero(d[1]) || fp_is_zero(d[2])) {
				ep_set_infty(p);
			} else {
				if (ep_curve_opt_a() == RLC_ONE) {
					/* n2 = 4(16h0 + h7). */
					fp_dbl(h[0], h[0]);
					fp_dbl(h[0], h[0]);
					fp_dbl(h[0], h[0]);
					fp_dbl(h[0], h[0]);
					fp_add(x2, h[0], h[7]);
					fp_dbl(x2, x2);
					fp_dbl(x2, x2);
				} else {
					/* n2 = h8 + h9 + h2h7 + h10. */
					fp_mul(t1, h[0], h[3]);
					fp_mul(x2, h[2], h[7]);
					fp_add(x2, x2, t1);
				}
				/* n1 = n2 + h9 + h10\omega, n2 = n1 + h10. */
				fp_add(x2, x2, w);
				fp_mul(x3, v, p->x);
				fp_add(x1, x2, x3);
				fp_add(x2, x2, v);
				/* n3 = h1(h9 + 8*16h0) + 4096 - h7(h4 - 3(4h1 + 16h0) - h7). */
				if (ep_curve_opt_a() == RLC_ONE) {
					fp_dbl(h[2], h[1]);
					fp_dbl(h[2], h[2]);
					fp_add(x3, h[2], h[0]);
				} else {
					fp_mul(x3, h[2], h[0]);
					fp_add(x3, x3, h[1]);
					fp_mul(x3, x3, h[2]);
				}
				fp_dbl(t1, x3);
				fp_add(x3, x3, t1);
				fp_sub(x3, h[4], x3);
				fp_sub(x3, x3, h[7]);
				fp_mul(x3, x3, h[7]);
				if (ep_curve_opt_a() == RLC_ONE) {
					fp_dbl(h[0], h[0]);
					fp_dbl(h[0], h[0]);
					fp_dbl(h[0], h[0]);
					fp_set_dig(t2, 64);
					fp_sqr(t2, t2);
				} else {
					fp_dbl(h[0], t1);
					fp_sqr(t2, h[3]);
				}
				fp_add(h[0], h[0], w);
				fp_mul(t1, h[0], h[1]);
				fp_sub(x3, t1, x3);
				fp_add(x3, x3, t2);

				/* Invert d0, d1 and d2 simultaneously. */
				fp_inv_sim(d, d, 3);
				fp_mul(p->x, x1, d[0]);
				fp_mul(x2, x2, d[1]);
				fp_mul(x3, x3, d[2]);
			}
		} else {
			/* This is the SwiftEC case per se. */
			if (ep_curve_opt_a() != RLC_ZERO) {
				RLC_THROW(ERR_NO_VALID);
			} else {
				/* h_0 = t1^3, h1 = t2^2, h2 = h0 + b - h1, h3 = 2h1 + h2. */
				fp_sqr(h[0], t1);
				fp_mul(h[0], h[0], t1);
				fp_sqr(h[1], t2);
				fp_add(h[2], h[0], ctx->ep_b);
				fp_sub(h[2], h[2], h[1]);
				fp_dbl(h[3], h[1]);
				fp_add(h[3], h[3], h[2]);
				/* h6 = t1\tau, v = h7 = h2h6, h8 = 2h6t2.*/
				fp_mul(x3, t1, ctx->ep_map_c[4]);
				fp_mul(v, h[2], x3);
				fp_mul(x3, x3, t2);
				fp_dbl(x3, x3);

				/* n1 = h8(h7 - t1h3), n2 = 2h3^2, d1 = 2h3h8*/
				fp_mul(x1, t1, h[3]);
				fp_sub(x1, v, x1);
				fp_mul(x1, x1, x3);
				fp_dbl(y, h[3]);
				fp_sqr(y, y);
				fp_mul(w, h[3], x3);
				fp_dbl(w, w);

				if (fp_is_zero(w)) {
					ep_set_infty(p);
				} else {
					fp_inv(w, w);
					fp_mul(p->x, x1, w);
					fp_add(x2, t1, p->x);
					fp_neg(x2, x2);
					fp_mul(x3, y, w);
					fp_sqr(x3, x3);
					fp_add(x3, x3, t1);
				}
			}
		}

		ep_rhs(p->y, p->x);
		ep_rhs(v, x2);
		ep_rhs(w, x3);

		int c2 = fp_is_sqr(v);
		int c3 = fp_is_sqr(w);

		fp_copy_sec(p->y, v, c2);
		fp_copy_sec(p->x, x2, c2);
		fp_copy_sec(p->y, w, c3);
		fp_copy_sec(p->x, x3, c3);
                findsqroot(p->y, p->y);
		fp_neg(w, p->y);
		fp_copy_sec(p->y, w, fp_is_even(p->y) ^ s);
		fp_set_dig(p->z, 1);
		p->coord = BASIC;
		/* Multiply by cofactor. */
		ep_mul_cof(p, p);
	}
	RLC_CATCH_ANY {
		RLC_THROW(ERR_CAUGHT);
	}
	RLC_FINALLY {
		bn_free(k);
		fp_free(v);
		fp_free(w);
		fp_free(y);
		fp_free(t1);
		fp_free(t2);
		fp_free(x1);
		fp_free(x2);
		fp_free(x3);
		fp_free(d[0]);
		fp_free(d[1]);
		fp_free(d[2]);
		for (size_t i = 0; i < 8; i++) {
			fp_free(h[i]);
		}
	}
}

/*============================================================================*/
/* Public definitions                                                         */
/*============================================================================*/

#if EP_MAP == BASIC || !defined(STRIP)

void ep_map_basic(ep_t p, const uint8_t *msg, size_t len) {
	/* enough space for two field elements plus extra bytes for uniformity */
	const size_t elm = (FP_PRIME + ep_param_level() + 7) / 8;
	uint8_t *r = RLC_ALLOCA(uint8_t, elm);

	if (r == NULL) {
		RLC_THROW(ERR_NO_BUFFER);
		return;
	}

	RLC_TRY {
		md_xmd(r, elm, msg, len, (const uint8_t *)RLC_DSTAG, strlen(RLC_DSTAG));
		ep_map_basic_impl(p, r, elm);
	}
	RLC_CATCH_ANY {
		RLC_THROW(ERR_CAUGHT);
	}
	RLC_FINALLY {
		RLC_FREE(r);
	}
}

#endif

#if EP_MAP == SSWUM || !defined(STRIP)

void ep_map_sswum2(ep_t p, const uint8_t *msg, size_t len) {
	/* enough space for two field elements plus extra bytes for uniformity */
	const size_t elm = (FP_PRIME + ep_param_level() + 7) / 8;
	uint8_t *r = RLC_ALLOCA(uint8_t, 2 * elm);

	if (r == NULL) {
		RLC_THROW(ERR_NO_BUFFER);
		return;
	}

	RLC_TRY {
		/* for hash_to_field, need to hash to a pseudorandom string */
		/* XXX(rsw) the below assumes that we want to use MD_MAP for hashing.
		 *          Consider making the hash function a per-curve option!
		 */
		md_xmd(r, 2 * elm, msg, len, (const uint8_t *)RLC_DSTAG,
				sizeof(RLC_DSTAG));
		/* figure out which hash function to use */
		const int abNeq0 = (ep_curve_opt_a() != RLC_ZERO) &&
				(ep_curve_opt_b() != RLC_ZERO);
		void (*const map_fn)(ep_t, const fp_t) = 
				(ep_curve_is_ctmap() || abNeq0 ? ep_map_sswu : ep_map_svdw);

		ep_map_sswum_impl2(p, r, 2 * elm, map_fn);
	}
	RLC_CATCH_ANY {
		RLC_THROW(ERR_CAUGHT);
	}
	RLC_FINALLY {
		RLC_FREE(r);
	}
}

#endif

#if EP_MAP == SWIFT || !defined(STRIP)

void ep_map_swift2(ep_t p, const uint8_t *msg, size_t len) {
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
		md_xmd(r, 2*elm + 1, msg, len, (const uint8_t *)RLC_DSTAG,
				sizeof(RLC_DSTAG));

		ep_map_swift_impl2(p, r, 2 * elm + 1);
	}
	RLC_CATCH_ANY {
		RLC_THROW(ERR_CAUGHT);
	}
	RLC_FINALLY {
		RLC_FREE(r);
	}
}

#endif

size_t ep_map_rnd_size(void) {
	const size_t elm = (FP_PRIME + ep_param_level() + 7) / 8;

#if EP_MAP == BASIC
	return elm;
#elif EP_MAP == SSWUM
	return 2 * elm;
#elif EP_MAP == SWIFT
	return 2 * elm + 1;
#endif
}

void ep_map_rnd(ep_t p, const uint8_t *uniform_bytes, size_t len) {
	/* Make sure that input is long enough for any of the hash functons. */
	if (len < ep_map_rnd_size()) {
		RLC_THROW(ERR_NO_BUFFER);
		ep_set_infty(p);
		return;
	}

#if EP_MAP == BASIC
	ep_map_basic_impl2(p, uniform_bytes, len);
#elif EP_MAP == SWIFT
	ep_map_swift_impl2(p, uniform_bytes, len);
#elif EP_MAP == SSWUM
	/* figure out which hash function to use */
	const int abNeq0 = (ep_curve_opt_a() != RLC_ZERO) &&
			(ep_curve_opt_b() != RLC_ZERO);
	void (*const map_fn)(ep_t, const fp_t) = 
			(ep_curve_is_ctmap() || abNeq0 ? ep_map_sswu : ep_map_svdw);

	ep_map_sswum_impl2(p, uniform_bytes, len, map_fn);
#endif
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
        printf("Error setting BLS48 curve parameters.\n");
        core_clean();
        return 1;
    }


    
	const char *msg = "Hashing to BLS48-571 and this is the massege";
        size_t msg_len = strlen(msg);
        ep_t P;
        ep_null(P);
        ep_new(P);
	

	printf("Using curve: bls48-571, SWU\n");
   	MEASURE(ep_map_sswum2(P, (const uint8_t *)msg, msg_len);)
   	printf("RDTSC_clk_min=%f\n",RDTSC_clk_min);
	printf("RDTSC_clk_median=%f\n",RDTSC_clk_median);
	printf("RDTSC_clk_max=%f\n",RDTSC_clk_max);
        ep_print(P);
        
        printf("Using curve: bls48-571, SwiftEC\n");
	MEASURE(ep_map_swift2(P, (const uint8_t *)msg, msg_len);)
	printf("RDTSC_clk_min=%f\n",RDTSC_clk_min);
	printf("RDTSC_clk_median=%f\n",RDTSC_clk_median);
	printf("RDTSC_clk_max=%f\n",RDTSC_clk_max);
        ep_print(P);
        ep_free(P);

    core_clean();
    return 0;
}
