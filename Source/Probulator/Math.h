#pragma once

#include "Common.h"

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace Probulator
{
	static const float pi = glm::pi<float>();
	static const float twoPi = 2.0f * glm::pi<float>();
	static const float fourPi = 4.0f * glm::pi<float>();

	using glm::vec2;
	using glm::vec3;
	using glm::vec4;

	using glm::ivec2;
	using glm::ivec3;
	using glm::ivec4;

	using glm::mat2;
	using glm::mat3;
	using glm::mat4;

	using glm::abs;
	using glm::atan;
	using glm::cos;
	using glm::exp;
	using glm::log;
	using glm::max;
	using glm::min;
	using glm::mix;
	using glm::sin;
	using glm::sinh;
	using glm::sqrt;

	template <typename T>
	inline T sqr(const T& x)
	{
		return x*x;
	}

	template <typename T>
	inline T cube(const T& x)
	{
		return x*x*x;
	}

	inline float saturate(float x)
	{
		return max(0.0f, min(x, 1.0f));
	}

	template <typename T>
	inline float dotMax0(const T& a, const T& b)
	{
		return max(0.0f, dot(a, b));
	}

	template <typename T>
	inline float dotSaturate(const T& a, const T& b)
	{
		return saturate(dot(a, b));
	}

	inline void sinCos(float x, float* outSinX, float* outCosX)
	{
		*outSinX = sin(x);
		*outCosX = cos(x);
	}

	inline mat3 makeOrthogonalBasis(vec3 n)
	{
        float sign = copysign(1.0f, n.z);
        const float a = -1.0f / (sign + n.z);
        const float b = n.x * n.y * a;
        vec3 b1(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
        vec3 b2(b, sign + n.y * n.y * a, -n.y);
		
		return mat3(b1, b2, n);
	}
    
    inline glm::dmat3 makeOrthogonalBasisDouble(glm::dvec3 n)
    {
        double sign = copysign(1.0, n.z);
        const double a = -1.0 / (sign + n.z);
        const double b = n.x * n.y * a;
        glm::dvec3 b1(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
        glm::dvec3 b2(b, sign + n.y * n.y * a, -n.y);
        
        return glm::dmat3(b1, b2, n);
    }

	inline float latLongTexelArea(ivec2 pos, ivec2 imageSize)
	{
		vec2 uv0 = vec2(pos) / vec2(imageSize);
		vec2 uv1 = vec2(pos + 1) / vec2(imageSize);

		float theta0 = pi*(uv0.x*2.0f - 1.0f);
		float theta1 = pi*(uv1.x*2.0f - 1.0f);

		float phi0 = pi*(uv0.y - 0.5f);
		float phi1 = pi*(uv1.y - 0.5f);

		return abs(theta1 - theta0) * abs(sin(phi1) - sin(phi0));
	}
    
    inline double latLongTexelAreaDouble(ivec2 pos, ivec2 imageSize)
    {
        vec2 uv0 = vec2(pos) / vec2(imageSize);
        vec2 uv1 = vec2(pos + 1) / vec2(imageSize);
        
        double theta0 = M_PI*(uv0.x*2.0 - 1.0);
        double theta1 = M_PI*(uv1.x*2.0 - 1.0);
        
        double phi0 = M_PI*(uv0.y - 0.5);
        double phi1 = M_PI*(uv1.y - 0.5);
        
        return abs(theta1 - theta0) * abs(sin(phi1) - sin(phi0));
    }


	inline vec2 cartesianToLatLongTexcoord(vec3 p)
	{
		// http://gl.ict.usc.edu/Data/HighResProbes

		float u = (1.0f + atan(p.x, -p.y) / pi);
		float v = acos(p.z) / pi;

		return vec2(u * 0.5f, v);
	}

	inline vec3 latLongTexcoordToCartesian(vec2 uv)
	{
		// http://gl.ict.usc.edu/Data/HighResProbes

		float theta = pi*(uv.x*2.0f - 1.0f);
		float phi = pi*uv.y;

		float x = sin(phi)*sin(theta);
        float y = -sin(phi)*cos(theta);
		float z = cos(phi);

		return vec3(x, y, z);
	}

	inline vec3 sphericalToCartesian(vec2 thetaPhi)
	{
		// https://graphics.stanford.edu/papers/envmap/envmap.pdf

		float theta = thetaPhi.x;
		float phi = thetaPhi.y;

		float sinTheta, cosTheta;
		sinCos(theta, &sinTheta, &cosTheta);

		float sinPhi, cosPhi;
		sinCos(phi, &sinPhi, &cosPhi);

		float x = sinTheta * cosPhi;
		float y = sinTheta * sinPhi;
		float z = cosTheta;

		return vec3(x, y, z);
	}

	inline vec2 cartesianToSpherical(vec3 p)
	{
		// https://graphics.stanford.edu/papers/envmap/envmap.pdf

		float phi = atan(p.y, p.x);
		float theta = acos(p.z);

		return vec2(theta, phi);
	}


	inline vec2 sampleHammersley(u32 i, u32 n)
	{
		u32 bits = i;
		bits = (bits << 16u) | (bits >> 16u);
		bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
		bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
		bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
		bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
		float vdc = float(bits) * 2.3283064365386963e-10f;
		return vec2(float(i) / float(n), vdc);
	}
    
    inline glm::dvec2 sampleHammersleyDouble(u32 i, u32 n)
    {
        u32 bits = i;
        bits = (bits << 16u) | (bits >> 16u);
        bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
        bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
        bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
        bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
        double vdc = float(bits) * 2.3283064365386963e-10f;
        return glm::dvec2(double(i) / double(n), vdc);
    }
    
    inline float sampleHalton(u32 index, u32 base)
    {
        float f = 1.f;
        float r = 0.f;
        
        while (index > 0) {
            f = f / float(base);
            r += f * float(index % base);
            index /= base;
        }
        return r;
    }
    
    inline double sampleHaltonDouble(u32 index, u32 base)
    {
        double f = 1.f;
        double r = 0.f;
        
        while (index > 0) {
            f = f / double(base);
            r += f * double(index % base);
            index /= base;
        }
        return r;
    }

	inline vec3 sampleUniformHemisphere(float u, float v)
	{
		float phi = v * twoPi;
		float cosTheta = u;
		float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

		float sinPhi, cosPhi;
		sinCos(phi, &sinPhi, &cosPhi);

		return vec3(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
	}
    
    inline glm::dvec3 sampleUniformHemisphereDouble(double u, double v)
    {
        double phi = v * twoPi;
        double cosTheta = u;
        double sinTheta = sqrt(1.0f - cosTheta * cosTheta);
        
        double sinPhi = sin(phi);
        double cosPhi = cos(phi);
        
        return glm::dvec3(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
    }

	inline vec3 sampleVogelsSphere(u32 i, u32 n)
	{
		// http://blog.marmakoide.org/?p=1

		float goldenAngle = pi * (3.0f - sqrt(5.0f));

		float theta = goldenAngle * i;
		float t = n > 1 ? float(i) / (n - 1) : 0.5f;
		float z = mix(1.0f - 1.0f / n, 1.0f / n - 1.0f, t);
		float radius = sqrt(1.0f - z*z);

		float sinTheta, cosTheta;
		sinCos(theta, &sinTheta, &cosTheta);

		float x = radius * cosTheta;
		float y = radius * sinTheta;

		return vec3(x, y, z);
	}

	inline vec3 sampleUniformSphere(float u, float v)
	{
		float z = 1.0f - 2.0f * u;
		float r = sqrt(max(0.0f, 1.0f - z*z));
		float phi = twoPi * v;

		float sinPhi, cosPhi;
		sinCos(phi, &sinPhi, &cosPhi);

		float x = r * cosPhi;
		float y = r * sinPhi;

		return vec3(x, y, z);
	}
    
    inline glm::dvec3 sampleUniformSphereDouble(double u, double v)
    {
        double z = 1.0 - 2.0 * u;
        double r = sqrt(max(0.0, 1.0 - z*z));
        double phi = 2 * M_PI * v;
        
        double sinPhi = sin(phi);
        double cosPhi = cos(phi);
        
        double x = r * cosPhi;
        double y = r * sinPhi;
        
        return glm::dvec3(x, y, z);
    }

	inline vec3 sampleUniformSphere(vec2 uv)
	{
		return sampleUniformSphere(uv.x, uv.y);
	}
    
    inline glm::dvec3 sampleUniformSphereDouble(glm::dvec2 uv)
    {
        return sampleUniformSphereDouble(uv.x, uv.y);
    }

	inline vec3 sampleCosineHemisphere(float u, float v)
	{
		float phi = v * twoPi;
		float cosTheta = sqrt(u);
		float sinTheta = sqrt(1.0f - u);

		float sinPhi, cosPhi;
		sinCos(phi, &sinPhi, &cosPhi);

		float x = cosPhi * sinTheta;
		float y = sinPhi * sinTheta;
		float z = cosTheta;

		return vec3(x, y, z);
	}
    
    inline glm::dvec3 sampleCosineHemisphereDouble(double u, double v)
    {
        double phi = v * 2 * M_PI;
        double cosTheta = sqrt(u);
        double sinTheta = sqrt(1.0 - u);
        
        double sinPhi = sin(phi);
        double cosPhi = cos(phi);
        
        double x = cosPhi * sinTheta;
        double y = sinPhi * sinTheta;
        double z = cosTheta;
        
        return glm::dvec3(x, y, z);
    }

    inline glm::dvec3 sampleCosineHemisphereDouble(const glm::dvec2& uv)
    {
        return sampleCosineHemisphereDouble(uv.x, uv.y);
    }
    
	inline vec3 sampleCosineHemisphere(const vec2& uv)
	{
		return sampleCosineHemisphere(uv.x, uv.y);
	}

	inline float rgbLuminance(const vec3& color)
	{
		return dot(vec3(0.2126f, 0.7152f, 0.0722f), color);
	}
    
    // Building an Orthonormal Basis, Revisited. Duff et. al., Pixar. http://jcgt.org/published/0006/01/01/paper.pdf
    inline void constructOrthonormalBasis(const vec3 &n, vec3 *b1, vec3 *b2) {
        float sign = copysign(1.0f, n.z);
        const float a = -1.0f / (sign + n.z);
        const float b = n.x * n.y * a;
        *b1 = vec3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
        *b2 = vec3(b, sign + n.y * n.y * a, -n.y);
    }

    inline vec3 F_Schlick(const vec3 &f0, float f90, float u) {
        return f0 + (f90 - f0) * pow(1.f - u, 5.f);
    }
    
    inline float F_Schlick(const float f0, float f90, float u) {
        return f0 + (f90 - f0) * pow(1.f - u, 5.f);
    }
    
    // Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs, http://jcgt.org/published/0003/02/03/paper.pdf
    inline float SmithLambda(float cosThetaM, float alphaG) {
        float alphaG2 = alphaG * alphaG;
        
        float cosThetaM2 = cosThetaM * cosThetaM;
        float sinThetaM2 = 1 - cosThetaM2;
        
        return 0.5f * (-1.f + sqrt(1.f + alphaG2 * sinThetaM2 / cosThetaM2));
    }
    
    // Multiple-Scattering Microfacet BSDFs with the Smith Model
    // Heitz et. al.
    // https://jo.dreggn.org/home/2016_microfacets.pdf
    // Incoming is the view direction and outgoing is the light direction
    inline float SmithGGXMaskingShadowingG1Reflection(vec3 incoming, vec3 outgoing, vec3 microfacetNormal, float alpha) {
        float VdotH = dot(incoming, microfacetNormal);
        return (VdotH >= 0.f ? 1.f : -1.f) / (1.f + SmithLambda(VdotH, alpha));
    }
    
    inline float V_SmithGGXCorrelated(float NdotL, float NdotV, float alphaG) {
        float alphaG2 = alphaG * alphaG;
        
        float Lambda_GGXV = NdotL * sqrt((-NdotV * alphaG2 + NdotV) * NdotV + alphaG2);
        
        float Lambda_GGXL = NdotV * sqrt((-NdotL * alphaG2 + NdotL) * NdotL + alphaG2);
        
        return 0.5f / (max(Lambda_GGXV + Lambda_GGXL, 1e-6f));
    }
    
    inline float D_GGX(float NdotH, float alpha) {
        float a2 = alpha * alpha;
        float f = (NdotH * a2 - NdotH) * NdotH + 1;
        return a2 / max(f * f, 1e-24f);
    }
    
    // Multiple-Scattering Microfacet BSDFs with the Smith Model
    // Heitz et. al.
    // https://jo.dreggn.org/home/2016_microfacets.pdf
    // Incoming is the view direction and outgoing is the light direction
    inline float SmithGGXMaskingShadowingG2OverG1Reflection(vec3 incoming, vec3 outgoing, vec3 microfacetNormal, float alpha) {
        float VdotH = dot(incoming, microfacetNormal);
        float LdotH = dot(outgoing, microfacetNormal);
        
        float G1Inverse = 1.f + SmithLambda(incoming.z, alpha);
        
        float numerator = (VdotH > 0 ? 1.f : 0.f) * (LdotH > 0 ? 1.f : 0.f);
        float denominator = G1Inverse + SmithLambda(outgoing.z, alpha);
        return numerator / (denominator * G1Inverse);
    }
    
    // Eric Heitz. A Simpler and Exact Sampling Routine for the GGX Distribution of Visible Normals.
    // [Research Report] Unity Technologies. 2017. <hal-01509746>
    // https://developer.blender.org/D3461
    inline vec3 sampleGGXVNDF(const vec3 &V_, float alpha_x, float alpha_y, float U1, float U2) {
        // stretch view
        vec3 V = normalize(vec3(alpha_x * V_.x, alpha_y * V_.y, V_.z));
        float r = sqrt(U1);
        
        // Handle special case of normal incidence and/or near-specularity.
        //    if (V.z >= 0.9999f) {
        //        return vec3(0, 0, 1);
        //    }
        
        // orthonormal basis
        vec3 T1 = (V.z < 0.9999) ? normalize(cross(V, vec3(0,0,1))) : vec3(1,0,0);
        vec3 T2 = cross(T1, V);
        
        // sample point with polar coordinates (r, phi)
        float a = 1.0 / (1.0 + V.z);
        float phi = (U2<a) ? U2/a * pi : pi + (U2-a)/(1.0-a) * pi;
        float P1 = r*cos(phi);
        float P2 = r*sin(phi)*((U2<a) ? 1.0 : V.z);
        // compute normal
        vec3 N = P1*T1 + P2*T2 + sqrt(max(0.0f, 1.0f - P1*P1 - P2*P2))*V;
        // unstretch
        N = normalize(vec3(alpha_x*N.x, alpha_y*N.y, max(0.0f, N.z)));
        return N;
    }
}
