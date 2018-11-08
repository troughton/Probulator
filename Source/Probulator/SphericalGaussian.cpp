#include "SphericalGaussian.h"

#include <stdio.h>

namespace Probulator
{
	float sgIntegral(float lambda)
	{
		return fourPi * (0.5f - 0.5f*exp(-2.0f*lambda)) / lambda;
	}

	vec3 sgDot(const SphericalGaussian& a, const SphericalGaussian& b)
	{
		float dM = length(a.lambda*a.p + b.lambda*b.p);
		vec3 num = sinh(dM) * fourPi * a.mu * b.mu;
		float den = exp(a.lambda + b.lambda) * dM;
		return num / den;
	}

	float sgEvaluate(const vec3& p, float lambda, const vec3& v)
	{
		float dp = dot(v, p);
		return exp(lambda * (dp - 1.0f));
	}

	vec3 sgEvaluate(const SphericalGaussian& sg, const vec3& v)
	{
		return sg.mu * sgEvaluate(sg.p, sg.lambda, v);
	}

	SphericalGaussian sgCross(const SphericalGaussian& a, const SphericalGaussian& b)
	{
		vec3 pM = (a.lambda*a.p + b.lambda*b.p) / (a.lambda + b.lambda);
		float pMLength = length(pM);
		float lambdaM = a.lambda + b.lambda;

		SphericalGaussian r;
		r.p = pM / pMLength;
		r.lambda = lambdaM * pMLength;
		r.mu = a.mu * b.mu * exp(lambdaM * (pMLength - 1.0f));

		return r;
	}

	float sgFindMu(float targetLambda, float targetIntegral)
	{
		return targetIntegral / sgIntegral(targetLambda);
	}

	float sgFindMu(float targetLambda, float lambda, float mu)
	{
		float targetIntegral = sgIntegral(lambda)*mu;
		return sgFindMu(targetLambda, targetIntegral);
	}

    // Stephen Hill [2016], https://mynameismjp.wordpress.com/2016/10/09/sg-series-part-3-diffuse-lighting-from-an-sg-light-source/
    vec3 sgIrradianceFitted(const SphericalGaussian& lightingLobe, const vec3& normal)
    {
        if(lightingLobe.lambda == 0.f)
            return lightingLobe.mu;
        
        const float muDotN = dot(lightingLobe.p, normal);
        const float lambda = lightingLobe.lambda;
        
        const float c0 = 0.36f;
        const float c1 = 1.0f / (4.0f * c0);
        
        float eml  = exp(-lambda);
        float em2l = eml * eml;
        float rl   = 1.f / lambda;
        
        float scale = 1.0f + 2.0f * em2l - rl;
        float bias  = (eml - em2l) * rl - em2l;
        
        float x  = sqrt(1.0f - scale);
        float x0 = c0 * muDotN;
        float x1 = c1 * x;
        
        float n = x0 + x1;
        
        float y = saturate(muDotN);
        if(abs(x0) <= x1)
            y = n * n / x;
        
        float result = scale * y + bias;
        
        return result * lightingLobe.mu * sgIntegral(lightingLobe.lambda);
    }
    
    
    inline SphericalGaussian DistributionTermSG(vec3 direction, float roughness)
    {
        float m2 = roughness * roughness;
        SphericalGaussian distribution;
        distribution.mu = vec3(1.f / (pi * m2));
        distribution.p = direction;
        distribution.lambda = 2.f / m2;
        
        return distribution;
    }
    
    AnisotropicSphericalGaussian sgWarpDistribution(const SphericalGaussian &sg, vec3 view) {
        AnisotropicSphericalGaussian warp;
        
        // Generate any orthonormal basis with Z pointing in the
        // direction of the reflected view vector
        warp.basisZ = reflect(-view, sg.p);
        
        constructOrthonormalBasis(warp.basisZ, &warp.basisX, &warp.basisY);
        
        float dotDirO = max(dot(view, sg.p), 0.0001f);
        
        // Second derivative of the sharpness with respect to how
        // far we are from basis Axis direction
        warp.lambdaX = sg.lambda / (8.0f * dotDirO * dotDirO);
        warp.lambdaY = sg.lambda / 8.0f;
        
        warp.mu = sg.mu;
        
        return warp;
    }
    
    vec3 sgGGXSpecular(const SphericalGaussian &sg, vec3 normal, float roughness, vec3 view, vec3 f0) {
        // Create an SG that approximates the NDF
        SphericalGaussian ndf = DistributionTermSG(normal, roughness);
        
        // Apply a warpring operation that will bring the SG from
        // the half-angle domain the the the lighting domain.
        AnisotropicSphericalGaussian warpedNDF = sgWarpDistribution(ndf, view);
        
        // Convolve the NDF with the light
        vec3 output = warpedNDF.convolvedWithSG(sg);
        
        // Parameters needed for evaluating the visibility term
        vec3 warpDir = warpedNDF.basisZ;
        float nDotL = saturate(dot(normal, warpDir));
        float nDotV = saturate(dot(normal, view));
        vec3 h = normalize(warpDir + view);
        
        // Visibility term
        output *= V_SmithGGXCorrelated(nDotL, nDotV, roughness);
        
        // Fresnel
        output *= F_Schlick(f0, 1.f, dot(warpDir, h));
        
        // Cosine term
        output *= nDotL;
        
        return output; // max(output, 0.0f);
    }
    
    vec3 AnisotropicSphericalGaussian::evaluate(vec3 dir) const {
        float sTerm = saturate(dot(this->basisZ, dir));
        float lambdaTerm = this->lambdaX * dot(dir, this->basisX)
        * dot(dir, this->basisX);
        float muTerm = this->lambdaY * dot(dir, this->basisY)
        * dot(dir, this->basisY);
        return this->mu * sTerm * exp(-lambdaTerm - muTerm);
    }
    
    vec3 AnisotropicSphericalGaussian::convolvedWithSG(const SphericalGaussian sg) const {
        // The ASG paper specifes an isotropic SG as
        // exp(2 * nu * (dot(v, axis) - 1)),
        // so we must divide our SG sharpness by 2 in order
        // to get the nup parameter expected by the ASG formula
        float nu = sg.lambda * 0.5f;
        
        AnisotropicSphericalGaussian convolveASG;
        convolveASG.basisX = this->basisX;
        convolveASG.basisY = this->basisY;
        convolveASG.basisZ = this->basisZ;
        
        convolveASG.lambdaX = (nu * this->lambdaX) /
        (nu + this->lambdaX);
        convolveASG.lambdaY = (nu * this->lambdaY) /
        (nu + this->lambdaY);
        
        convolveASG.mu = vec3(pi / sqrt((nu + this->lambdaX) * (nu + this->lambdaY)));
        
        vec3 asgResult = convolveASG.evaluate(sg.p);
        return asgResult * sg.mu * this->mu;
    }

}
