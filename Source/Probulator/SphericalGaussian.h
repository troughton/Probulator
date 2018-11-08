#pragma once

#include "Math.h"

namespace Probulator
{
	// All-Frequency Rendering of Dynamic, Spatially-Varying Reflectance
	// http://research.microsoft.com/en-us/um/people/johnsny/papers/sg.pdf
	struct SphericalGaussian
	{
		vec3 p; // lobe axis
		float lambda; // sharpness
		vec3 mu; // amplitude
        
        inline float weight(const vec3 &direction) const {
            return exp(lambda * (dot(direction, p) - 1.f));
        }
        
        inline vec3 evaluate(const vec3 &direction) const {
            return mu * this->weight(direction);
        }
	};
    
    struct AnisotropicSphericalGaussian {
        vec3 mu;
        vec3 basisZ;
        vec3 basisX;
        vec3 basisY;
        float lambdaX;
        float lambdaY;
        
        vec3 evaluate(vec3 direction) const;
        vec3 convolvedWithSG(const SphericalGaussian sg) const;
    };

	// Calculates an integral of a product of two SGs over a sphere
	vec3 sgDot(const SphericalGaussian& a, const SphericalGaussian& b);

	// Calculates an SG that evaluates to a product of two SGs
	SphericalGaussian sgCross(const SphericalGaussian& a, const SphericalGaussian& b);

	// Evaluate Spherical Gaussian for a given position on a sphere
	float sgEvaluate(const vec3& p, float lambda, const vec3& v);
	vec3 sgEvaluate(const SphericalGaussian& sg, const vec3& v);

	// Calculates spherical integral for SG with mu=1.0 and given lambda
	float sgIntegral(float lambda);

	// SG fitted for a cosine lobe
	inline float sgCosineMu() { return 1.170f; }
	inline float sgCosineLambda() { return 2.133f; }
	inline SphericalGaussian sgCosineLobe(const vec3& p = vec3(0.0f, 0.0f, 1.0f))
	{
		SphericalGaussian sg;
		sg.mu = vec3(sgCosineMu());
		sg.lambda = sgCosineLambda();
		sg.p = p;
		return sg;
	}

	// Find SG mu for a given lambda that will integrate to a given value over a sphere
	float sgFindMu(float targetLambda, float targetIntegral);

	// Find SG mu for a given lambda to match total energy of another SG
	float sgFindMu(float targetLambda, float lambda, float mu);

    // Compute the irradiance in a given direction using a curve fit.
    vec3 sgIrradianceFitted(const SphericalGaussian& lightingLobe, const vec3& normal);
    
    vec3 sgGGXSpecular(const SphericalGaussian &sg, vec3 normal, float roughness, vec3 view, vec3 f0);
}
