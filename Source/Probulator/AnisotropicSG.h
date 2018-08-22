#pragma once

#include "Math.h"
#include "SphericalGaussian.h"

namespace Probulator
{
    // Anisotropic Spherical Gaussians
    // http://cg.cs.tsinghua.edu.cn/people/%7Ekun/asg/paper_asg.pdf
    struct AnisotropicSG
    {
        vec3 amplitude;
        vec3 basisZ;
        vec3 basisX;
        vec3 basisY;
        float lambdaX; // Sharpness X
        float lambdaY; // Sharpness Y
    };
    
    inline vec3 asgEvaluate(const AnisotropicSG& asg, const vec3& v) {
        float sTerm = saturate(dot(asg.basisZ, dir));
        float lambdaTerm = asg.lambdaX * dot(dir, asg.basisX) * dot(dir, asg.basisX);
        float muTerm     = asg.lambdaY * dot(dir, asg.basisY) * dot(dir, asg.basisY);
        return asg.amplitude * sTerm * exp(-lambdaTerm - muTerm);
    }
 
    inline vec3 convolveASGWithSG(const AnisotropicSG& asg, const SphericalGaussian& sg) {
        // The ASG paper specifes an isotropic SG as
        // exp(2 * nu * (dot(v, axis) - 1)),
        // so we must divide our SG sharpness by 2 in order
        // to get the nup parameter expected by the ASG formula
        float nu = sg.lambda * 0.5f;
        
        AnisotropicSG convolveASG;
        convolveASG.basisX = asg.basisX;
        convolveASG.basisY = asg.basisY;
        convolveASG.basisZ = asg.basisZ;
        
        convolveASG.lambdaX = (nu * asg.lambdaX) / (nu + asg.lambdaX);
        convolveASG.lambdaY = (nu * asg.lambdaY) / (nu + asg.lambdaY);
        
        convolveASG.amplitude = pi / sqrt((nu + asg.lambdaX) *
                                          (nu + asg.lambdaY));
        
        vec3 asgResult = asgEvaluate(convolveASG, sg.p);
        return asgResult * sg.mu * asg.amplitude;
    }
}
