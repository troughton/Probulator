#pragma once

#include <Probulator/Experiments.h>

namespace Probulator {

class ExperimentSGBase : public Experiment
{
public:

    ExperimentSGBase& setLobeCountAndLambda(u32 lobeCount, float lambda)
    {
        m_lobeCount = lobeCount;
        m_lambda = lambda;
        return *this;
    }

    ExperimentSGBase& setAmbientLobeEnabled(bool state)
    {
        m_ambientLobeEnabled = state;
        return *this;
    }
    
    ExperimentSGBase& setNonNegativeSolve(bool state)
    {
        m_nonNegativeSolve = state;
        return *this;
    }

    void run(SharedData& data) override
    {
        generateLobes();
        solveForRadiance(data.m_radianceSamples);
        generateRadianceImage(data);
        generateIrradianceImage(data);
        generateSpecularImage(data);
    }

	void getProperties(std::vector<Property>& outProperties) override
	{
		Experiment::getProperties(outProperties);
		outProperties.push_back(Property("Lobe count", reinterpret_cast<int*>(&m_lobeCount)));
		outProperties.push_back(Property("Ambient lobe enabled", &m_ambientLobeEnabled));
		outProperties.push_back(Property("Lambda", &m_lambda));
	}

    bool m_nonNegativeSolve = false;
    bool m_ambientLobeEnabled = false;
    u32 m_lobeCount = 1;
    float m_lambda = 0.0f;
    SgBasis m_lobes;
    
    void generateLobes()
    {
        std::vector<vec3> sgLobeDirections(m_lobeCount);

        m_lobes.resize(m_lobeCount);
        for (u32 lobeIt = 0; lobeIt < m_lobeCount; ++lobeIt)
        {
            sgLobeDirections[lobeIt] = sampleVogelsSphere(lobeIt, m_lobeCount);

            m_lobes[lobeIt].p = sgLobeDirections[lobeIt];
            m_lobes[lobeIt].lambda = m_lambda;
            m_lobes[lobeIt].mu = vec3(0.0f);
        }

        if (m_ambientLobeEnabled)
        {
            SphericalGaussian lobe;
            lobe.p = vec3(0.0f, 0.0f, 1.0f);
            lobe.lambda = 0.0f; // cover entire sphere
            lobe.mu = vec3(0.0f);
            m_lobes.push_back(lobe);
        }
        
//        m_lobeCount = 12;
//        m_lambda = 3.4f;
//
//        const float kT = 0.618034f;
//
//        const vec3 vertexPositions[12] = {
//            vec3(1.0, kT, 0.0),
//            vec3(-1.0, kT, 0.0),
//            vec3(1.0, -kT, -0.0),
//            vec3(-1.0, -kT, 0.0),
//            vec3(0.0, 1.0, kT),
//            vec3(-0.0, -1.0, kT),
//            vec3(0.0, 1.0, -kT),
//            vec3(0.0, -1.0, -kT),
//            vec3(kT, 0.0, 1.0),
//            vec3(-kT, 0.0, 1.0),
//            vec3(kT, -0.0, -1.0),
//            vec3(-kT, -0.0, -1.0)
//        };
//
//        m_lobes.resize(m_lobeCount);
//        for (u32 lobeIt = 0; lobeIt < m_lobeCount; ++lobeIt)
//        {
//            m_lobes[lobeIt].p = normalize(vertexPositions[lobeIt]);
//            m_lobes[lobeIt].lambda = m_lambda;
//            m_lobes[lobeIt].mu = vec3(0.0f);
//        }

    }

protected:

    virtual void solveForRadiance(const std::vector<RadianceSample>& radianceSamples) = 0;


    void generateRadianceImage(const SharedData& data)
    {
        m_radianceImage = Image(data.m_outputSize);
        m_radianceImage.forPixels2D([&](vec4& pixel, ivec2 pixelPos)
        {
            vec3 direction = data.m_directionImage.at(pixelPos);
            vec3 sampleSg = sgBasisEvaluate(m_lobes, direction);
            pixel = vec4(sampleSg, 1.0f);
        });
    }

    void generateIrradianceImage(const SharedData& data)
    {
        
        m_irradianceImage = Image(data.m_outputSize);
        m_irradianceImage.forPixels2D([&](vec4& pixel, ivec2 pixelPos)
        {
            vec3 normal = data.m_directionImage.at(pixelPos);
            vec3 sampleSg = sgBasisIrradianceFitted(m_lobes, normal);
            pixel = vec4(sampleSg, 1.0f);
        });
    }
    
    void generateSpecularImage(const SharedData& data)
    {
        
        m_specularImage = Image(data.m_outputSize);
        m_specularImage.forPixels2D([&](vec4& pixel, ivec2 pixelPos)
                                      {
                                          
                                          if (specularFixedNormal) {
                                              vec3 normal = vec3(0, 1, 0);
                                              vec3 R = data.m_directionImage.at(pixelPos);
                                              if (R.y < 0.f) {
                                                  pixel = vec4(0, 0, 0, 1);
                                                  return;
                                              }
                                              
                                              vec3 V = reflect(-R, normal);
                                              
                                              vec3 sampleSG = vec3(0.f);
                                              
                                              for (const SphericalGaussian &lobe : m_lobes) {
                                                  sampleSG += sgGGXSpecular(lobe, normal, ggxAlpha, V, vec3(1.f));
                                              }
                                              pixel = vec4(sampleSG, 1.0f);
                                          } else {
                                              vec3 normal = data.m_directionImage.at(pixelPos);
                                              
                                              vec3 sampleSG = vec3(0.f);
                                              
                                              for (const SphericalGaussian &lobe : m_lobes) {
                                                  sampleSG += sgGGXSpecular(lobe, normal, ggxAlpha, normal, vec3(1.f));
                                              }
                                              pixel = vec4(sampleSG, 1.0f);
                                          }
                                      });
    }
};

class ExperimentSGNaive : public ExperimentSGBase
{
public:

    void solveForRadiance(const std::vector<RadianceSample>& radianceSamples) override
    {
        const u32 lobeCount = (u32)m_lobes.size();
        const u32 sampleCount = (u32)radianceSamples.size();
        const float normFactor = sgBasisNormalizationFactor(m_lambda, lobeCount);

        for (const RadianceSample& sample : radianceSamples)
        {
            for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
            {
                const SphericalGaussian& sg = m_lobes[lobeIt];
                float w = sgEvaluate(sg.p, sg.lambda, sample.direction);
                m_lobes[lobeIt].mu += sample.value * normFactor * (w / sampleCount);
            }
        }
    }
};
    
    class ExperimentSGRunningAverageOld : public ExperimentSGBase
    {
    public:
        void solveForRadiance(const std::vector<RadianceSample>& _radianceSamples) override
        {
            const u32 lobeCount = (u32)m_lobes.size();
            
            float lobeWeights[lobeCount];
            
            for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt) {
                lobeWeights[lobeIt] = 0.f;
            }
            
            std::vector<RadianceSample> radianceSamples = _radianceSamples;
            // The samples should be uniformly randomly distributed (or stratified) for best results.
            std::random_shuffle(radianceSamples.begin(), radianceSamples.end());
            
            float i = 0;
            for (size_t sampleIdx = 0; sampleIdx < radianceSamples.size(); sampleIdx += 1) {
                i += 1;
                
                const RadianceSample& sample = radianceSamples[sampleIdx];
                
                vec3 currentValue = vec3(0.f);
                
                float sampleLobeWeights[lobeCount];
                for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt) {
                    float dotProduct = dot(m_lobes[lobeIt].p, sample.direction);
                    float weight = exp(m_lobes[lobeIt].lambda * (dotProduct - 1.0));
                    currentValue += m_lobes[lobeIt].mu * weight;
                    
                    sampleLobeWeights[lobeIt] = weight;
                }
                
                
                for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt) {
                    float weight = sampleLobeWeights[lobeIt];
                    if (weight == 0.f) { continue; }
                    
                    lobeWeights[lobeIt] += weight;
                    
                    float sphericalIntegralEstimate = lobeWeights[lobeIt] / i;
                    
                    vec3 otherLobesContribution = currentValue - m_lobes[lobeIt].mu * weight;
                    vec3 deltaValue = weight * (sample.value - otherLobesContribution + (1 - weight) * m_lobes[lobeIt].mu) / sphericalIntegralEstimate;
                    
                    m_lobes[lobeIt].mu += (deltaValue - m_lobes[lobeIt].mu) / i;
                    
                    if (m_nonNegativeSolve) {
                        m_lobes[lobeIt].mu = max(m_lobes[lobeIt].mu, vec3(0.f));
                    }
                }
            }
        }
    };
    
class ExperimentSGRunningAverage : public ExperimentSGBase
{
public:
    
//    void solveForRadiance(const std::vector<RadianceSample>& radianceSamples) override
//    {
//        const u32 lobeCount = (u32)m_lobes.size();
//
//        float lobeMCSphericalIntegrals[lobeCount];
//
//        for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt) {
//            lobeMCSphericalIntegrals[lobeIt] = 0.f;
//        }
//
//        float lobePrecomputedSphericalIntegrals[lobeCount];
//        for (u64 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
//        {
//            lobePrecomputedSphericalIntegrals[lobeIt] = (1.f - exp(-4.f * m_lobes[lobeIt].lambda)) / (4 * m_lobes[lobeIt].lambda);
//        }
//
//        float totalSampleWeight = 0.f;
//
//        float i = 0.f;
//        for (const RadianceSample& sample : radianceSamples) {
//            i += 1.f;
//            const float sampleWeight = 1.f; //1.f - expf(-0.1 * i / float(radianceSamples.size()));
//            totalSampleWeight += sampleWeight;
//            float sampleWeightScale = sampleWeight / totalSampleWeight;
//
//            vec3 currentEstimate = vec3(0.f);
//
//            float sampleLobeWeights[lobeCount];
//            for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt) {
//                float dotProduct = dot(m_lobes[lobeIt].p, sample.direction);
//                float weight = exp(m_lobes[lobeIt].lambda * (dotProduct - 1.0));
//                currentEstimate += m_lobes[lobeIt].mu * weight;
//
//                sampleLobeWeights[lobeIt] = weight;
//            }
//
//            vec3 delta = sample.value - currentEstimate;
//
//            for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt) {
//                float weight = sampleLobeWeights[lobeIt];
//                if (weight == 0.f) { continue; }
//
//                float sphericalIntegralGuess = weight * weight;
//
//                // Update the MC-computed integral of the lobe over the domain.
//                lobeMCSphericalIntegrals[lobeIt] += (sphericalIntegralGuess - lobeMCSphericalIntegrals[lobeIt]) * sampleWeightScale;
//            }
//
//            vec3 lobeAmplitudeDeltas[lobeCount];
//            for (u64 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
//            {
//                lobeAmplitudeDeltas[lobeIt] = vec3(0);
//            }
//
//            for (u32 it = 0; it < 1; it += 1) {
//                for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt) {
//                    float weight = sampleLobeWeights[lobeIt];
//                    if (weight == 0.f) { continue; }
//
//                    // The most accurate method requires using the MC-computed integral,
//                    // since then bias in the estimate will partially cancel out.
//                    // However, if you don't want to store a weight per-lobe you can instead substitute it with the
//                    // precomputed integral at a slight increase in error.
//
//                    // Clamp the MC-computed integral to within a reasonable ad-hoc factor of the actual integral to avoid noise.
//                    float sphericalIntegral = max(lobeMCSphericalIntegrals[lobeIt], lobePrecomputedSphericalIntegrals[lobeIt]);
//
//                    vec3 projection = (m_lobes[lobeIt].mu + lobeAmplitudeDeltas[lobeIt]) * weight;
//                    vec3 newValue = (delta + projection) * weight / sphericalIntegral;
//
//                    lobeAmplitudeDeltas[lobeIt] = (newValue - m_lobes[lobeIt].mu) * sampleWeightScale;
//
////                    delta += projection - (m_lobes[lobeIt].mu + lobeAmplitudeDeltas[lobeIt]) * weight;
//                }
//            }
//
//
//            for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt) {
//                m_lobes[lobeIt].mu += (newValue - m_lobes[lobeIt].mu) * sampleWeightScale;
//
//                if (m_nonNegativeSolve) {
//                    m_lobes[lobeIt].mu = max(m_lobes[lobeIt].mu, vec3(0.f));
//                }
//            }
//
//        }
//    }
    
    void solveForRadiance(const std::vector<RadianceSample>& radianceSamples) override
    {
        const u32 lobeCount = (u32)m_lobes.size();
        
        float lobeMCSphericalIntegrals[lobeCount];
        float lobePrecomputedSphericalIntegrals[lobeCount];
        
        for (u64 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
        {
            lobeMCSphericalIntegrals[lobeIt] = 0.f;
            lobePrecomputedSphericalIntegrals[lobeIt] = (1.f - exp(-4.f * m_lobes[lobeIt].lambda)) / (4 * m_lobes[lobeIt].lambda);
        }
        
        float totalSampleWeight = 0.f;
        
        
        for (const RadianceSample& sample : radianceSamples) {
            const float sampleWeight = 1.f;
            totalSampleWeight += sampleWeight;
            float sampleWeightScale = sampleWeight / totalSampleWeight;
            
            vec3 delta = sample.value;
            
            float sampleLobeWeights[lobeCount];
            
            for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt) {
                float dotProduct = dot(m_lobes[lobeIt].p, sample.direction);
                float weight = exp(m_lobes[lobeIt].lambda * (dotProduct - 1.0f));
                delta -= m_lobes[lobeIt].mu * weight;
                
                sampleLobeWeights[lobeIt] = weight;
            }
            
            for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt) {
                float weight = sampleLobeWeights[lobeIt];
                if (weight == 0.f) { continue; }
                
                float sphericalIntegralGuess = weight * weight;
                
                // Update the MC-computed integral of the lobe over the domain.
                lobeMCSphericalIntegrals[lobeIt] += (sphericalIntegralGuess - lobeMCSphericalIntegrals[lobeIt]) * sampleWeightScale;
                
                // The most accurate method requires using the MC-computed integral,
                // since then bias in the estimate will partially cancel out.
                // However, if you don't want to store a weight per-lobe you can instead substitute it with the
                // precomputed integral at a slight increase in error.
                
                // Clamp the MC-computed integral to within a reasonable ad-hoc factor of the actual integral to avoid noise.
//                float sphericalIntegral = max(lobeMCSphericalIntegrals[lobeIt], lobePrecomputedSphericalIntegrals[lobeIt]);
                float sphericalIntegral = sampleWeightScale + (1.f - sampleWeightScale) * lobeMCSphericalIntegrals[lobeIt];
                
                //                vec3 newValue = (delta + projection) * weight / sphericalIntegral;
                //                m_lobes[lobeIt].mu += (newValue - m_lobes[lobeIt].mu) * sampleWeightScale;
                
                float deltaScale = weight * sampleWeightScale / sphericalIntegral;
//                float dampingTerm = 1.f + (weight * weight / sphericalIntegral - 1.f) * sampleWeightScale;
//                m_lobes[lobeIt].mu *= dampingTerm;

                m_lobes[lobeIt].mu += 3.f * delta * deltaScale;
                
                if (m_nonNegativeSolve) {
                    m_lobes[lobeIt].mu = max(m_lobes[lobeIt].mu, vec3(0.f));
                }
                
                delta *= 1.0f - deltaScale * weight;
            }
        }
    }
};
    
    
class ExperimentSGBakingLab : public ExperimentSGBase
{
    public:
    
    void solveForRadiance(const std::vector<RadianceSample>& _radianceSamples) override
    {
        const u32 lobeCount = (u32)m_lobes.size();
        
        std::vector<RadianceSample> radianceSamples = _radianceSamples;
        std::random_shuffle(radianceSamples.begin(), radianceSamples.end());
        
        // Project color samples onto the SGs
        for (const RadianceSample& sample : radianceSamples) {
            for (size_t i = 0; i < lobeCount; ++i) {
                SphericalGaussian sg1;
                SphericalGaussian sg2;
                sg1.mu = m_lobes[i].mu;
                sg1.p = m_lobes[i].p;
                sg1.lambda = m_lobes[i].lambda;
                sg2.mu = sample.value;
                sg2.p = normalize(sample.direction);
                
                if (dot(sample.direction, sg1.p) > 0.0f) {
                    float dotRes = dot(sg1.p, sg2.p);
                    float factor = (dotRes - 1.0f) * sg1.lambda;
                    float wgt = exp(factor);
                    m_lobes[i].mu += sg2.mu * wgt;
                }
            }
        }
        
        // Weight the samples by the monte carlo factor for uniformly sampling the sphere
        float monteCarloFactor = ((4.0f * pi) / radianceSamples.size());
          for (size_t i = 0; i < lobeCount; ++i) {
              m_lobes[i].mu *= monteCarloFactor;
          }
    }
};

class ExperimentSGLS : public ExperimentSGBase
{
public:

    void solveForRadiance(const std::vector<RadianceSample>& radianceSamples) override
    {
        m_lobes = sgFitLeastSquares(m_lobes, radianceSamples);
        
        printf("\nLeast Square Lobe µs:\n");
        for (u32 lobeIt = 0; lobeIt < m_lobeCount; ++lobeIt)
        {
            printf("%.3f, %.3f, %.3f\n", m_lobes[lobeIt].mu.x, m_lobes[lobeIt].mu.y, m_lobes[lobeIt].mu.z);
        }
    }
};

class ExperimentSGNNLS : public ExperimentSGBase
{
public:

    void solveForRadiance(const std::vector<RadianceSample>& radianceSamples) override
    {
        m_lobes = sgFitNNLeastSquares(m_lobes, radianceSamples);
        
        printf("\nNNLS Lobe µs:\n");
        for (u32 lobeIt = 0; lobeIt < m_lobeCount; ++lobeIt)
        {
            printf("%.3f, %.3f, %.3f\n", m_lobes[lobeIt].mu.x, m_lobes[lobeIt].mu.y, m_lobes[lobeIt].mu.z);
        }
    }
};

class ExperimentSGGA : public ExperimentSGBase
{
public:

    u32 m_populationCount = 50;
    u32 m_generationCount = 2000;

    ExperimentSGBase& setPopulationAndGenerationCount(u32 populationCount, u32 generationCount)
    {
        m_populationCount = populationCount;
        m_generationCount = generationCount;
        return *this;
    }

    void solveForRadiance(const std::vector<RadianceSample>& radianceSamples) override
    {
        m_lobes = sgFitNNLeastSquares(m_lobes, radianceSamples); // NNLS is used to seed GA
        m_lobes = sgFitGeneticAlgorithm(m_lobes, radianceSamples, m_populationCount, m_generationCount);
    }

	void getProperties(std::vector<Property>& outProperties) override
	{
		ExperimentSGBase::getProperties(outProperties);
		outProperties.push_back(Property("Population count", reinterpret_cast<int*>(&m_populationCount)));
		outProperties.push_back(Property("Generation count", reinterpret_cast<int*>(&m_generationCount)));
	}
};

}
