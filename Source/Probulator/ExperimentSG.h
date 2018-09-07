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
    
class ExperimentSGRunningAverage : public ExperimentSGBase
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
        
        for (const RadianceSample& sample : radianceSamples) {
            // What's the current value of the SG in the sample's direction?
            vec3 currentValue = vec3(0.f);
            
            float sampleLobeWeights[lobeCount];
            float sampleLobeWeightSum = 0.f;
            for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt) {
                float dotProduct = dot(m_lobes[lobeIt].p, sample.direction);
                float weight = exp(m_lobes[lobeIt].lambda * (dotProduct - 1.0));
                currentValue += m_lobes[lobeIt].mu * weight;
                
                sampleLobeWeights[lobeIt] = weight;
                sampleLobeWeightSum += weight;
            }
            
            vec3 deltaValue = sample.value - currentValue;
            
            for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt) {
                float weight = sampleLobeWeights[lobeIt]; // / sampleLobeWeightSum; // Not dividing by the total weight seems to give slightly better results.
                if (weight == 0.f) { continue; }
                
                lobeWeights[lobeIt] += weight;
                
                float weightScale = weight / lobeWeights[lobeIt];
                m_lobes[lobeIt].mu += deltaValue * weightScale;
                
                if (m_nonNegativeSolve) {
                    m_lobes[lobeIt].mu = max(m_lobes[lobeIt].mu, vec3(0.f));
                }
            }
        }
    }
    
//    void solveForRadiance(const std::vector<RadianceSample>& _radianceSamples) override
//    {
//        const u32 lobeCount = (u32)m_lobes.size();
//
//        float lobeWeights[lobeCount];
//
//        for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
//        {
//            lobeWeights[lobeIt] = 0.f;
//        }
//
//        std::vector<RadianceSample> radianceSamples = _radianceSamples;
//
//        std::random_shuffle(radianceSamples.begin(), radianceSamples.end());
//
//        for (u32 sampleIdx = 0; sampleIdx < radianceSamples.size(); sampleIdx += 1)
//        {
//            const RadianceSample& sample = radianceSamples[sampleIdx]; //(sampleIdx + radianceSamples.size() / 2) % radianceSamples.size()];
//
//            // What's the value for all of the other lobes?
//            vec3 currentValue = vec3(0.f);
//
//            float sampleLobeWeights[lobeCount];
//            float sampleLobeWeightSum = 0.f;
//            for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
//            {
//                float dotProduct = dot(m_lobes[lobeIt].p, sample.direction);
//                float weight = exp(m_lobes[lobeIt].lambda * (dotProduct - 1.0));
//                currentValue += m_lobes[lobeIt].mu * weight;
//
//                float confidence = lobeWeights[lobeIt] * float(lobeCount) / float(max(sampleIdx, 1u));
////                printf("Sample %u: Confidence for lobe %u is %.3f, weight is %.3f.\n", sampleIdx, lobeIt, confidence, weight);
//
//                /*
//                 High weight, low confidence: bias this lobe towards the new sample
//                 Low weight, high confidence: reduce the weight further on the new sample.
//                 High weight, high confidence: add the new sample (and maybe make sure that the results match)
//                 Low weight, low confidence: add the new sample.
//                 */
//
//                sampleLobeWeights[lobeIt] = weight;
//                sampleLobeWeightSum += weight;
//            }
//
//            // What's the μ that gets us to that delta?
//            vec3 deltaValue = sample.value - currentValue;
//
//            for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
//            {
//                float weight = sampleLobeWeights[lobeIt]; // / sampleLobeWeightSum;
//                if (weight == 0.f) { continue; }
//
//                lobeWeights[lobeIt] += fabs(weight);
//
//                float weightScale = weight / lobeWeights[lobeIt];
//
//
//                // And then compute how much this lobe needs to change to compensate
//                m_lobes[lobeIt].mu += deltaValue * weightScale;
////                m_lobes[lobeIt].mu = max(m_lobes[lobeIt].mu, vec3(0));
//
////                printf("Sample %u: Confidence for lobe %u is %.3f, weight is %.3f.\n", sampleIdx, lobeIt, confidence, weight);
//
//            }
//
//
////            std::ostringstream radianceFilename;
////            radianceFilename << "radiance";
////            if (sampleIdx < 10) radianceFilename << "0";
////            if (sampleIdx < 100) radianceFilename << "0";
////            if (sampleIdx < 1000) radianceFilename << "0";
////            radianceFilename << sampleIdx << ".png";
////
////            auto radianceImage = Image(ivec2(256, 128));
////            radianceImage.forPixels2D([&](vec4& pixel, ivec2 pixelPos)
////                                        {
////                                            vec2 uv = (vec2(pixelPos) + vec2(0.5f)) / vec2(ivec2(256, 128));
////                                            vec3 direction = latLongTexcoordToCartesian(uv);
////
////                                            vec3 sampleSg = sgBasisEvaluate(m_lobes, direction);
////                                            pixel = vec4(sampleSg, 1.0f);
////                                        });
////
////            radianceImage.writePng(radianceFilename.str().c_str());
//        }
//
//
//        printf("\nRunning Average Lobe µs:\n");
//        for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
//        {
//            printf("%.3f, %.3f, %.3f\n", m_lobes[lobeIt].mu.x, m_lobes[lobeIt].mu.y, m_lobes[lobeIt].mu.z);
//        }
//    }
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
