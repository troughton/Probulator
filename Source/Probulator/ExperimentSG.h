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
    
class ExperimentSGCustom : public ExperimentSGBase
{
public:
    
    void solveForRadiance(const std::vector<RadianceSample>& radianceSamples) override
    {
        const u32 lobeCount = (u32)m_lobes.size();

        float lobeWeights[lobeCount];
        
        for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
        {
            lobeWeights[lobeIt] = 0.f;
        }

        for (const RadianceSample& sample : radianceSamples)
        {
            // What's the value for all of the other lobes?
            vec3 currentValue = vec3(0.f);
            
            float sampleLobeWeights[lobeCount];
            float sampleLobeWeightSum = 0.f;
            for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
            {
                float dotProduct = dot(m_lobes[lobeIt].p, sample.direction);
                float weight = exp(m_lobes[lobeIt].lambda * (dotProduct - 1.0));
                currentValue += m_lobes[lobeIt].mu * weight;
                
                sampleLobeWeights[lobeIt] = weight;
                sampleLobeWeightSum += weight;
            }
            
            // What's the μ that gets us to that delta?
            vec3 deltaValue = sample.value - currentValue;

            for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
            {
                float weight = sampleLobeWeights[lobeIt] / sampleLobeWeightSum;
                if (weight == 0.f) { continue; }

                lobeWeights[lobeIt] += weight;

                float weightScale = weight / lobeWeights[lobeIt];

                // And then compute how much this lobe needs to change to compensate
                m_lobes[lobeIt].mu += deltaValue * weightScale;
            }
        }
    }
};

class ExperimentSGLS : public ExperimentSGBase
{
public:

    void solveForRadiance(const std::vector<RadianceSample>& radianceSamples) override
    {
        m_lobes = sgFitLeastSquares(m_lobes, radianceSamples);
    }
};

class ExperimentSGNNLS : public ExperimentSGBase
{
public:

    void solveForRadiance(const std::vector<RadianceSample>& radianceSamples) override
    {
        m_lobes = sgFitNNLeastSquares(m_lobes, radianceSamples);
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
