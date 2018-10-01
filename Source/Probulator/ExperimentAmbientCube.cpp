#include "ExperimentAmbientCube.h"

#include <Eigen/Eigen>
#include <Eigen/nnls.h>

namespace Probulator {

ExperimentAmbientCube::AmbientCube ExperimentAmbientCube::solveAmbientCubeLeastSquares(const ImageBase<vec3>& directions, const Image& irradiance)
{
	using namespace Eigen;

	AmbientCube ambientCube;

	const u64 sampleCount = directions.getPixelCount();

	MatrixXf A;
	A.resize(sampleCount, 6);

	for (u64 sampleIt = 0; sampleIt < sampleCount; ++sampleIt)
	{
		const vec3& direction = directions.at(sampleIt);
		vec3 dirSquared = direction * direction;

		if (direction.x < 0)
		{
			A(sampleIt, 0) = dirSquared.x;
			A(sampleIt, 1) = 0.0f;
		}
		else
		{
			A(sampleIt, 0) = 0.0f;
			A(sampleIt, 1) = dirSquared.x;
		}

		if (direction.y < 0)
		{
			A(sampleIt, 2) = dirSquared.y;
			A(sampleIt, 3) = 0.0f;
		}
		else
		{
			A(sampleIt, 2) = 0.0f;
			A(sampleIt, 3) = dirSquared.y;
		}

		if (direction.z < 0)
		{
			A(sampleIt, 4) = dirSquared.z;
			A(sampleIt, 5) = 0.0f;
		}
		else
		{
			A(sampleIt, 4) = 0.0f;
			A(sampleIt, 5) = dirSquared.z;
		}
	}

	NNLS<MatrixXf> solver(A);

	VectorXf b;
	b.resize(sampleCount);

	for (u32 channelIt = 0; channelIt < 3; ++channelIt)
	{
		for (u64 sampleIt = 0; sampleIt < sampleCount; ++sampleIt)
		{
			b[sampleIt] = irradiance.at(sampleIt)[channelIt];
		}

		solver.solve(b);
		VectorXf x = solver.x();

		for (u64 basisIt = 0; basisIt < 6; ++basisIt)
		{
			ambientCube.irradiance[basisIt][channelIt] = x[basisIt];
		}
	}

	return ambientCube;
}
    
    inline float RandomFloat(float a, float b) {
        float random = ((float) rand()) / (float) RAND_MAX;
        float diff = b - a;
        float r = random * diff;
        return a + r;
    }
    
ExperimentAmbientCube::AmbientCube ExperimentAmbientCube::solveAmbientCubeRunningAverage(const ImageBase<vec3>& directions, const Image& irradiance)
{
    AmbientCube ambientCube;
    float cubeWeights[6] = { 0 };
    
    const u64 sampleCount = directions.getPixelCount();
    
    std::vector<u64> sampleIndices;
    sampleIndices.resize(sampleCount);
    std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
    
    std::random_shuffle(sampleIndices.begin(), sampleIndices.end());
    
    for (u64 sampleIt : sampleIndices)
    {
        const vec3& direction = directions.at(sampleIt);
        
        // What's the current value in the sample's direction?
        vec3 currentValue = ambientCube.evaluate(direction);
        vec4 targetValue = irradiance.at(sampleIt);
        
        vec3 delta = vec3(targetValue.x, targetValue.y, targetValue.z) - currentValue;
        
        vec3 dirSquared = direction * direction;
        
        for (u64 i = 0; i < 3; i += 1) {
            u64 index = direction[i] < 0 ? (2 * i) : (2 * i + 1);
            
            float weight = dirSquared[i];
            if (weight == 0.f) {
                continue;
            }
            
            cubeWeights[index] += weight;
           
            float weightScale = weight / cubeWeights[index];
            ambientCube.irradiance[index] += delta * weightScale;
            
            if (true /* nonNegative */) {
                ambientCube.irradiance[index] = max(ambientCube.irradiance[index], vec3(0.f));
            }
        }
    }
    
    return ambientCube;
}

ExperimentAmbientCube::AmbientCube ExperimentAmbientCube::solveAmbientCubeProjection(const Image& irradiance)
{
	AmbientCube ambientCube;

	vec3 cubeDirections[6] =
	{
		vec3(-1.0f, 0.0f, 0.0f),
		vec3(+1.0f, 0.0f, 0.0f),
		vec3(0.0f, -1.0f, 0.0f),
		vec3(0.0f, +1.0f, 0.0f),
		vec3(0.0f, 0.0f, -1.0f),
		vec3(0.0f, 0.0f, +1.0f),
	};

	for(u32 i=0; i<6; ++i)
	{
		vec2 texcoord = cartesianToLatLongTexcoord(cubeDirections[i]);
		ambientCube.irradiance[i] = (vec3)irradiance.sampleNearest(texcoord);
	}

	return ambientCube;
}

void ExperimentAmbientCube::run(SharedData& data)
{
	AmbientCube ambientCube;

    switch (m_solveType) {
        case SolveType::Projection:
            ambientCube = solveAmbientCubeProjection(m_input->m_irradianceImage);
            break;
        case SolveType::LeastSquares:
            ambientCube = solveAmbientCubeLeastSquares(data.m_directionImage, m_input->m_irradianceImage);
            break;
        case SolveType::RunningAverage:
            ambientCube = solveAmbientCubeRunningAverage(data.m_directionImage, m_input->m_irradianceImage);
            break;
    }

	m_radianceImage = Image(data.m_outputSize);
	m_irradianceImage = Image(data.m_outputSize);

	data.m_directionImage.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
	{
		vec3 sampleIrradianceH = ambientCube.evaluate(direction);
		m_irradianceImage.at(pixelPos) = vec4(sampleIrradianceH, 1.0f);
		m_radianceImage.at(pixelPos) = vec4(0.0f);
	});
}

}
