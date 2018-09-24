#pragma once

#include <Probulator/Experiments.h>

namespace Probulator {

class ExperimentAmbientCube : public Experiment
{
public:

	struct AmbientCube
	{
		vec3 irradiance[6];

		vec3 evaluate(const vec3& direction) const
		{
			vec3 dirSquared = direction * direction;

			vec3 result
				= dirSquared.x * (direction.x < 0 ? irradiance[0] : irradiance[1])
				+ dirSquared.y * (direction.y < 0 ? irradiance[2] : irradiance[3])
				+ dirSquared.z * (direction.z < 0 ? irradiance[4] : irradiance[5]);

			return result;
		}
	};
    
    enum class SolveType : int {
        LeastSquares,
        RunningAverage,
        Projection
    };

	static AmbientCube solveAmbientCubeLeastSquares(const ImageBase<vec3>& directions, const Image& irradiance);
    static AmbientCube solveAmbientCubeRunningAverage(const ImageBase<vec3>& directions, const Image& irradiance);
	static AmbientCube solveAmbientCubeProjection(const Image& irradiance);

	void run(SharedData& data) override;

	void getProperties(std::vector<Property>& outProperties) override
	{
		Experiment::getProperties(outProperties);
		outProperties.push_back(Property("Solve Type", reinterpret_cast<int*>(&m_solveType)));
	}

	ExperimentAmbientCube& setSolveType(SolveType solveType)
    {
        m_solveType = solveType;
        return *this;
    }

    SolveType m_solveType = SolveType::LeastSquares;
};

}
