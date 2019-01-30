#include "SGFitLeastSquares.h"
#include <Eigen/Eigen>
#include <Eigen/nnls.h>

namespace Probulator
{
	SgBasis sgFitLeastSquaresOld(const SgBasis& basis, const std::vector<RadianceSample>& samples)
	{
		using namespace Eigen;
		SgBasis result = basis;

		MatrixXf A;
		A.resize(samples.size(), basis.size());
		for (u64 sampleIt = 0; sampleIt < samples.size(); ++sampleIt)
		{
			for (u64 lobeIt = 0; lobeIt < basis.size(); ++lobeIt)
			{
				A(sampleIt, lobeIt) = sgEvaluate(basis[lobeIt].p, basis[lobeIt].lambda, samples[sampleIt].direction);
			}
		}

		for (u32 channelIt = 0; channelIt < 3; ++channelIt)
		{
			VectorXf b;
			b.resize(samples.size());
			for (u64 sampleIt = 0; sampleIt < samples.size(); ++sampleIt)
			{
				b[sampleIt] = samples[sampleIt].value[channelIt];
			}

			VectorXf x = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
			for (u64 lobeIt = 0; lobeIt < basis.size(); ++lobeIt)
			{
				result[lobeIt].mu[channelIt] = x[lobeIt];
			}
		}

		return result;
	}
    
    SgBasis sgFitLeastSquares(const SgBasis& basis, const std::vector<RadianceSample>& samples)
    {
        
        printf("Lobe directions:\n");
        for (const SphericalGaussian& lobe : basis) {
            printf("float3(%f, %f, %f),\n", lobe.p.x, lobe.p.y, lobe.p.z);
        }
        printf("\nLobe lambda: %f\n", basis[0].lambda);
        
        // Using the raw moments and inverse Gram matrix: Peter-Pike Sloan
        
        using namespace Eigen;
        SgBasis result = basis;
        
        const float sampleScale = (hemisphericalIntegral ? 2 * M_PI : 4 * M_PI) / float(samples.size());
        
        MatrixXf gram;
        gram.resize(basis.size(), basis.size());
        for (u64 lobeAIt = 0; lobeAIt < basis.size(); ++lobeAIt)
        {
            for (u64 lobeBIt = lobeAIt; lobeBIt < basis.size(); ++lobeBIt)
            {
                float integral = 0.f;
                
                for (const RadianceSample &sample : samples) {
                    float lobeAWeight = exp(basis[lobeAIt].lambda * (dot(sample.direction, basis[lobeAIt].p) - 1.0));
                    float lobeBWeight = exp(basis[lobeBIt].lambda * (dot(sample.direction, basis[lobeBIt].p) - 1.0));
                    integral += lobeAWeight * lobeBWeight * sampleScale; // * 4 * .pi
                }
                
                gram(lobeAIt, lobeBIt) = integral;
                gram(lobeBIt, lobeAIt) = integral;
            }
        }
        
        printf("Row diagonally dominant checks:\n");
        for (u64 row = 0; row < basis.size(); ++row)
        {
            float total = fabs(gram(row, row));
            
            for (u64 col = 0; col < basis.size(); ++col)
            {
                if (col == row) { continue; }
                
                total -= fabs(gram(row, col));
            }
            printf("%llu: %f\n", row, total);
        }
        
        printf("Gram Matrix:\n");
        
        for (u64 lobeAIt = 0; lobeAIt < basis.size(); ++lobeAIt)
        {
            printf("[ ");
            for (u64 lobeBIt = 0; lobeBIt < basis.size(); ++lobeBIt)
            {
                printf("%f, ", gram(lobeAIt, lobeBIt));
            }
            printf("],\n");
        }
        
        printf("Inverse Gram Matrix:\n");
        
        MatrixXf gramInverse = gram.inverse();
        
        for (u64 lobeAIt = 0; lobeAIt < basis.size(); ++lobeAIt)
        {
            printf("[ ");
            for (u64 lobeBIt = 0; lobeBIt < basis.size(); ++lobeBIt)
            {
                printf("%f, ", gramInverse(lobeAIt, lobeBIt));
            }
            printf("],\n");
        }
        
        MatrixXf diagonalInv = MatrixXf::Identity(gram.rows(), gram.cols());
        for (u64 lobeIt = 0; lobeIt < basis.size(); ++lobeIt)
        {
            diagonalInv.coeffRef(lobeIt, lobeIt) = 1.0 / gram(lobeIt, lobeIt);
        }
        
        MatrixXf cJabobi = MatrixXf::Identity(gram.rows(), gram.cols()) - diagonalInv * gram;
        
        printf("CJacobi Eigenvalues:\n");
        auto eigenValues = cJabobi.eigenvalues();
        for (u64 i = 0; i < eigenValues.rows(); i += 1) {
            printf("%f %f\n", eigenValues[i].real(), eigenValues[i].imag());
        }
        
        MatrixXf identityRight = gramInverse * gram;
        
        for (u32 channelIt = 0; channelIt < 3; ++channelIt)
        {
            VectorXf rawMoments;
            rawMoments.resize(basis.size());
            
            for (u64 lobeIt = 0; lobeIt < basis.size(); lobeIt += 1) {
                float lobeTotal = 0.f;
                for (u64 sampleIt = 0; sampleIt < samples.size(); ++sampleIt) {
                    float sample = samples[sampleIt].value[channelIt];
                    float weight = exp(basis[lobeIt].lambda * (dot(samples[sampleIt].direction, basis[lobeIt].p) - 1.0));
                    lobeTotal += sample * weight;
                }
                rawMoments[lobeIt] = lobeTotal * sampleScale;
            }
            
            VectorXf x = gramInverse * rawMoments;
            
            for (u64 lobeIt = 0; lobeIt < basis.size(); ++lobeIt)
            {
                result[lobeIt].mu[channelIt] = x[lobeIt];
            }
        }
        
        return result;
    }
    
    SgBasis sgFitLeastSquaresMoments2(const SgBasis& basis, const std::vector<RadianceSample>& samples)
    {
        // Using the raw moments and inverse Gram matrix, running average edition: Peter-Pike Sloan
        
        using namespace Eigen;
        SgBasis result = basis;
        
        MatrixXf gram;
        gram.resize(basis.size(), basis.size());
        for (u64 lobeAIt = 0; lobeAIt < basis.size(); ++lobeAIt)
        {
            for (u64 lobeBIt = lobeAIt; lobeBIt < basis.size(); ++lobeBIt)
            {
                float integral = 0.f;
                
                for (const RadianceSample &sample : samples) {
                    float lobeAWeight = exp(basis[lobeAIt].lambda * (dot(sample.direction, basis[lobeAIt].p) - 1.0));
                    float lobeBWeight = exp(basis[lobeBIt].lambda * (dot(sample.direction, basis[lobeBIt].p) - 1.0));
                    integral += lobeAWeight * lobeBWeight; // * 4 * .pi
                }
                
                integral *= 4 * M_PI / float(samples.size());
                
                gram(lobeAIt, lobeBIt) = integral;
                gram(lobeBIt, lobeAIt) = integral;
            
            }
        }
        
        MatrixXf gramInverse = gram.inverse();
        
        for (u64 lobeAIt = 0; lobeAIt < basis.size(); ++lobeAIt)
        {
            printf("( ");
            for (u64 lobeBIt = 0; lobeBIt < basis.size(); ++lobeBIt)
            {
                printf("%.3f ", gramInverse(lobeAIt, lobeBIt));
            }
            printf(")\n");
        }
        
        for (u32 channelIt = 0; channelIt < 3; ++channelIt)
        {
            VectorXf rawMoments;
            rawMoments.resize(basis.size());
            
            for (u64 lobeIt = 0; lobeIt < basis.size(); lobeIt += 1) {
                float lobeMean = 0.f;
                float lobeWeight = 0.f;
                
                for (u64 sampleIt = 0; sampleIt < samples.size(); ++sampleIt) {
                    float sample = samples[sampleIt].value[channelIt];
                    float weight = exp(basis[lobeIt].lambda * (dot(samples[sampleIt].direction, basis[lobeIt].p) - 1.0));
                    float value = sample * weight;
                    
                    lobeWeight += 1.f;
                    float delta = value - lobeMean;
                    lobeMean += delta * (1.f / lobeWeight);
                }
                rawMoments[lobeIt] = lobeMean * 4 * M_PI;
            }
            
            VectorXf x = gramInverse * rawMoments;
            
            for (u64 lobeIt = 0; lobeIt < basis.size(); ++lobeIt)
            {
                result[lobeIt].mu[channelIt] = x[lobeIt];
            }
        }
        
        return result;
    }

	// Non-negative version of least squares
	SgBasis sgFitNNLeastSquares(const SgBasis& basis, const std::vector<RadianceSample>& samples)
	{
		using namespace Eigen;
		SgBasis result = basis;

		MatrixXf A;
		A.resize(samples.size(), basis.size());
		for (u64 sampleIt = 0; sampleIt < samples.size(); ++sampleIt)
		{
			for (u64 lobeIt = 0; lobeIt < basis.size(); ++lobeIt)
			{
				A(sampleIt, lobeIt) = sgEvaluate(basis[lobeIt].p, basis[lobeIt].lambda, samples[sampleIt].direction);
			}
		}

		NNLS<MatrixXf> nnlssolver(A);
		for (u32 channelIt = 0; channelIt < 3; ++channelIt)
		{
			VectorXf b;
			b.resize(samples.size());
			for (u64 sampleIt = 0; sampleIt < samples.size(); ++sampleIt)
			{
				b[sampleIt] = samples[sampleIt].value[channelIt];
			}

			// -- run the solver
			nnlssolver.solve(b);
			VectorXf x = nnlssolver.x();

			for (u64 lobeIt = 0; lobeIt < basis.size(); ++lobeIt)
			{
				result[lobeIt].mu[channelIt] = x[lobeIt];
			}
		}

		return result;
	}
}
