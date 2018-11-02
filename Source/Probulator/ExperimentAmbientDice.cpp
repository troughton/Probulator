#include "ExperimentAmbientDice.h"

#include <Eigen/Eigen>
#include <Eigen/nnls.h>

#include "ExperimentAmbientD20.h"

namespace Probulator {
    
    const float AmbientDice::kT = 0.618034f;
    const float AmbientDice::kT2 = kT * kT;
    
    const vec3 AmbientDice::vertexPositions[12] = {
        vec3(1.0, kT, 0.0),
        vec3(-1.0, kT, 0.0),
        vec3(1.0, -kT, -0.0),
        vec3(-1.0, -kT, 0.0),
        vec3(0.0, 1.0, kT),
        vec3(-0.0, -1.0, kT),
        vec3(0.0, 1.0, -kT),
        vec3(0.0, -1.0, -kT),
        vec3(kT, 0.0, 1.0),
        vec3(-kT, 0.0, 1.0),
        vec3(kT, -0.0, -1.0),
        vec3(-kT, -0.0, -1.0)
    };
    
    const vec3 AmbientDice::tangents[12] = {
//        vec3(0.27639312, -0.44721365, -0.85065085),
//        vec3(0.27639312, 0.44721365, 0.85065085),
//        vec3(0.27639312, 0.44721365, 0.85065085),
//        vec3(0.27639312, -0.44721365, 0.85065085),
//        vec3(1.0, -0.0, -0.0),
//        vec3(1.0, -0.0, 0.0),
//        vec3(1.0, -0.0, 0.0),
//        vec3(1.0, 0.0, 0.0),
//        vec3(0.8506508, -0.0, -0.52573115),
//        vec3(0.8506508, 0.0, 0.52573115),
//        vec3(0.8506508, 0.0, 0.52573115),
//        vec3(0.8506508, -0.0, -0.52573115)

        vec3(-0.52573115, 0.85065085, 0.0),
        vec3(-0.52573115, -0.85065085, 0.0),
        vec3(0.52573115, 0.85065085, 0.0),
        vec3(0.52573115, -0.85065085, 0.0),
        vec3(-0.99999994, 0.0, 0.0),
        vec3(0.99999994, -0.0, 0.0),
        vec3(-0.99999994, 0.0, 0.0),
        vec3(0.99999994, 0.0, 0.0),
        vec3(-0.0, 1.0, 0.0),
        vec3(-0.0, -1.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        vec3(0.0, -1.0, 0.0),
    };
    
    const vec3 AmbientDice::bitangents[12] = {
//        vec3(-0.44721365, 0.72360677, -0.52573115),
//        vec3(0.44721365, 0.72360677, -0.52573115),
//        vec3(-0.44721365, -0.72360677, 0.52573115),
//        vec3(-0.44721365, 0.72360677, 0.52573115),
//        vec3(-0.0, 0.525731, -0.85065085),
//        vec3(-0.0, 0.525731, 0.85065085),
//        vec3(0.0, -0.525731, -0.85065085),
//        vec3(-0.0, -0.525731, 0.85065085),
//        vec3(-0.0, 1.0, -0.0),
//        vec3(0.0, 1.0, -0.0),
//        vec3(-0.0, -1.0, 0.0),
//        vec3(0.0, -1.0, 0.0)

        vec3(0.0, -0.0, 1.0),
        vec3(0.0, 0.0, 1.0),
        vec3(0.0, -0.0, 1.0),
        vec3(0.0, 0.0, 1.0),
        vec3(0.0, -0.52573115, 0.85065085),
        vec3(0.0, 0.52573115, 0.85065085),
        vec3(0.0, 0.52573115, 0.85065085),
        vec3(0.0, -0.52573115, 0.85065085),
        vec3(-0.8506507, -0.0, 0.5257311),
        vec3(0.8506507, 0.0, 0.5257311),
        vec3(0.8506507, -0.0, 0.5257311),
        vec3(-0.8506507, 0.0, 0.5257311),
    };
    
    const u32 AmbientDice::triangleIndices[20][3] = {
        { 0, 4, 8 },
        { 1, 4, 9 },
        { 2, 5, 8 },
        { 3, 5, 9 },
        { 0, 6, 10 },
        { 1, 6, 11 },
        { 2, 7, 10 },
        { 3, 7, 11 },
        { 4, 8, 9 },
        { 5, 8, 9 },
        { 6, 10, 11 },
        { 7, 10, 11 },
        { 0, 2, 8 },
        { 1, 3, 9 },
        { 0, 2, 10 },
        { 1, 3, 11 },
        { 0, 4, 6 },
        { 1, 4, 6 },
        { 2, 5, 7 },
        { 3, 5, 7 },
    };
    const vec3 AmbientDice::triangleBarycentricNormals[20][3] = {
        { vec3(0.9510565, 0.36327127, -0.58778524), vec3(-0.58778524, 0.9510565, 0.36327127), vec3(0.36327127, -0.58778524, 0.9510565) },
        { vec3(-0.9510565, 0.36327127, -0.58778524), vec3(0.58778524, 0.9510565, 0.36327127), vec3(-0.36327127, -0.58778524, 0.9510565) },
        { vec3(0.9510565, -0.36327127, -0.58778524), vec3(-0.58778524, -0.9510565, 0.36327127), vec3(0.36327127, 0.58778524, 0.9510565) },
        { vec3(-0.9510565, -0.36327127, -0.58778524), vec3(0.58778524, -0.9510565, 0.36327127), vec3(-0.36327127, 0.58778524, 0.9510565) },
        { vec3(0.9510565, 0.36327127, 0.58778524), vec3(-0.58778524, 0.9510565, -0.36327127), vec3(0.36327127, -0.58778524, -0.9510565) },
        { vec3(-0.9510565, 0.36327127, 0.58778524), vec3(0.58778524, 0.9510565, -0.36327127), vec3(-0.36327127, -0.58778524, -0.9510565) },
        { vec3(0.9510565, -0.36327127, 0.58778524), vec3(-0.58778524, -0.9510565, -0.36327127), vec3(0.36327127, 0.58778524, -0.9510565) },
        { vec3(-0.9510565, -0.36327127, 0.58778524), vec3(0.58778524, -0.9510565, -0.36327127), vec3(-0.36327127, 0.58778524, -0.9510565) },
        { vec3(-0.0, 1.1755705, -0.0), vec3(0.9510565, -0.36327127, 0.58778524), vec3(-0.9510565, -0.36327127, 0.58778524) },
        { vec3(0.0, -1.1755705, 0.0), vec3(0.9510565, 0.36327127, 0.58778524), vec3(-0.9510565, 0.36327127, 0.58778524) },
        { vec3(0.0, 1.1755705, -0.0), vec3(0.9510565, -0.36327127, -0.58778524), vec3(-0.9510565, -0.36327127, -0.58778524) },
        { vec3(-0.0, -1.1755705, 0.0), vec3(0.9510565, 0.36327127, -0.58778524), vec3(-0.9510565, 0.36327127, -0.58778524) },
        { vec3(0.58778524, 0.9510565, -0.36327127), vec3(0.58778524, -0.9510565, -0.36327127), vec3(-0.0, -0.0, 1.1755705) },
        { vec3(-0.58778524, 0.9510565, -0.36327127), vec3(-0.58778524, -0.9510565, -0.36327127), vec3(0.0, 0.0, 1.1755705) },
        { vec3(0.58778524, 0.9510565, 0.36327127), vec3(0.58778524, -0.9510565, 0.36327127), vec3(0.0, 0.0, -1.1755705) },
        { vec3(-0.58778524, 0.9510565, 0.36327127), vec3(-0.58778524, -0.9510565, 0.36327127), vec3(-0.0, -0.0, -1.1755705) },
        { vec3(1.1755705, -0.0, -0.0), vec3(-0.36327127, 0.58778524, 0.9510565), vec3(-0.36327127, 0.58778524, -0.9510565) },
        { vec3(-1.1755705, 0.0, 0.0), vec3(0.36327127, 0.58778524, 0.9510565), vec3(0.36327127, 0.58778524, -0.9510565) },
        { vec3(1.1755705, 0.0, 0.0), vec3(-0.36327127, -0.58778524, 0.9510565), vec3(-0.36327127, -0.58778524, -0.9510565) },
        { vec3(-1.1755705, -0.0, -0.0), vec3(0.36327127, -0.58778524, 0.9510565), vec3(0.36327127, -0.58778524, -0.9510565) },
    };
    
    inline void constructOrthonormalBasis(vec3 n, vec3 *b1, vec3 *b2) {
        float sign = copysign(1.0f, n.z);
        const float a = -1.0f / (sign + n.z);
        const float b = n.x * n.y * a;
        *b1 = vec3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
        *b2 = vec3(b, sign + n.y * n.y * a, -n.y);
    }
   
    
    vec3 AmbientDice::hybridCubicBezier(u32 i0, u32 i1, u32 i2, float b0, float b1, float b2) const {
        
        VertexWeights w0, w1, w2;
        this->hybridCubicBezierWeights(i0, i1, i2, b0, b1, b2, &w0, &w1, &w2);
        
        return w0.value * this->vertices[i0].value +
               w0.directionalDerivativeU * this->vertices[i0].directionalDerivativeU +
               w0.directionalDerivativeV * this->vertices[i0].directionalDerivativeV +
               w1.value * this->vertices[i1].value +
               w1.directionalDerivativeU * this->vertices[i1].directionalDerivativeU +
               w1.directionalDerivativeV * this->vertices[i1].directionalDerivativeV +
               w2.value * this->vertices[i2].value +
               w2.directionalDerivativeU * this->vertices[i2].directionalDerivativeU +
               w2.directionalDerivativeV * this->vertices[i2].directionalDerivativeV;
    }
    
    void AmbientDice::hybridCubicBezierWeights(u32 i0, u32 i1, u32 i2, float b0, float b1, float b2, VertexWeights *w0Out, VertexWeights *w1Out, VertexWeights *w2Out) const {
        const float alpha = 0.5f * sqrt(0.5f * (5.0f + sqrt(5.0f))); // 0.9510565163
        const float beta = -0.5f * sqrt(0.1f * (5.0f + sqrt(5.0f))); // -0.4253254042
        
        const float a0 = (sqrt(5.0f) - 5.0f) / 40.0f; // -0.06909830056
        const float a1 = (11.0f * sqrt(5.0f) - 15.0f) / 40.0f; // 0.2399186938
        const float a2 = sqrt(5.0f) / 10.0f; // 0.2236067977
        
        // Project the edges onto the sphere.
        vec3 v0V1 = AmbientDice::vertexPositions[i1] - AmbientDice::vertexPositions[i0];
        v0V1 = v0V1 - normalize(AmbientDice::vertexPositions[i0]) * dot(normalize(AmbientDice::vertexPositions[i0]), v0V1);
        v0V1 = normalize(v0V1);
        
        vec3 v1V0 = AmbientDice::vertexPositions[i0] - AmbientDice::vertexPositions[i1];
        v1V0 = v1V0 - normalize(AmbientDice::vertexPositions[i1]) * dot(normalize(AmbientDice::vertexPositions[i1]), v1V0);
        v1V0 = normalize(v1V0);
        
        vec3 v0V2 = AmbientDice::vertexPositions[i2] - AmbientDice::vertexPositions[i0];
        v0V2 = v0V2 - normalize(AmbientDice::vertexPositions[i0]) * dot(normalize(AmbientDice::vertexPositions[i0]), v0V2);
        v0V2 = normalize(v0V2);
        
        vec3 v2V0 = AmbientDice::vertexPositions[i0] - AmbientDice::vertexPositions[i2];
        v2V0 = v2V0 - normalize(AmbientDice::vertexPositions[i2]) * dot(normalize(AmbientDice::vertexPositions[i2]), v2V0);
        v2V0 = normalize(v2V0);
        
        vec3 v1V2 = AmbientDice::vertexPositions[i2] - AmbientDice::vertexPositions[i1];
        v1V2 = v1V2 - normalize(AmbientDice::vertexPositions[i1]) * dot(normalize(AmbientDice::vertexPositions[i1]), v1V2);
        v1V2 = normalize(v1V2);
        
        vec3 v2V1 = AmbientDice::vertexPositions[i1] - AmbientDice::vertexPositions[i2];
        v2V1 = v2V1 - normalize(AmbientDice::vertexPositions[i2]) * dot(normalize(AmbientDice::vertexPositions[i2]), v2V1);
        v2V1 = normalize(v2V1);
        
        
        vec3 v0 = AmbientDice::vertexPositions[i0];
        vec3 v1 = AmbientDice::vertexPositions[i1];
        vec3 v2 = AmbientDice::vertexPositions[i2];
        
        vec3 edge0N = normalize( cross( v0, v1 ) );
        vec3 edge1N = normalize( cross( v1, v2 ) );
        vec3 edge2N = normalize( cross( v2, v0 ) );
        
        v0V1 = normalize(  cross( edge0N, v0 ) );
        v0V2 = normalize( -cross( edge2N, v0 ) );
        v1V0 = normalize( -cross( edge0N, v1 ) );
        v1V2 = normalize(  cross( edge1N, v1 ) );
        v2V0 = normalize(  cross( edge2N, v2 ) );
        v2V1 = normalize( -cross( edge1N, v2 ) );
        
        const float fValueFactor = -beta / alpha; // 0.4472135955
        const float fDerivativeFactor = 1.0 / (3.0 * alpha); // 0.3504874081
        
        const float weightDenom = b1 * b2 + b0 * b2 + b0 * b1;
        
        float w0 = (b1 * b2) / weightDenom;
        float w1 = (b0 * b2) / weightDenom;
        float w2 = (b0 * b1) / weightDenom;
        
        if (b0 == 1.0) {
            w0 = 1.0;
            w1 = 0.0;
            w2 = 0.0;
        } else if (b1 == 1.0) {
            w0 = 0.0;
            w1 = 1.0;
            w2 = 0.0;
        } else if (b2 == 1.0) {
            w0 = 0.0;
            w1 = 0.0;
            w2 = 1.0;
        }
        
        // https://en.wikipedia.org/wiki/Bézier_triangle
        // Notation: cxyz means alpha^x, beta^y, gamma^z.
        
        float v0ValueWeight = 0.0;
        float v1ValueWeight = 0.0;
        float v2ValueWeight = 0.0;
        
        float v0DUWeight = 0.0;
        float v1DUWeight = 0.0;
        float v2DUWeight = 0.0;
        
        float v0DVWeight = 0.0;
        float v1DVWeight = 0.0;
        float v2DVWeight = 0.0;
        
        const float b0_2 = b0 * b0;
        const float b1_2 = b1 * b1;
        const float b2_2 = b2 * b2;
        
        // Add c300, c030, and c003
        float c300Weight = b0_2 * b0;
        float c030Weight = b1_2 * b1;
        float c003Weight = b2_2 * b2;
        
        float c120Weight = 3 * b0 * b1_2;
        float c021Weight = 3 * b1_2 * b2;
        float c210Weight = 3 * b0_2 * b1;
        float c012Weight = 3 * b1 * b2_2;
        float c201Weight = 3 * b0_2 * b2;
        float c102Weight = 3 * b0 * b2_2;
        
        const float c111Weight = 6 * b0 * b1 * b2;
        const float c0_111Weight = w0 * c111Weight;
        const float c1_111Weight = w1 * c111Weight;
        const float c2_111Weight = w2 * c111Weight;
        
        v1ValueWeight += a0 * c0_111Weight;
        v2ValueWeight += a0 * c1_111Weight;
        v0ValueWeight += a0 * c2_111Weight;
        
        c021Weight += a1 * c0_111Weight;
        c012Weight += a1 * c0_111Weight;
        c003Weight += a0 * c0_111Weight;
        c120Weight += a2 * c0_111Weight;
        c102Weight += a2 * c0_111Weight;
        
        c102Weight += a1 * c1_111Weight;
        c201Weight += a1 * c1_111Weight;
        c300Weight += a0 * c1_111Weight;
        c012Weight += a2 * c1_111Weight;
        c210Weight += a2 * c1_111Weight;
        
        c210Weight += a1 * c2_111Weight;
        c120Weight += a1 * c2_111Weight;
        c030Weight += a0 * c2_111Weight;
        c201Weight += a2 * c2_111Weight;
        c021Weight += a2 * c2_111Weight;
        
        
        v0ValueWeight += fValueFactor * c210Weight;
        v0DUWeight += fDerivativeFactor * dot(v0V1, AmbientDice::tangents[i0]) * c210Weight;
        v0DVWeight += fDerivativeFactor * dot(v0V1, AmbientDice::bitangents[i0]) * c210Weight;
        
        v0ValueWeight += fValueFactor * c201Weight;
        v0DUWeight += fDerivativeFactor * dot(v0V2, AmbientDice::tangents[i0]) * c201Weight;
        v0DVWeight += fDerivativeFactor * dot(v0V2, AmbientDice::bitangents[i0]) * c201Weight;
        
        v1ValueWeight += fValueFactor * c120Weight;
        v1DUWeight += fDerivativeFactor * dot(v1V0, AmbientDice::tangents[i1]) * c120Weight;
        v1DVWeight += fDerivativeFactor * dot(v1V0, AmbientDice::bitangents[i1]) * c120Weight;
        
        v1ValueWeight += fValueFactor * c021Weight;
        v1DUWeight += fDerivativeFactor * dot(v1V2, AmbientDice::tangents[i1]) * c021Weight;
        v1DVWeight += fDerivativeFactor * dot(v1V2, AmbientDice::bitangents[i1]) * c021Weight;
        
        v2ValueWeight += fValueFactor * c102Weight;
        v2DUWeight += fDerivativeFactor * dot(v2V0, AmbientDice::tangents[i2]) * c102Weight;
        v2DVWeight += fDerivativeFactor * dot(v2V0, AmbientDice::bitangents[i2]) * c102Weight;
        
        v2ValueWeight += fValueFactor * c012Weight;
        v2DUWeight += fDerivativeFactor * dot(v2V1, AmbientDice::tangents[i2]) * c012Weight;
        v2DVWeight += fDerivativeFactor * dot(v2V1, AmbientDice::bitangents[i2]) * c012Weight;
        
        v0ValueWeight += c300Weight;
        v1ValueWeight += c030Weight;
        v2ValueWeight += c003Weight;
        
//        v0ValueWeight = max(0.f, v0ValueWeight);
//        v1ValueWeight = max(0.f, v1ValueWeight);
//        v2ValueWeight = max(0.f, v2ValueWeight);
//
//        v0DUWeight = max(0.f, v0DUWeight);
//        v0DVWeight = max(0.f, v0DVWeight);
//
//        v1DUWeight = max(0.f, v1DUWeight);
//        v1DVWeight = max(0.f, v1DVWeight);
//
//        v2DUWeight = max(0.f, v2DUWeight);
//        v2DVWeight = max(0.f, v2DVWeight);
        
        assert(v0ValueWeight >= 0.f);
        assert(v1ValueWeight >= 0.f);
        assert(v2ValueWeight >= 0.f);
        
//        assert(v0DUWeight >= 0.f);
//        assert(v1DUWeight >= 0.f);
//        assert(v2DUWeight >= 0.f);
//
//        assert(v0DVWeight >= 0.f);
//        assert(v1DVWeight >= 0.f);
//        assert(v2DVWeight >= 0.f);
        
        
        *w0Out = { v0ValueWeight, v0DUWeight, v0DVWeight };
        *w1Out = { v1ValueWeight, v1DUWeight, v1DVWeight };
        *w2Out = { v2ValueWeight, v2DUWeight, v2DVWeight };
    }
    
    void AmbientDice::hybridCubicBezierWeights(vec3 direction, u32 *i0Out, u32 *i1Out, u32 *i2Out, VertexWeights *w0Out, VertexWeights *w1Out, VertexWeights *w2Out) const {
        
        u32 i0, i1, i2;
        float b0, b1, b2;
        this->computeBarycentrics(direction, &i0, &i1, &i2, &b0, &b1, &b2);
        
        this->hybridCubicBezierWeights(i0, i1, i2, b0, b1, b2, w0Out, w1Out, w2Out);
        
        *i0Out = i0;
        *i1Out = i1;
        *i2Out = i2;
    }
    
    void AmbientDice::srbfWeights(vec3 direction, float *weightsOut) const {
        for (u64 i = 0; i < 12; i += 1) {
            float dotProduct = max(dot(direction, normalize(AmbientDice::vertexPositions[i])), 0.f);
            float cos2 = dotProduct * dotProduct;
            float cos4 = cos2 * cos2;
            
            weightsOut[i] = 0.7f * (0.5f * cos2) + 0.3f * (5.f / 6.f * cos4);
        }
    }
    
    AmbientDice ExperimentAmbientDice::solveAmbientDiceRunningAverage(const ImageBase<vec3>& directions, const Image& irradiance)
    {
        AmbientDice ambientDice;
        float vertexWeights[12] = { 0 };
        
        const u64 sampleCount = directions.getPixelCount();
        
        std::vector<u64> sampleIndices;
        sampleIndices.resize(sampleCount);
        std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
        
        std::random_shuffle(sampleIndices.begin(), sampleIndices.end());
        
        for (u64 sampleIt : sampleIndices)
        {
            const vec3& direction = directions.at(sampleIt);
            
            // What's the current value in the sample's direction?
            vec3 currentValue = ambientDice.evaluateLinear(direction);
            vec4 targetValue = irradiance.at(sampleIt);
            
            vec3 delta = vec3(targetValue.x, targetValue.y, targetValue.z) - currentValue;
            
            u32 triIndex = ambientDice.indexIcosahedronTriangle(direction);
            u32 i0 = AmbientDice::triangleIndices[triIndex][0];
            u32 i1 = AmbientDice::triangleIndices[triIndex][1];
            u32 i2 = AmbientDice::triangleIndices[triIndex][2];
            
            const vec3& v0 = AmbientDice::vertexPositions[i0];
            const vec3& v1 = AmbientDice::vertexPositions[i1];
            const vec3& v2 = AmbientDice::vertexPositions[i2];
            
            vec3 n0 = normalize(cross(v1, v2));
            vec3 n1 = normalize(cross(v0, v2));
            vec3 n2 = normalize(cross(v0, v1));
            
            float b0 = dot(direction, n0) / dot(v0, n0);
            float b1 = dot(direction, n1) / dot(v1, n1);
            float b2 = dot(direction, n2) / dot(v2, n2);
            
            //            return hybridCubicBezier(i0, i1, i2, b0, b1, b2);
            
            if (b0 != 0.f) {
                u32 index = i0;
                float weight = b0;
                
                vertexWeights[index] += weight;
                float weightScale = weight / vertexWeights[index];
                ambientDice.vertices[index].value += delta * weightScale;
                
                if (false /* nonNegative */) {
                    ambientDice.vertices[index].value = max(ambientDice.vertices[index].value, vec3(0.f));
                }
            }
            
            if (b1 != 0.f) {
                u32 index = i1;
                float weight = b1;
                if (weight == 0.f) { continue; }
                
                vertexWeights[index] += weight;
                float weightScale = weight / vertexWeights[index];
                ambientDice.vertices[index].value += delta * weightScale;
                
                if (false /* nonNegative */) {
                    ambientDice.vertices[index].value = max(ambientDice.vertices[index].value, vec3(0.f));
                }
            }
            
            if (b2 != 0.f) {
                u32 index = i2;
                float weight = b2;
                if (weight == 0.f) { continue; }
                
                vertexWeights[index] += weight;
                float weightScale = weight / vertexWeights[index];
                ambientDice.vertices[index].value += delta * weightScale;
                
                if (false /* nonNegative */) {
                    ambientDice.vertices[index].value = max(ambientDice.vertices[index].value, vec3(0.f));
                }
            }
        }
        
        for (u32 v = 0; v < 12; v += 1) {
            if (v < 4) {
                ambientDice.vertices[v].value = vec3(float(v) * 0.5f / 4.f + 0.5f, 0.f, 0.f);
            } else if (v < 8) {
                ambientDice.vertices[v].value = vec3(0.f, float(v - 4) * 0.5f / 4.f + 0.5f, 0.f);
            } else {
                ambientDice.vertices[v].value = vec3(0.f, 0.f, float(v - 8) * 0.5f / 4.f + 0.5f);
            }
        }
        
        return ambientDice;
    }
    
    AmbientDice ExperimentAmbientDice::solveAmbientDiceRunningAverageBezier(const ImageBase<vec3>& directions, const Image& irradiance)
    {
        AmbientDice ambientDice;
        AmbientDice::VertexWeights vertexWeights[12] = { { 0.f, 0.f, 0.f } };
        
        const u64 sampleCount = directions.getPixelCount();
        
        std::vector<u64> sampleIndices;
        sampleIndices.resize(sampleCount);
        std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
        
        std::random_shuffle(sampleIndices.begin(), sampleIndices.end());
        
        float sampleIndex = 0.f;
        for (u64 sampleIt : sampleIndices)
        {
            sampleIndex += 1;
            
            const vec3& direction = directions.at(sampleIt);
            
            u32 i0, i1, i2;
            AmbientDice::VertexWeights weights[3];
            ambientDice.hybridCubicBezierWeights(direction, &i0, &i1, &i2, &weights[0], &weights[1], &weights[2]);
            
            const u32 indices[3] = { i0, i1, i2 };
            
            // What's the current value in the sample's direction?
            vec4 targetValue = irradiance.at(sampleIt);
            vec3 currentEstimate =
            weights[0].value * ambientDice.vertices[i0].value +
            weights[0].directionalDerivativeU * ambientDice.vertices[i0].directionalDerivativeU +
            weights[0].directionalDerivativeV * ambientDice.vertices[i0].directionalDerivativeV +
            weights[1].value * ambientDice.vertices[i1].value +
            weights[1].directionalDerivativeU * ambientDice.vertices[i1].directionalDerivativeU +
            weights[1].directionalDerivativeV * ambientDice.vertices[i1].directionalDerivativeV +
            weights[2].value * ambientDice.vertices[i2].value +
            weights[2].directionalDerivativeU * ambientDice.vertices[i2].directionalDerivativeU +
            weights[2].directionalDerivativeV * ambientDice.vertices[i2].directionalDerivativeV;
            
            const vec3 delta = vec3(targetValue.x, targetValue.y, targetValue.z) - currentEstimate;
            
            const float sampleWeightScale = 1.f / sampleIndex;
            
            for (u64 i = 0; i < 3; i += 1) {
                u32 index = indices[i];
                const AmbientDice::VertexWeights &weight = weights[i];
                
                vertexWeights[index].value += (weight.value * weight.value - vertexWeights[index].value) * sampleWeightScale;
                vertexWeights[index].directionalDerivativeU += (weight.directionalDerivativeU * weight.directionalDerivativeU - vertexWeights[index].directionalDerivativeU) * sampleWeightScale;
                vertexWeights[index].directionalDerivativeV += (weight.directionalDerivativeV * weight.directionalDerivativeV - vertexWeights[index].directionalDerivativeV) * sampleWeightScale;
                
                {
                    float sphericalIntegral = sampleWeightScale + (1 - sampleWeightScale) * vertexWeights[index].value;
                    
                    float deltaScale = 1.0f * weights[i].value * sampleWeightScale / sphericalIntegral;
                    ambientDice.vertices[index].value += delta * deltaScale;
                }
                
                {
                    float sphericalIntegral = sampleWeightScale + (1 - sampleWeightScale) * vertexWeights[index].directionalDerivativeU;
                    
                    float deltaScale = 1.0f * weights[i].directionalDerivativeU * sampleWeightScale / sphericalIntegral;
                    ambientDice.vertices[index].directionalDerivativeU += delta * deltaScale;
                }
                
                {
                    float sphericalIntegral = sampleWeightScale + (1 - sampleWeightScale) * vertexWeights[index].directionalDerivativeV;
                    
                    float deltaScale = 1.0f * weights[i].directionalDerivativeV * sampleWeightScale / sphericalIntegral;
                    ambientDice.vertices[index].directionalDerivativeV += delta * deltaScale;
                }
                
                if (false /* nonNegative */) {
                    ambientDice.vertices[index].value = max(ambientDice.vertices[index].value, vec3(0.f));
                    ambientDice.vertices[index].directionalDerivativeU = max(ambientDice.vertices[index].directionalDerivativeU, vec3(0.f));
                    ambientDice.vertices[index].directionalDerivativeV = max(ambientDice.vertices[index].directionalDerivativeV, vec3(0.f));
                }
            }
        }
        
        return ambientDice;
    }
    
    AmbientDice ExperimentAmbientDice::solveAmbientDiceRunningAverageSRBF(const ImageBase<vec3>& directions, const Image& irradiance)
    {
        AmbientDice ambientDice;
        AmbientDice::VertexWeights vertexWeights[12] = { { 0.f, 0.f, 0.f } };
        
        const u64 sampleCount = directions.getPixelCount();
        
        std::vector<u64> sampleIndices;
        sampleIndices.resize(sampleCount);
        std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
        
        std::random_shuffle(sampleIndices.begin(), sampleIndices.end());
        
        float sampleIndex = 0.f;
        for (u64 sampleIt : sampleIndices)
        {
            sampleIndex += 1;
            
            const vec3& direction = directions.at(sampleIt);
            
            float weights[12];
            ambientDice.srbfWeights(direction, weights);
            
            // What's the current value in the sample's direction?
            vec4 targetValue = irradiance.at(sampleIt);
            vec3 currentEstimate = ambientDice.evaluateSRBF(direction);
            
            const vec3 delta = vec3(targetValue.x, targetValue.y, targetValue.z) - currentEstimate;
            
            const float sampleWeightScale = 1.f / sampleIndex;
            
            for (u64 i = 0; i < 12; i += 1) {
                float weight = weights[i];
                
                vertexWeights[i].value += (weight * weight - vertexWeights[i].value) * sampleWeightScale;
                
                float sphericalIntegral = sampleWeightScale + (1 - sampleWeightScale) * vertexWeights[i].value;
                
                float deltaScale = 3.0f * weight * sampleWeightScale / sphericalIntegral;
                ambientDice.vertices[i].value += delta * deltaScale;
                
                if (false /* nonNegative */) {
                    ambientDice.vertices[i].value = max(ambientDice.vertices[i].value, vec3(0.f));
                }
            }
        }
        
        return ambientDice;
    }
    
//    AmbientDice ExperimentAmbientDice::solveAmbientDiceLeastSquares(const ImageBase<vec3>& directions, const Image& irradiance)
//    {
//        using namespace Eigen;
//
//        AmbientDice ambientDice;
//
//        const u64 sampleCount = directions.getPixelCount();
//
//        MatrixXf moments = MatrixXf::Zero(36, 3);
//
//        for (u64 sampleIt = 0; sampleIt < sampleCount; ++sampleIt)
//        {
//            const vec3& direction = directions.at(sampleIt);
//            const vec4& colour = irradiance.at(sampleIt);
//
//            u32 i0, i1, i2;
//            AmbientDice::VertexWeights weights[3];
//            ambientDice.hybridCubicBezierWeights(direction, &i0, &i1, &i2, &weights[0], &weights[1], &weights[2]);
//
//            moments(3 * i0 + 0, 0) += weights[0].value * colour.r / float(sampleCount);
//            moments(3 * i0 + 1, 0) += weights[0].directionalDerivativeU * colour.r / float(sampleCount);
//            moments(3 * i0 + 2, 0) += weights[0].directionalDerivativeV * colour.r / float(sampleCount);
//
//            moments(3 * i1 + 0, 0) += weights[1].value * colour.r / float(sampleCount);
//            moments(3 * i1 + 1, 0) += weights[1].directionalDerivativeU * colour.r / float(sampleCount);
//            moments(3 * i1 + 2, 0) += weights[1].directionalDerivativeV * colour.r / float(sampleCount);
//
//            moments(3 * i2 + 0, 0) += weights[2].value * colour.r / float(sampleCount);
//            moments(3 * i2 + 1, 0) += weights[2].directionalDerivativeU * colour.r / float(sampleCount);
//            moments(3 * i2 + 2, 0) += weights[2].directionalDerivativeV * colour.r / float(sampleCount);
//
//            moments(3 * i0 + 0, 1) += weights[0].value * colour.g / float(sampleCount);
//            moments(3 * i0 + 1, 1) += weights[0].directionalDerivativeU * colour.g / float(sampleCount);
//            moments(3 * i0 + 2, 1) += weights[0].directionalDerivativeV * colour.g / float(sampleCount);
//
//            moments(3 * i1 + 0, 1) += weights[1].value * colour.g / float(sampleCount);
//            moments(3 * i1 + 1, 1) += weights[1].directionalDerivativeU * colour.g / float(sampleCount);
//            moments(3 * i1 + 2, 1) += weights[1].directionalDerivativeV * colour.g / float(sampleCount);
//
//            moments(3 * i2 + 0, 1) += weights[2].value * colour.g / float(sampleCount);
//            moments(3 * i2 + 1, 1) += weights[2].directionalDerivativeU * colour.g / float(sampleCount);
//            moments(3 * i2 + 2, 1) += weights[2].directionalDerivativeV * colour.g / float(sampleCount);
//
//            moments(3 * i0 + 0, 2) += weights[0].value * colour.b / float(sampleCount);
//            moments(3 * i0 + 1, 2) += weights[0].directionalDerivativeU * colour.b / float(sampleCount);
//            moments(3 * i0 + 2, 2) += weights[0].directionalDerivativeV * colour.b / float(sampleCount);
//
//            moments(3 * i1 + 0, 2) += weights[1].value * colour.b / float(sampleCount);
//            moments(3 * i1 + 1, 2) += weights[1].directionalDerivativeU * colour.b / float(sampleCount);
//            moments(3 * i1 + 2, 2) += weights[1].directionalDerivativeV * colour.b / float(sampleCount);
//
//            moments(3 * i2 + 0, 2) += weights[2].value * colour.b / float(sampleCount);
//            moments(3 * i2 + 1, 2) += weights[2].directionalDerivativeU * colour.b / float(sampleCount);
//            moments(3 * i2 + 2, 2) += weights[2].directionalDerivativeV * colour.b / float(sampleCount);
//        }
//
//        MatrixXf gram;
//        gram.resize(36, 36);
//        for (u64 lobeAIt = 0; lobeAIt < 36; ++lobeAIt)
//        {
//            for (u64 lobeBIt = lobeAIt; lobeBIt < 36; ++lobeBIt)
//            {
//                float integral = 0.f;
//
//                for (u64 sampleIt = 0; sampleIt < sampleCount; ++sampleIt)
//                {
//                    float allWeights[36] = { 0.f };
//
//                    const vec3& direction = directions.at(sampleIt);
//
//                    u32 i0, i1, i2;
//                    AmbientDice::VertexWeights weights[3];
//                    ambientDice.hybridCubicBezierWeights(direction, &i0, &i1, &i2, &weights[0], &weights[1], &weights[2]);
//
//                    allWeights[3 * i0 + 0] = weights[0].value;
//                    allWeights[3 * i0 + 1] = weights[0].directionalDerivativeU;
//                    allWeights[3 * i0 + 2] = weights[0].directionalDerivativeV;
//
//                    allWeights[3 * i1 + 0] = weights[1].value;
//                    allWeights[3 * i1 + 1] = weights[1].directionalDerivativeU;
//                    allWeights[3 * i1 + 2] = weights[1].directionalDerivativeV;
//
//                    allWeights[3 * i2 + 0] = weights[2].value;
//                    allWeights[3 * i2 + 1] = weights[2].directionalDerivativeU;
//                    allWeights[3 * i2 + 2] = weights[2].directionalDerivativeV;
//
//                    integral += allWeights[lobeAIt] * allWeights[lobeBIt];
//                }
//
//                integral *= /* 4 * M_PI */ 1.f / float(sampleCount);
//
//                gram(lobeAIt, lobeBIt) = integral;
//                gram(lobeBIt, lobeAIt) = integral;
//            }
//        }
//
//        NNLS<MatrixXf> solver(gram);
//
//        VectorXf b;
//        b.resize(36);
//
//        for (u32 channelIt = 0; channelIt < 3; ++channelIt)
//        {
//            for (u64 lobeIt = 0; lobeIt < 36; ++lobeIt)
//            {
//                b[lobeIt] = moments(lobeIt, channelIt);
//            }
//
//            solver.solve(b);
//            VectorXf x = solver.x();
//
//            for (u64 basisIt = 0; basisIt < 12; ++basisIt)
//            {
//                ambientDice.vertices[basisIt].value[channelIt] = x[3 * basisIt];
//                ambientDice.vertices[basisIt].directionalDerivativeU[channelIt] = x[3 * basisIt + 1];
//                ambientDice.vertices[basisIt].directionalDerivativeV[channelIt] = x[3 * basisIt + 2];
//            }
//        }
//
//        return ambientDice;
//    }
    
    AmbientDice ExperimentAmbientDice::solveAmbientDiceLeastSquares(const ImageBase<vec3>& directions, const Image& irradiance)
    {
        using namespace Eigen;

        AmbientDice ambientDice;

        const u64 sampleCount = directions.getPixelCount();

        MatrixXf A = MatrixXf::Zero(sampleCount, 36);

        for (u64 sampleIt = 0; sampleIt < sampleCount; ++sampleIt)
        {
            const vec3& direction = directions.at(sampleIt);

            u32 i0, i1, i2;
            AmbientDice::VertexWeights weights[3];
            ambientDice.hybridCubicBezierWeights(direction, &i0, &i1, &i2, &weights[0], &weights[1], &weights[2]);

            A(sampleIt, 3 * i0) = weights[0].value;
            A(sampleIt, 3 * i0 + 1) = weights[0].directionalDerivativeU;
            A(sampleIt, 3 * i0 + 2) = weights[0].directionalDerivativeV;

            A(sampleIt, 3 * i1) = weights[1].value;
            A(sampleIt, 3 * i1 + 1) = weights[1].directionalDerivativeU;
            A(sampleIt, 3 * i1 + 2) = weights[1].directionalDerivativeV;

            A(sampleIt, 3 * i2) = weights[2].value;
            A(sampleIt, 3 * i2 + 1) = weights[2].directionalDerivativeU;
            A(sampleIt, 3 * i2 + 2) = weights[2].directionalDerivativeV;

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

            for (u64 basisIt = 0; basisIt < 12; ++basisIt)
            {
                ambientDice.vertices[basisIt].value[channelIt] = x[3 * basisIt];
                ambientDice.vertices[basisIt].directionalDerivativeU[channelIt] = x[3 * basisIt + 1];
                ambientDice.vertices[basisIt].directionalDerivativeV[channelIt] = x[3 * basisIt + 2];
            }
        }

        return ambientDice;
    }
    
    AmbientDice ExperimentAmbientDice::solveAmbientDiceLeastSquaresSRBF(const ImageBase<vec3>& directions, const Image& irradiance)
    {
        using namespace Eigen;

        AmbientDice ambientDice;

        const u64 sampleCount = directions.getPixelCount();

        MatrixXf A = MatrixXf::Zero(sampleCount, 12);

        for (u64 sampleIt = 0; sampleIt < sampleCount; ++sampleIt)
        {
            const vec3& direction = directions.at(sampleIt);

            float weights[12];
            ambientDice.srbfWeights(direction, weights);

            for (u64 i = 0; i < 12; i += 1) {
                A(sampleIt, i) = weights[i];
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

            for (u64 basisIt = 0; basisIt < 12; ++basisIt)
            {
                ambientDice.vertices[basisIt].value[channelIt] = x[basisIt];
            }
        }

        return ambientDice;
    }
    
    inline void setBasisFunction( AmbientDiceIS &ambientDice, unsigned int index  )
    {
        // generates a response function, index is used since we have 48 of these in general, 36 for bezier patches, 12 for linear functions
        ambientDice.clear();
        
        if ( index < 36 )
        {
            const unsigned int vertCheck = index / 3;
            ambientDice.valuesRGB[vertCheck].r = 1.0f;
            ambientDice.gradientsRGB[vertCheck][0].g = 1.0f;
            ambientDice.gradientsRGB[vertCheck][1].b = 1.0f;
        }
        else if ( index < 48 )
        {
            // these are the linear functions - drive them from CoCg.r
            const unsigned int vertCheck = index - 36;
            ambientDice.valuesCoCg[vertCheck].x = 1.0f;
        }
        else if ( index < 60 )
        {
            const unsigned int vertCheck = index - 48;
            
            ambientDice.valuesQuadRGB[vertCheck].r = 1.0f;
            ambientDice.valuesQuadY[vertCheck] = 1.0f;
        }
    }
    
    inline void compareResponseImages(ExperimentAmbientDice::SharedData& data) {
        std::vector<ImageBase<float>> m_responseImages;
        m_responseImages.resize( 36 + 12 );
        
        AmbientDiceIS ambientDice( AmbientDiceIS::IM_HybridCubicBezierSpherical, 1, AmbientDiceIS::CM_RGB, 0.f, 0.f );
        
        AmbientDice dice;
        
        for ( unsigned int vert = 0; vert < 12; vert++ )
        {
            m_responseImages[vert * 3 + 0] = ImageBase<float>( data.m_outputSize );
            m_responseImages[vert * 3 + 1] = ImageBase<float>( data.m_outputSize );
            m_responseImages[vert * 3 + 2] = ImageBase<float>( data.m_outputSize );
            // below is for linear functions...
            m_responseImages[36 + vert] = ImageBase<float>( data.m_outputSize );
            m_responseImages[36 + vert].fill( 0.0f );
            
            // this now does 3 basis functions at a time always...
            setBasisFunction( ambientDice, vert * 3 );
            
            for (u32 i = 0; i < 12; i += 1) {
                dice.vertices[i] = { {  } };
            }
            dice.vertices[vert] = { vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1) };
            
            data.m_directionImage.forPixels2D( [&]( const vec3& direction, ivec2 pixelPos )
                                              {
                                                  vec3 sampleBF = ambientDice.evaluate( direction );
                                                  
                                                  vec3 otherSample = dice.evaluateBezier(direction);
                                                  
                                                  assert(fabs(sampleBF.r - otherSample.r) < 1e-4f);
                                                  assert(fabs(sampleBF.g - otherSample.g) < 1e-4f);
                                                  assert(fabs(sampleBF.b - otherSample.b) < 1e-4f);
                                                  
//                                                  u32 i0, i1, i2;
//                                                  float b0, b1, b2;
//                                                  dice.computeBarycentrics(direction, &i0, &i1, &i2, &b0, &b1, &b2);
//                                                  u32 indices[3] = { i0, i1, i2 };
//
//                                                  float baryCoords[3];
//                                                  ambientDice.getWeightsSphericalInterpolation( direction, ambientDice.getIcosaDir(indices[0]),  ambientDice.getIcosaDir(indices[1]), ambientDice.getIcosaDir(indices[2]), baryCoords );
//
//                                                  assert(baryCoords[0] == b0);
//                                                  assert(baryCoords[1] == b1);
//                                                  assert(baryCoords[2] == b2);
                                                  
                                                  m_responseImages[vert * 3 + 0].at( pixelPos ) = sampleBF.r;// getVec3NonZero( sampleBF );
                                                  m_responseImages[vert * 3 + 1].at( pixelPos ) = sampleBF.g;
                                                  m_responseImages[vert * 3 + 2].at( pixelPos ) = sampleBF.b;
                                                  
                                              } );
        }
        
        std::vector<ImageBase<float>> m_responseImages2;
        m_responseImages2.resize( 36 );
        
        for ( unsigned int vert = 0; vert < 12; vert++ )
        {
            m_responseImages2[vert * 3 + 0] = ImageBase<float>( data.m_outputSize );
            m_responseImages2[vert * 3 + 1] = ImageBase<float>( data.m_outputSize );
            m_responseImages2[vert * 3 + 2] = ImageBase<float>( data.m_outputSize );
        }
        data.m_directionImage.forPixels2D( [&]( const vec3& direction, ivec2 pixelPos )
                                          {
                                              
                                              u32 i0, i1, i2;
                                              AmbientDice::VertexWeights w0, w1, w2;
                                              dice.hybridCubicBezierWeights(direction, &i0, &i1, &i2, &w0, &w1, &w2);
                                              
                                              
                                              m_responseImages2[i0 * 3 + 0].at( pixelPos ) = w0.value;
                                              m_responseImages2[i0 * 3 + 1].at( pixelPos ) = w0.directionalDerivativeU;
                                              m_responseImages2[i0 * 3 + 2].at( pixelPos ) = w0.directionalDerivativeV;
                                              
                                              m_responseImages2[i1 * 3 + 0].at( pixelPos ) = w1.value;
                                              m_responseImages2[i1 * 3 + 1].at( pixelPos ) = w1.directionalDerivativeU;
                                              m_responseImages2[i1 * 3 + 2].at( pixelPos ) = w1.directionalDerivativeV;
                                              
                                              m_responseImages2[i2 * 3 + 0].at( pixelPos ) = w2.value;
                                              m_responseImages2[i2 * 3 + 1].at( pixelPos ) = w2.directionalDerivativeU;
                                              m_responseImages2[i2 * 3 + 2].at( pixelPos ) = w2.directionalDerivativeV;
                                              
                                          } );
        
        data.m_directionImage.forPixels2D( [&]( const vec3& direction, ivec2 pixelPos ) {
            for (u64 i = 0; i < 36; i += 1) {
                float delta = m_responseImages2[i].at(pixelPos) - m_responseImages[i].at(pixelPos);
                if (fabs(delta) > 1e-6f) {
                    printf("Direction (%.2f, %.2f, %.2f):\n", direction.x, direction.y, direction.z);
                    printf("%llu: %f - %f, ", i, m_responseImages[i].at(pixelPos), m_responseImages2[i].at(pixelPos));
                    
                }
            }
        });
    }
    
    void ExperimentAmbientDice::run(SharedData& data)
    {
        compareResponseImages(data);
        AmbientDice ambientDiceRadiance = solveAmbientDiceLeastSquaresSRBF(data.m_directionImage, m_input->m_radianceImage);
        AmbientDice ambientDice = solveAmbientDiceLeastSquaresSRBF(data.m_directionImage, m_input->m_irradianceImage);

        m_radianceImage = Image(data.m_outputSize);
        m_irradianceImage = Image(data.m_outputSize);

        data.m_directionImage.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
                                          {
                                              vec3 sampleRadiance = ambientDiceRadiance.evaluateSRBF(direction);
                                              m_radianceImage.at(pixelPos) = vec4(sampleRadiance, 1.0f);
                                              
                                              vec3 sampleIrradiance = ambientDice.evaluateSRBF(direction);
                                              
                                              m_irradianceImage.at(pixelPos) = vec4(sampleIrradiance, 1.0f);
                                          });
    }
}
