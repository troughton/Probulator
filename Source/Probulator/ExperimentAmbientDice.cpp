#include "ExperimentAmbientDice.h"

#include <Eigen/Eigen>
#include <Eigen/nnls.h>

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
        vec3(0.27639312, -0.44721365, -0.85065085),
        vec3(0.27639312, 0.44721365, 0.85065085),
        vec3(0.27639312, 0.44721365, 0.85065085),
        vec3(0.27639312, -0.44721365, 0.85065085),
        vec3(1.0, -0.0, -0.0),
        vec3(1.0, -0.0, 0.0),
        vec3(1.0, -0.0, 0.0),
        vec3(1.0, 0.0, 0.0),
        vec3(0.8506508, -0.0, -0.52573115),
        vec3(0.8506508, 0.0, 0.52573115),
        vec3(0.8506508, 0.0, 0.52573115),
        vec3(0.8506508, -0.0, -0.52573115)
    };
    
    const vec3 AmbientDice::bitangents[12] = {
        vec3(-0.44721365, 0.72360677, -0.52573115),
        vec3(0.44721365, 0.72360677, -0.52573115),
        vec3(-0.44721365, -0.72360677, 0.52573115),
        vec3(-0.44721365, 0.72360677, 0.52573115),
        vec3(-0.0, 0.525731, -0.85065085),
        vec3(-0.0, 0.525731, 0.85065085),
        vec3(0.0, -0.525731, -0.85065085),
        vec3(-0.0, -0.525731, 0.85065085),
        vec3(-0.0, 1.0, -0.0),
        vec3(0.0, 1.0, -0.0),
        vec3(-0.0, -1.0, 0.0),
        vec3(0.0, -1.0, 0.0)
    };
    
    
    vec3 AmbientDice::hybridCubicBezier(u32 i0, u32 i1, u32 i2, float b0, float b1, float b2) const {
        const float alpha = 0.5 * sqrt(0.5 * (5.0 + sqrt(5.0)));
        const float beta = -0.5 * sqrt(0.1 * (5.0 + sqrt(5.0)));
        
        const float a0 = (sqrt(5.0) - 5.0) / 40.0;
        const float a1 = (11.0 * sqrt(5.0) - 15.0) / 40.0;
        const float a2 = sqrt(5.0) / 10.0;
        
        vec3 c300 = this->vertices[i0].value;
        vec3 c030 = this->vertices[i1].value;
        vec3 c003 = this->vertices[i2].value;
        
        vec3 v0V1 = AmbientDice::vertexPositions[i1] - AmbientDice::vertexPositions[i0];
        vec3 v0V2 = AmbientDice::vertexPositions[i2] - AmbientDice::vertexPositions[i0];
        vec3 v1V2 = AmbientDice::vertexPositions[i2] - AmbientDice::vertexPositions[i1];
        
        const float fValueFactor = -beta / alpha;
        const float fDerivativeFactor = 1.0 / (3.0 * alpha);
        
        vec3 c210 = fValueFactor * this->vertices[i0].value +
        fDerivativeFactor * (dot(v0V1, AmbientDice::tangents[i0]) * this->vertices[i0].directionalDerivativeU + dot(v0V1, AmbientDice::bitangents[i0]) * this->vertices[i0].directionalDerivativeV);
        
        vec3 c201 = fValueFactor * this->vertices[i0].value +
        fDerivativeFactor * (dot(v0V2, AmbientDice::tangents[i0]) * this->vertices[i0].directionalDerivativeU + dot(v0V2, AmbientDice::bitangents[i0]) * this->vertices[i0].directionalDerivativeV);
        
        vec3 c120 = fValueFactor * this->vertices[i1].value +
        fDerivativeFactor * (dot(-v0V1, AmbientDice::tangents[i1]) * this->vertices[i1].directionalDerivativeU + dot(-v0V1, AmbientDice::bitangents[i1]) * this->vertices[i1].directionalDerivativeV);
        
        vec3 c021 = fValueFactor * this->vertices[i1].value +
        fDerivativeFactor * (dot(v1V2, AmbientDice::tangents[i1]) * this->vertices[i1].directionalDerivativeU + dot(v1V2, AmbientDice::bitangents[i1]) * this->vertices[i1].directionalDerivativeV);
        
        vec3 c102 = fValueFactor * this->vertices[i2].value +
        fDerivativeFactor * (dot(-v0V2, AmbientDice::tangents[i2]) * this->vertices[i2].directionalDerivativeU + dot(-v0V2, AmbientDice::bitangents[i2]) * this->vertices[i2].directionalDerivativeV);
        
        vec3 c012 = fValueFactor * this->vertices[i2].value +
        fDerivativeFactor * (dot(-v1V2, AmbientDice::tangents[i2]) * this->vertices[i2].directionalDerivativeU + dot(-v1V2, AmbientDice::bitangents[i2]) * this->vertices[i2].directionalDerivativeV);
        
        
        vec3 c0_111 = a0 * c030 + a1 * c021 + a1 * c012 + a0 * c003 + a2 * c120 + a2 * c102;
        vec3 c1_111 = a0 * c003 + a1 * c102 + a1 * c201 + a0 * c300 + a2 * c012 + a2 * c210;
        vec3 c2_111 = a0 * c300 + a1 * c210 + a1 * c120 + a0 * c030 + a2 * c201 + a2 * c021;
        
        float weightDenom = b1 * b2 + b0 * b2 + b0 * b1;
        
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
        
        vec3 c111 = w0 * c0_111 + w1 * c1_111 + w2 * c2_111;
        
        // https://en.wikipedia.org/wiki/Bézier_triangle
        // Notation: cxyz means alpha^x, beta^y, gamma^z.
        
        float b0_2 = b0 * b0;
        float b1_2 = b1 * b1;
        float b2_2 = b2 * b2;
        
        vec3 interpolated = b1_2 * b1 * c030 + 3.f * c120 * b0 * b1_2 + 3.f * c021 * b1_2 * b2 +
        3.f * c210 * b0_2 * b1 + 6.f * c111 * b0 * b1 * b2 + 3.f * c012 * b1 * b2_2 +
        b0_2 * b0 * c300 + 3.f * c201 * b0_2 * b2 + 3.f * c102 * b0 * b2_2 + c003 * b2_2 * b2;
        
        return interpolated;
    }
    
    void AmbientDice::hybridCubicBezierWeights(u32 i0, u32 i1, u32 i2, float b0, float b1, float b2, VertexWeights *w0Out, VertexWeights *w1Out, VertexWeights *w2Out) const {
        const float alpha = 0.5f * sqrt(0.5f * (5.0f + sqrt(5.0f)));
        const float beta = -0.5f * sqrt(0.1f * (5.0f + sqrt(5.0f)));
        
        const float a0 = (sqrt(5.0f) - 5.0f) / 40.0f;
        const float a1 = (11.0f * sqrt(5.0f) - 15.0f) / 40.0f;
        const float a2 = sqrt(5.0f) / 10.0f;
        
        const vec3 v0V1 = AmbientDice::vertexPositions[i1] - AmbientDice::vertexPositions[i0];
        const vec3 v0V2 = AmbientDice::vertexPositions[i2] - AmbientDice::vertexPositions[i0];
        const vec3 v1V2 = AmbientDice::vertexPositions[i2] - AmbientDice::vertexPositions[i1];
        
        const float fValueFactor = -beta / alpha;
        const float fDerivativeFactor = 1.0 / (3.0 * alpha);
        
        
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
        
        const float b0_2 = b0 * b0;
        const float b1_2 = b1 * b1;
        const float b2_2 = b2 * b2;
        
        float v0ValueWeight = 0.0;
        float v1ValueWeight = 0.0;
        float v2ValueWeight = 0.0;
        
        float v0DUWeight = 0.0;
        float v1DUWeight = 0.0;
        float v2DUWeight = 0.0;
        
        float v0DVWeight = 0.0;
        float v1DVWeight = 0.0;
        float v2DVWeight = 0.0;
        
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
        
        v1ValueWeight += fValueFactor * c120Weight;
        v1DUWeight += fDerivativeFactor * dot(-v0V1, AmbientDice::tangents[i1]) * c120Weight;
        v1DVWeight += fDerivativeFactor * dot(-v0V1, AmbientDice::bitangents[i1]) * c120Weight;
        
        v1ValueWeight += fValueFactor * c021Weight;
        v1DUWeight += fDerivativeFactor * dot(v1V2, AmbientDice::tangents[i1]) * c021Weight;
        v1DVWeight += fDerivativeFactor * dot(v1V2, AmbientDice::bitangents[i1]) * c021Weight;
        
        v0ValueWeight += fValueFactor * c210Weight;
        v0DUWeight += fDerivativeFactor * dot(v0V1, AmbientDice::tangents[i0]) * c210Weight;
        v0DVWeight += fDerivativeFactor * dot(v0V1, AmbientDice::bitangents[i0]) * c210Weight;
        
        v2ValueWeight += fValueFactor * c012Weight;
        v2DUWeight += fDerivativeFactor * dot(-v1V2, AmbientDice::tangents[i2]) * c012Weight;
        v2DVWeight += fDerivativeFactor * dot(-v1V2, AmbientDice::bitangents[i2]) * c012Weight;
        
        v0ValueWeight += fValueFactor * c201Weight;
        v0DUWeight += fDerivativeFactor * dot(v0V2, AmbientDice::tangents[i0]) * c201Weight;
        v0DVWeight += fDerivativeFactor * dot(v0V2, AmbientDice::bitangents[i0]) * c201Weight;
        
        v2ValueWeight += fValueFactor * c102Weight;
        v2DUWeight += fDerivativeFactor * dot(-v0V2, AmbientDice::tangents[i2]) * c102Weight;
        v2DVWeight += fDerivativeFactor * dot(-v0V2, AmbientDice::bitangents[i2]) * c102Weight;
        
        v0ValueWeight += c300Weight;
        v1ValueWeight += c030Weight;
        v2ValueWeight += c003Weight;
        
        printf("Weights are %.3f, %.3f, %.3f\n", v0ValueWeight, v1ValueWeight, v2ValueWeight);
        
        *w0Out = { v0ValueWeight, v0DUWeight, v0DVWeight };
        *w1Out = { v1ValueWeight, v1DUWeight, v1DVWeight };
        *w2Out = { v2ValueWeight, v2DUWeight, v2DVWeight };
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
            
            
            u32 i0, i1, i2;
            ambientDice.indexIcosahedron(direction, &i0, &i1, &i2);
            
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
        AmbientDice::VertexWeights vertexWeights[12] = { { 0 } };
        
        const u64 sampleCount = directions.getPixelCount();
        
        std::vector<u64> sampleIndices;
        sampleIndices.resize(sampleCount);
        std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
        
        std::random_shuffle(sampleIndices.begin(), sampleIndices.end());
        
        for (u64 sampleIt : sampleIndices)
        {
            const vec3& direction = directions.at(sampleIt);
            
            u32 i0, i1, i2;
            ambientDice.indexIcosahedron(direction, &i0, &i1, &i2);
            
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
            
            u32 indices[3] = {i0, i1, i2};
            AmbientDice::VertexWeights weights[3];
            ambientDice.hybridCubicBezierWeights(i0, i1, i2, b0, b1, b2, &weights[0], &weights[1], &weights[2]);
            
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
            
            printf("Current estimate: %.3f, %.3f, %.3f\n", currentEstimate.r, currentEstimate.g, currentEstimate.b);
            
            const vec3 delta = vec3(targetValue.x, targetValue.y, targetValue.z) - currentEstimate;
            
            for (u64 i = 0; i < 3; i += 1) {
                u32 index = indices[i];
                const AmbientDice::VertexWeights &weight = weights[i];
                
                vertexWeights[index].value += weight.value;
                vertexWeights[index].directionalDerivativeU += weight.directionalDerivativeU;
                vertexWeights[index].directionalDerivativeV += weight.directionalDerivativeV;
                
                if (weight.value != 0.f) {
                    float weightScale = weight.value / vertexWeights[index].value;
                    ambientDice.vertices[index].value += delta * weightScale;
                }
                
//                if (weight.directionalDerivativeU != 0.f) {
//                    float weightScale = weight.directionalDerivativeU / vertexWeights[index].directionalDerivativeU;
//                    ambientDice.vertices[index].directionalDerivativeU += delta * weightScale;
//                }
//                
//                if (weight.directionalDerivativeV != 0.f) {
//                    float weightScale = weight.directionalDerivativeV / vertexWeights[index].directionalDerivativeV;
//                    ambientDice.vertices[index].directionalDerivativeV += delta * weightScale;
//                }
                
                if (false /* nonNegative */) {
                    ambientDice.vertices[index].value = max(ambientDice.vertices[index].value, vec3(0.f));
                    ambientDice.vertices[index].directionalDerivativeU = max(ambientDice.vertices[index].directionalDerivativeU, vec3(0.f));
                    ambientDice.vertices[index].directionalDerivativeV = max(ambientDice.vertices[index].directionalDerivativeV, vec3(0.f));
                }
            }
        }
        
        return ambientDice;
    }
    
    void ExperimentAmbientDice::run(SharedData& data)
    {
        AmbientDice ambientDice = solveAmbientDiceRunningAverageBezier(data.m_directionImage, m_input->m_irradianceImage);

        m_radianceImage = Image(data.m_outputSize);
        m_irradianceImage = Image(data.m_outputSize);

        data.m_directionImage.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
                                          {
                                              vec3 sampleIrradianceH = ambientDice.evaluateBezier(direction);
                                             
                                        
                                              
//                                              u32 triangleIndex = ambientDice.indexIcosahedronTriangle(direction);
//
//                                              vec3 colour;
//                                              if (triangleIndex < 8) {
//                                                  colour = vec3(float(triangleIndex) * 0.5f / 7.f + 0.5f);
//                                              } else if (triangleIndex < 12) {
//                                                  colour = vec3(float(triangleIndex - 8) / 4.f * 0.5f + 0.5f, 0.f, 0.f);
//                                              } else if (triangleIndex < 16) {
//                                                  colour = vec3(0.f, float(triangleIndex - 12) / 4.f * 0.5f + 0.5f, 0.f);
//                                              } else{
//                                                  colour = vec3(0.f, 0.f, float(triangleIndex - 16) / 4.f * 0.5f + 0.5f);
//                                              }
                                              m_irradianceImage.at(pixelPos) = vec4(sampleIrradianceH, 1.0f);
                                              m_radianceImage.at(pixelPos) = vec4(0.0f);
                                          });
    }
}
