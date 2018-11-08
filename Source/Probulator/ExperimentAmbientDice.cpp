#include <iostream>

#include "ExperimentAmbientDice.h"

#include <Eigen/Eigen>
#include <Eigen/nnls.h>

#include "ExperimentAmbientD20.h"

#include <chrono>

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
    
    const vec3 AmbientDice::srbfNormalisedVertexPositions[6] = {
        normalize(vec3(1.0, kT, 0.0)),
        normalize(vec3(-1.0, kT, 0.0)),
        normalize(vec3(0.0, 1.0, kT)),
        normalize(vec3(-0.0, -1.0, kT)),
        normalize(vec3(kT, 0.0, 1.0)),
        normalize(vec3(kT, -0.0, -1.0))
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

//        vec3(-0.52573115, 0.85065085, 0.0),
//        vec3(-0.52573115, -0.85065085, 0.0),
//        vec3(0.52573115, 0.85065085, 0.0),
//        vec3(0.52573115, -0.85065085, 0.0),
//        vec3(-0.99999994, 0.0, 0.0),
//        vec3(0.99999994, -0.0, 0.0),
//        vec3(-0.99999994, 0.0, 0.0),
//        vec3(0.99999994, 0.0, 0.0),
//        vec3(-0.0, 1.0, 0.0),
//        vec3(-0.0, -1.0, 0.0),
//        vec3(0.0, 1.0, 0.0),
//        vec3(0.0, -1.0, 0.0),
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

//        vec3(0.0, -0.0, 1.0),
//        vec3(0.0, 0.0, 1.0),
//        vec3(0.0, -0.0, 1.0),
//        vec3(0.0, 0.0, 1.0),
//        vec3(0.0, -0.52573115, 0.85065085),
//        vec3(0.0, 0.52573115, 0.85065085),
//        vec3(0.0, 0.52573115, 0.85065085),
//        vec3(0.0, -0.52573115, 0.85065085),
//        vec3(-0.8506507, -0.0, 0.5257311),
//        vec3(0.8506507, 0.0, 0.5257311),
//        vec3(0.8506507, -0.0, 0.5257311),
//        vec3(-0.8506507, 0.0, 0.5257311),
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
    
    const float AmbientDice::triDerivativeTangentFactors[20][6] = {
        { -0.34100485f, -0.238272f, 0.3504874f, 0.21661313f, 0.2981424f, -0.11388027f },
        { 0.34100485f, 0.238272f, -0.3504874f, -0.21661313f, -0.2981424f, 0.11388027f },
        { 0.027519437f, 0.35801283f, 0.3504874f, 0.21661313f, 0.2981424f, -0.11388027f },
        { 0.34100485f, 0.238272f, -0.3504874f, -0.21661313f, -0.2981424f, 0.11388027f },
        { 0.027519437f, 0.35801283f, 0.3504874f, 0.21661313f, 0.2981424f, -0.11388027f },
        { -0.027519437f, -0.35801283f, -0.3504874f, -0.21661313f, -0.2981424f, 0.11388027f },
        { -0.34100485f, -0.238272f, 0.3504874f, 0.21661313f, 0.2981424f, -0.11388027f },
        { -0.027519437f, -0.35801283f, -0.3504874f, -0.21661313f, -0.2981424f, 0.11388027f },
        { 0.21661313f, -0.21661313f, -0.11388027f, -0.36852428f, 0.11388027f, 0.36852428f },
        { 0.21661313f, -0.21661313f, -0.11388027f, -0.36852428f, 0.11388027f, 0.36852428f },
        { 0.21661313f, -0.21661313f, -0.11388027f, -0.36852428f, 0.11388027f, 0.36852428f },
        { 0.21661313f, -0.21661313f, -0.11388027f, -0.36852428f, 0.11388027f, 0.36852428f },
        { 0.1937447f, -0.238272f, 0.1937447f, 0.35801283f, 0.2981424f, 0.2981424f },
        { -0.1937447f, 0.238272f, -0.1937447f, 0.238272f, -0.2981424f, -0.2981424f },
        { 0.1937447f, 0.35801283f, 0.1937447f, -0.238272f, 0.2981424f, 0.2981424f },
        { -0.1937447f, -0.35801283f, -0.1937447f, -0.35801283f, -0.2981424f, -0.2981424f },
        { -0.34100485f, 0.027519437f, 0.3504874f, 0.0f, 0.3504874f, 0.0f },
        { 0.34100485f, -0.027519437f, -0.3504874f, 0.0f, -0.3504874f, 0.0f },
        { 0.027519437f, -0.34100485f, 0.3504874f, 0.0f, 0.3504874f, 0.0f },
        { 0.34100485f, -0.027519437f, -0.3504874f, 0.0f, -0.3504874f, 0.0f }
    };
    const float AmbientDice::triDerivativeBitangentFactors[20][6] = {
        { 0.1397348f, -0.2811345f, 0.11388027f, -0.2981424f, 0.21661313f, 0.3504874f },
        { 0.1397348f, -0.2811345f, 0.11388027f, -0.2981424f, 0.21661313f, 0.3504874f },
        { 0.36749536f, 0.08738982f, -0.11388027f, 0.2981424f, -0.21661313f, -0.3504874f },
        { -0.1397348f, 0.2811345f, -0.11388027f, 0.2981424f, -0.21661313f, -0.3504874f },
        { 0.36749536f, 0.08738982f, -0.11388027f, 0.2981424f, -0.21661313f, -0.3504874f },
        { 0.36749536f, 0.08738982f, -0.11388027f, 0.2981424f, -0.21661313f, -0.3504874f },
        { 0.1397348f, -0.2811345f, 0.11388027f, -0.2981424f, 0.21661313f, 0.3504874f },
        { -0.36749536f, -0.08738982f, 0.11388027f, -0.2981424f, 0.21661313f, 0.3504874f },
        { -0.2981424f, -0.2981424f, 0.3504874f, 0.0f, 0.3504874f, 0.0f },
        { 0.2981424f, 0.2981424f, -0.3504874f, 0.0f, -0.3504874f, 0.0f },
        { 0.2981424f, 0.2981424f, -0.3504874f, 0.0f, -0.3504874f, 0.0f },
        { -0.2981424f, -0.2981424f, 0.3504874f, 0.0f, 0.3504874f, 0.0f },
        { -0.31348547f, -0.2811345f, -0.31348547f, 0.08738982f, 0.21661313f, -0.21661313f },
        { -0.31348547f, -0.2811345f, 0.31348547f, 0.2811345f, 0.21661313f, -0.21661313f },
        { -0.31348547f, 0.08738982f, -0.31348547f, -0.2811345f, -0.21661313f, 0.21661313f },
        { -0.31348547f, 0.08738982f, 0.31348547f, -0.08738982f, -0.21661313f, 0.21661313f },
        { 0.1397348f, 0.36749536f, 0.11388027f, 0.36852428f, -0.11388027f, -0.36852428f },
        { 0.1397348f, 0.36749536f, 0.11388027f, 0.36852428f, -0.11388027f, -0.36852428f },
        { 0.36749536f, 0.1397348f, -0.11388027f, -0.36852428f, 0.11388027f, 0.36852428f },
        { -0.1397348f, -0.36749536f, -0.11388027f, -0.36852428f, 0.11388027f, 0.36852428f }
    };
    
    inline void constructOrthonormalBasis(vec3 n, vec3 *b1, vec3 *b2) {
        float sign = copysign(1.0f, n.z);
        const float a = -1.0f / (sign + n.z);
        const float b = n.x * n.y * a;
        *b1 = vec3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
        *b2 = vec3(b, sign + n.y * n.y * a, -n.y);
    }
    
    void AmbientDice::hybridCubicBezierWeights(u32 triIndex, float b0, float b1, float b2, VertexWeights *w0Out, VertexWeights *w1Out, VertexWeights *w2Out) const {
        const float alpha = 0.5f * sqrt(0.5f * (5.0f + sqrt(5.0f))); // 0.9510565163
        const float beta = -0.5f * sqrt(0.1f * (5.0f + sqrt(5.0f))); // -0.4253254042

        const float a0 = (sqrt(5.0f) - 5.0f) / 40.0f; // -0.06909830056
        const float a1 = (11.0f * sqrt(5.0f) - 15.0f) / 40.0f; // 0.2399186938
        const float a2 = sqrt(5.0f) / 10.0f; // 0.2236067977

        const float fValueFactor = -beta / alpha; // 0.4472135955

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
        v0DUWeight += AmbientDice::triDerivativeTangentFactors[triIndex][0] * c210Weight;
        v0DVWeight += AmbientDice::triDerivativeBitangentFactors[triIndex][0] * c210Weight;
        
        v0ValueWeight += fValueFactor * c201Weight;
        v0DUWeight += AmbientDice::triDerivativeTangentFactors[triIndex][1] * c201Weight;
        v0DVWeight += AmbientDice::triDerivativeBitangentFactors[triIndex][1] * c201Weight;

        v1ValueWeight += fValueFactor * c120Weight;
        v1DUWeight += AmbientDice::triDerivativeTangentFactors[triIndex][2] * c120Weight;
        v1DVWeight += AmbientDice::triDerivativeBitangentFactors[triIndex][2] * c120Weight;

        v1ValueWeight += fValueFactor * c021Weight;
        v1DUWeight += AmbientDice::triDerivativeTangentFactors[triIndex][3] * c021Weight;
        v1DVWeight += AmbientDice::triDerivativeBitangentFactors[triIndex][3] * c021Weight;

        v2ValueWeight += fValueFactor * c102Weight;
        v2DUWeight += AmbientDice::triDerivativeTangentFactors[triIndex][4] * c102Weight;
        v2DVWeight += AmbientDice::triDerivativeBitangentFactors[triIndex][4] * c102Weight;

        v2ValueWeight += fValueFactor * c012Weight;
        v2DUWeight += AmbientDice::triDerivativeTangentFactors[triIndex][5] * c012Weight;
        v2DVWeight += AmbientDice::triDerivativeBitangentFactors[triIndex][5] * c012Weight;

        v0ValueWeight += c300Weight;
        v1ValueWeight += c030Weight;
        v2ValueWeight += c003Weight;

        *w0Out = { v0ValueWeight, v0DUWeight, v0DVWeight };
        *w1Out = { v1ValueWeight, v1DUWeight, v1DVWeight };
        *w2Out = { v2ValueWeight, v2DUWeight, v2DVWeight };
    }
    
//    void AmbientDice::hybridCubicBezierWeights(u32 i0, u32 i1, u32 i2, float b0, float b1, float b2, VertexWeights *w0Out, VertexWeights *w1Out, VertexWeights *w2Out) const {
//        const float alpha = 0.5f * sqrt(0.5f * (5.0f + sqrt(5.0f))); // 0.9510565163
//        const float beta = -0.5f * sqrt(0.1f * (5.0f + sqrt(5.0f))); // -0.4253254042
//
//        const float a0 = (sqrt(5.0f) - 5.0f) / 40.0f; // -0.06909830056
//        const float a1 = (11.0f * sqrt(5.0f) - 15.0f) / 40.0f; // 0.2399186938
//        const float a2 = sqrt(5.0f) / 10.0f; // 0.2236067977
//
//        // Project the edges onto the sphere.
//        // This amounts to a no-op since the edge vectors are already tangent to the sphere.
//        vec3 v0V1 = AmbientDice::vertexPositions[i1] - AmbientDice::vertexPositions[i0];
//        vec3 v1V0 = -v0V1;
//        vec3 v0V2 = AmbientDice::vertexPositions[i2] - AmbientDice::vertexPositions[i0];
//        vec3 v2V0 = -v0V2;
//        vec3 v1V2 = AmbientDice::vertexPositions[i2] - AmbientDice::vertexPositions[i1];
//        vec3 v2V1 = -v1V2;
//
//
//        const float fValueFactor = -beta / alpha; // 0.4472135955
//        const float fDerivativeFactor = 1.0 / (3.0 * alpha); // 0.3504874081
//
//        const float weightDenom = b1 * b2 + b0 * b2 + b0 * b1;
//
//        float w0 = (b1 * b2) / weightDenom;
//        float w1 = (b0 * b2) / weightDenom;
//        float w2 = (b0 * b1) / weightDenom;
//
//        if (b0 == 1.0) {
//            w0 = 1.0;
//            w1 = 0.0;
//            w2 = 0.0;
//        } else if (b1 == 1.0) {
//            w0 = 0.0;
//            w1 = 1.0;
//            w2 = 0.0;
//        } else if (b2 == 1.0) {
//            w0 = 0.0;
//            w1 = 0.0;
//            w2 = 1.0;
//        }
//
//        // https://en.wikipedia.org/wiki/Bézier_triangle
//        // Notation: cxyz means alpha^x, beta^y, gamma^z.
//
//        float v0ValueWeight = 0.0;
//        float v1ValueWeight = 0.0;
//        float v2ValueWeight = 0.0;
//
//        float v0DUWeight = 0.0;
//        float v1DUWeight = 0.0;
//        float v2DUWeight = 0.0;
//
//        float v0DVWeight = 0.0;
//        float v1DVWeight = 0.0;
//        float v2DVWeight = 0.0;
//
//        const float b0_2 = b0 * b0;
//        const float b1_2 = b1 * b1;
//        const float b2_2 = b2 * b2;
//
//        // Add c300, c030, and c003
//        float c300Weight = b0_2 * b0;
//        float c030Weight = b1_2 * b1;
//        float c003Weight = b2_2 * b2;
//
//        float c120Weight = 3 * b0 * b1_2;
//        float c021Weight = 3 * b1_2 * b2;
//        float c210Weight = 3 * b0_2 * b1;
//        float c012Weight = 3 * b1 * b2_2;
//        float c201Weight = 3 * b0_2 * b2;
//        float c102Weight = 3 * b0 * b2_2;
//
//        const float c111Weight = 6 * b0 * b1 * b2;
//        const float c0_111Weight = w0 * c111Weight;
//        const float c1_111Weight = w1 * c111Weight;
//        const float c2_111Weight = w2 * c111Weight;
//
//        v1ValueWeight += a0 * c0_111Weight;
//        v2ValueWeight += a0 * c1_111Weight;
//        v0ValueWeight += a0 * c2_111Weight;
//
//        c021Weight += a1 * c0_111Weight;
//        c012Weight += a1 * c0_111Weight;
//        c003Weight += a0 * c0_111Weight;
//        c120Weight += a2 * c0_111Weight;
//        c102Weight += a2 * c0_111Weight;
//
//        c102Weight += a1 * c1_111Weight;
//        c201Weight += a1 * c1_111Weight;
//        c300Weight += a0 * c1_111Weight;
//        c012Weight += a2 * c1_111Weight;
//        c210Weight += a2 * c1_111Weight;
//
//        c210Weight += a1 * c2_111Weight;
//        c120Weight += a1 * c2_111Weight;
//        c030Weight += a0 * c2_111Weight;
//        c201Weight += a2 * c2_111Weight;
//        c021Weight += a2 * c2_111Weight;
//
//
//        v0ValueWeight += fValueFactor * c210Weight;
//        v0DUWeight += fDerivativeFactor * dot(v0V1, AmbientDice::tangents[i0]) * c210Weight;
//        v0DVWeight += fDerivativeFactor * dot(v0V1, AmbientDice::bitangents[i0]) * c210Weight;
//
//        float dot0 = dot(v0V1, AmbientDice::tangents[i0]);
//        float dot1 = dot(v0V1, AmbientDice::bitangents[i0]);
//
//        v0ValueWeight += fValueFactor * c201Weight;
//        v0DUWeight += fDerivativeFactor * dot(v0V2, AmbientDice::tangents[i0]) * c201Weight;
//        v0DVWeight += fDerivativeFactor * dot(v0V2, AmbientDice::bitangents[i0]) * c201Weight;
//
//        float dot2 = dot(v0V2, AmbientDice::tangents[i0]);
//        float dot3 = dot(v0V2, AmbientDice::bitangents[i0]);
//
//        v1ValueWeight += fValueFactor * c120Weight;
//        v1DUWeight += fDerivativeFactor * dot(v1V0, AmbientDice::tangents[i1]) * c120Weight;
//        v1DVWeight += fDerivativeFactor * dot(v1V0, AmbientDice::bitangents[i1]) * c120Weight;
//
//        float dot4 = dot(v1V0, AmbientDice::tangents[i1]);
//        float dot5 = dot(v1V0, AmbientDice::bitangents[i1]);
//
//        v1ValueWeight += fValueFactor * c021Weight;
//        v1DUWeight += fDerivativeFactor * dot(v1V2, AmbientDice::tangents[i1]) * c021Weight;
//        v1DVWeight += fDerivativeFactor * dot(v1V2, AmbientDice::bitangents[i1]) * c021Weight;
//
//        float dot6 = dot(v1V2, AmbientDice::tangents[i1]);
//        float dot7 = dot(v1V2, AmbientDice::bitangents[i1]);
//
//        v2ValueWeight += fValueFactor * c102Weight;
//        v2DUWeight += fDerivativeFactor * dot(v2V0, AmbientDice::tangents[i2]) * c102Weight;
//        v2DVWeight += fDerivativeFactor * dot(v2V0, AmbientDice::bitangents[i2]) * c102Weight;
//
//        float dot8 = dot(v2V0, AmbientDice::tangents[i2]);
//        float dot9 = dot(v2V0, AmbientDice::bitangents[i2]);
//
//        v2ValueWeight += fValueFactor * c012Weight;
//        v2DUWeight += fDerivativeFactor * dot(v2V1, AmbientDice::tangents[i2]) * c012Weight;
//        v2DVWeight += fDerivativeFactor * dot(v2V1, AmbientDice::bitangents[i2]) * c012Weight;
//
//        float dot10 = dot(v2V1, AmbientDice::tangents[i2]);
//        float dot11 = dot(v2V1, AmbientDice::bitangents[i2]);
//
//        printf("Dot Products: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", dot0, dot1, dot2, dot3, dot4, dot5, dot6, dot7, dot8, dot9, dot10, dot11);
//
//        v0ValueWeight += c300Weight;
//        v1ValueWeight += c030Weight;
//        v2ValueWeight += c003Weight;
//
//        *w0Out = { v0ValueWeight, v0DUWeight, v0DVWeight };
//        *w1Out = { v1ValueWeight, v1DUWeight, v1DVWeight };
//        *w2Out = { v2ValueWeight, v2DUWeight, v2DVWeight };
//    }
    
    void AmbientDice::hybridCubicBezierWeights(vec3 direction, u32 *i0Out, u32 *i1Out, u32 *i2Out, VertexWeights *w0Out, VertexWeights *w1Out, VertexWeights *w2Out) const {
        
        u32 triIndex, i0, i1, i2;
        float b0, b1, b2;
        this->computeBarycentrics(direction, &triIndex, &i0, &i1, &i2, &b0, &b1, &b2);
        
        this->hybridCubicBezierWeights(triIndex, b0, b1, b2, w0Out, w1Out, w2Out);
        
        *i0Out = i0;
        *i1Out = i1;
        *i2Out = i2;
    }
    
    void AmbientDice::srbfWeights(vec3 direction, float *weightsOut) const {
        for (u64 i = 0; i < 6; i += 1) {
            float dotProduct = dot(direction, AmbientDice::srbfNormalisedVertexPositions[i]);
            u32 index = dotProduct > 0 ? (2 * i) : (2 * i + 1);
            
            float cos2 = dotProduct * dotProduct;
            float cos4 = cos2 * cos2;
            
            weightsOut[index] = 0.7f * (0.5f * cos2) + 0.3f * (5.f / 6.f * cos4);
        }
    }
    
    vec3 AmbientDice::evaluateSRBF(const vec3& direction) const
    {
        vec3 result = vec3(0.f);
        for (u64 i = 0; i < 6; i += 1) {
            float dotProduct = dot(direction, AmbientDice::srbfNormalisedVertexPositions[i]);
            u32 index = dotProduct > 0 ? (2 * i) : (2 * i + 1);
            
            float cos2 = dotProduct * dotProduct;
            float cos4 = cos2 * cos2;

            float weight = 0.7f * (0.5f * cos2) + 0.3f * (5.f / 6.f * cos4);

//            float weight = 0.6f * exp(3.1f * (dotProduct - 1.f));
            result += weight * this->vertices[index].value;
        }
        
        return result;
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
            
            vec3 delta = vec3(targetValue.x, targetValue.y, targetValue.z) - currentEstimate;
            
            const float sampleWeightScale = 1.f / sampleIndex;
            
            const float alpha = 3.0f;
            const bool gaussSeidel = false;
            
            for (u64 i = 0; i < 3; i += 1) {
                u32 index = indices[i];
                const AmbientDice::VertexWeights &weight = weights[i];
                
                vertexWeights[index].value += (weight.value * weight.value - vertexWeights[index].value) * sampleWeightScale;
                vertexWeights[index].directionalDerivativeU += (weight.directionalDerivativeU * weight.directionalDerivativeU - vertexWeights[index].directionalDerivativeU) * sampleWeightScale;
                vertexWeights[index].directionalDerivativeV += (weight.directionalDerivativeV * weight.directionalDerivativeV - vertexWeights[index].directionalDerivativeV) * sampleWeightScale;
                
                {
                    float sphericalIntegral = sampleWeightScale + (1 - sampleWeightScale) * vertexWeights[index].value;
                    
                    float deltaScale = alpha * weights[i].value * sampleWeightScale / sphericalIntegral;
                    ambientDice.vertices[index].value += delta * deltaScale;
                    
                    if (gaussSeidel) {
                        delta *= 1.0f - deltaScale * vertexWeights[index].value;
                    }
                }
                
                {
                    float sphericalIntegral = sampleWeightScale + (1 - sampleWeightScale) * vertexWeights[index].directionalDerivativeU;
                    
                    float deltaScale = alpha * weights[i].directionalDerivativeU * sampleWeightScale / sphericalIntegral;
                    ambientDice.vertices[index].directionalDerivativeU += delta * deltaScale;
                    
                    if (gaussSeidel) {
                        delta *= 1.0f - deltaScale * vertexWeights[index].value;
                    }
                }
                
                {
                    float sphericalIntegral = sampleWeightScale + (1 - sampleWeightScale) * vertexWeights[index].directionalDerivativeV;
                    
                    float deltaScale = alpha * weights[i].directionalDerivativeV * sampleWeightScale / sphericalIntegral;
                    ambientDice.vertices[index].directionalDerivativeV += delta * deltaScale;
                    
                    if (gaussSeidel) {
                        delta *= 1.0f - deltaScale * vertexWeights[index].value;
                    }
                }
                
                if (false /* nonNegative */) {
                    ambientDice.vertices[index].value = max(ambientDice.vertices[index].value, vec3(0.f));
                    ambientDice.vertices[index].directionalDerivativeU = max(ambientDice.vertices[index].directionalDerivativeU, vec3(0.f));
                    ambientDice.vertices[index].directionalDerivativeV = max(ambientDice.vertices[index].directionalDerivativeV, vec3(0.f));
                }
            }
            
            for (u64 index = 0; index < 12; index += 1) {
                if (index == i0 || index == i1 || index == i2) continue;
                vertexWeights[index].value *= (1.f - sampleWeightScale);
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
            
            float weights[12] = { 0.f };
            ambientDice.srbfWeights(direction, weights);
            
            // What's the current value in the sample's direction?
            vec4 targetValue = irradiance.at(sampleIt);
            vec3 currentEstimate = ambientDice.evaluateSRBF(direction);
            
            const vec3 delta = vec3(targetValue.x, targetValue.y, targetValue.z) - currentEstimate;
            
            const float sampleWeightScale = 1.f / sampleIndex;
            
            const float alpha = 3.0f;
            
            for (u64 i = 0; i < 12; i += 1) {
                float weight = weights[i];
                
                vertexWeights[i].value += (weight * weight - vertexWeights[i].value) * sampleWeightScale;
                
                float sphericalIntegral = sampleWeightScale + (1 - sampleWeightScale) * vertexWeights[i].value;
                
                float deltaScale = alpha * weight * sampleWeightScale / sphericalIntegral;
                ambientDice.vertices[i].value += delta * deltaScale;
                
                if (false /* nonNegative */) {
                    ambientDice.vertices[i].value = max(ambientDice.vertices[i].value, vec3(0.f));
                }
            }
        }
        
        return ambientDice;
    }
    
//    AmbientDice ExperimentAmbientDice::solveAmbientDiceLeastSquares(ImageBase<vec3>& directions, const Image& irradiance)
//    {
//        using namespace Eigen;
//
//        AmbientDice ambientDice;
//
//        MatrixXf moments = MatrixXf::Zero(36, 3);
//
//        const ivec2 imageSize = directions.getSize();
//        directions.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
//                                          {
//                                              float texelArea = latLongTexelArea(pixelPos, imageSize);
////                                              float texelArea = 1.f / directions.getPixelCount();
//
//            const vec4& colour = irradiance.at(pixelPos);
//
//            u32 i0, i1, i2;
//            AmbientDice::VertexWeights weights[3];
//            ambientDice.hybridCubicBezierWeights(direction, &i0, &i1, &i2, &weights[0], &weights[1], &weights[2]);
//
//                                              moments(3 * i0 + 0, 0) += weights[0].value * colour.r * texelArea;
//                                              moments(3 * i0 + 1, 0) += weights[0].directionalDerivativeU * colour.r * texelArea;
//                                              moments(3 * i0 + 2, 0) += weights[0].directionalDerivativeV * colour.r * texelArea;
//                                              moments(3 * i1 + 0, 0) += weights[1].value * colour.r * texelArea;
//                                              moments(3 * i1 + 1, 0) += weights[1].directionalDerivativeU * colour.r * texelArea;
//                                              moments(3 * i1 + 2, 0) += weights[1].directionalDerivativeV * colour.r * texelArea;
//                                              moments(3 * i2 + 0, 0) += weights[2].value * colour.r * texelArea;
//                                              moments(3 * i2 + 1, 0) += weights[2].directionalDerivativeU * colour.r * texelArea;
//                                              moments(3 * i2 + 2, 0) += weights[2].directionalDerivativeV * colour.r * texelArea;
//                                              moments(3 * i0 + 0, 1) += weights[0].value * colour.g * texelArea;
//                                              moments(3 * i0 + 1, 1) += weights[0].directionalDerivativeU * colour.g * texelArea;
//                                              moments(3 * i0 + 2, 1) += weights[0].directionalDerivativeV * colour.g * texelArea;
//                                              moments(3 * i1 + 0, 1) += weights[1].value * colour.g * texelArea;
//                                              moments(3 * i1 + 1, 1) += weights[1].directionalDerivativeU * colour.g * texelArea;
//                                              moments(3 * i1 + 2, 1) += weights[1].directionalDerivativeV * colour.g * texelArea;
//                                              moments(3 * i2 + 0, 1) += weights[2].value * colour.g * texelArea;
//                                              moments(3 * i2 + 1, 1) += weights[2].directionalDerivativeU * colour.g * texelArea;
//                                              moments(3 * i2 + 2, 1) += weights[2].directionalDerivativeV * colour.g * texelArea;
//                                              moments(3 * i0 + 0, 2) += weights[0].value * colour.b * texelArea;
//                                              moments(3 * i0 + 1, 2) += weights[0].directionalDerivativeU * colour.b * texelArea;
//                                              moments(3 * i0 + 2, 2) += weights[0].directionalDerivativeV * colour.b * texelArea;
//                                              moments(3 * i1 + 0, 2) += weights[1].value * colour.b * texelArea;
//                                              moments(3 * i1 + 1, 2) += weights[1].directionalDerivativeU * colour.b * texelArea;
//                                              moments(3 * i1 + 2, 2) += weights[1].directionalDerivativeV * colour.b * texelArea;
//                                              moments(3 * i2 + 0, 2) += weights[2].value * colour.b * texelArea;
//                                              moments(3 * i2 + 1, 2) += weights[2].directionalDerivativeU * colour.b * texelArea;
//                                              moments(3 * i2 + 2, 2) += weights[2].directionalDerivativeV * colour.b * texelArea;
//                                          });
//
//        MatrixXf gram;
//        gram.resize(36, 36);
//        for (u64 lobeAIt = 0; lobeAIt < 36; ++lobeAIt)
//        {
//            for (u64 lobeBIt = lobeAIt; lobeBIt < 36; ++lobeBIt)
//            {
//                float integral = 0.f;
//
//                directions.forPixels2D([&](const vec3& direction, ivec2 pixelPos) {
//                    float texelArea = latLongTexelArea(pixelPos, imageSize);
////                    float texelArea = 1.f / directions.getPixelCount();
//
//                    float allWeights[36] = { 0.f };
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
//                    integral += allWeights[lobeAIt] * allWeights[lobeBIt] * texelArea;
//                });
//
////                integral *= /* 4 * M_PI */ 1.f / float(sampleCount);
//
//                gram(lobeAIt, lobeBIt) = integral;
//                gram(lobeBIt, lobeAIt) = integral;
//            }
//        }
//
//        std::cout << gram << std::endl;
//
//        auto solver = gram.jacobiSvd(ComputeThinU | ComputeThinV);
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
//            VectorXf x = solver.solve(b);
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
    
    AmbientDice ExperimentAmbientDice::solveAmbientDiceLeastSquares(ImageBase<vec3>& directions, const Image& irradiance)
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

        auto solver = A.jacobiSvd(ComputeThinU | ComputeThinV);

        VectorXf b;
        b.resize(sampleCount);

        for (u32 channelIt = 0; channelIt < 3; ++channelIt)
        {
            for (u64 sampleIt = 0; sampleIt < sampleCount; ++sampleIt)
            {
                b[sampleIt] = irradiance.at(sampleIt)[channelIt];
            }

            VectorXf x = solver.solve(b);

            for (u64 basisIt = 0; basisIt < 12; ++basisIt)
            {
                ambientDice.vertices[basisIt].value[channelIt] = x[3 * basisIt];
                ambientDice.vertices[basisIt].directionalDerivativeU[channelIt] = x[3 * basisIt + 1];
                ambientDice.vertices[basisIt].directionalDerivativeV[channelIt] = x[3 * basisIt + 2];
            }
        }

        return ambientDice;
    }
    
    AmbientDice ExperimentAmbientDice::solveAmbientDiceLeastSquaresSRBF(ImageBase<vec3>& directions, const Image& irradiance)
    {
        using namespace Eigen;

        AmbientDice ambientDice;

        const u64 sampleCount = directions.getPixelCount();

        MatrixXf A = MatrixXf::Zero(sampleCount, 12);

        for (u64 sampleIt = 0; sampleIt < sampleCount; ++sampleIt)
        {
            const vec3& direction = directions.at(sampleIt);

            float weights[12] = { 0.f };
            ambientDice.srbfWeights(direction, weights);

            for (u64 i = 0; i < 12; i += 1) {
                A(sampleIt, i) = weights[i];
            }

        }

        auto solver = A.jacobiSvd(ComputeThinU | ComputeThinV);

        VectorXf b;
        b.resize(sampleCount);

        for (u32 channelIt = 0; channelIt < 3; ++channelIt)
        {
            for (u64 sampleIt = 0; sampleIt < sampleCount; ++sampleIt)
            {
                b[sampleIt] = irradiance.at(sampleIt)[channelIt];
            }

            VectorXf x = solver.solve(b);

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
                                                  
//                                                  assert(fabs(sampleBF.r - otherSample.r) < 1e-4f);
//                                                  assert(fabs(sampleBF.g - otherSample.g) < 1e-4f);
//                                                  assert(fabs(sampleBF.b - otherSample.b) < 1e-4f);
                                                  
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
        
        m_radianceImage = Image(data.m_outputSize);
        m_specularImage = Image(data.m_outputSize);
        m_irradianceImage = Image(data.m_outputSize);
        
//        compareResponseImages(data);
        
        if (m_diceType == AmbientDiceTypeBezier) {
            AmbientDice ambientDiceRadiance = solveAmbientDiceLeastSquares(data.m_directionImage, m_input->m_radianceImage);
            AmbientDice ambientDiceSpecular = solveAmbientDiceLeastSquares(data.m_directionImage, m_input->m_specularImage);
            AmbientDice ambientDiceIrradiance = solveAmbientDiceLeastSquares(data.m_directionImage, m_input->m_irradianceImage);
        
//        float radianceFactor = 1.f - sqrt(ggxAlpha);
//        float irradianceFactor = 1.f - radianceFactor;
//        
//        for (u32 i = 0; i < 12; i += 1) {
//            
//            ambientDiceSpecular.vertices[i].value = ambientDiceRadiance.vertices[i].value * radianceFactor + ambientDice.vertices[i].value * irradianceFactor;
//            ambientDiceSpecular.vertices[i].directionalDerivativeU = ambientDiceRadiance.vertices[i].directionalDerivativeU * radianceFactor + ambientDice.vertices[i].directionalDerivativeU * irradianceFactor;
//            ambientDiceSpecular.vertices[i].directionalDerivativeV = ambientDiceRadiance.vertices[i].directionalDerivativeV * radianceFactor + ambientDice.vertices[i].directionalDerivativeV * irradianceFactor;
//        }
        
        data.m_directionImage.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
                                      {
                                          vec3 sampleRadiance = ambientDiceRadiance.evaluateBezier(direction);
                                          m_radianceImage.at(pixelPos) = vec4(sampleRadiance, 1.0f);
                                          
                                          vec3 sampleIrradiance = ambientDiceIrradiance.evaluateBezier(direction);
                                          m_irradianceImage.at(pixelPos) = vec4(sampleIrradiance, 1.0f);
                                          
                                          vec3 sampleSpecular = ambientDiceSpecular.evaluateBezier(direction);
                                          m_specularImage.at(pixelPos) = vec4(sampleSpecular, 1.0f);
                                      });
        } else /* if (m_diceType == AmbientDiceTypeSRBF) */ {
            AmbientDice ambientDiceRadiance = solveAmbientDiceLeastSquaresSRBF(data.m_directionImage, m_input->m_radianceImage);
            AmbientDice ambientDiceSpecular = solveAmbientDiceLeastSquaresSRBF(data.m_directionImage, m_input->m_specularImage);
            AmbientDice ambientDiceIrradiance = solveAmbientDiceLeastSquaresSRBF(data.m_directionImage, m_input->m_irradianceImage);
            
            data.m_directionImage.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
                                              {
                                                  vec3 sampleRadiance = ambientDiceRadiance.evaluateSRBF(direction);
                                                  m_radianceImage.at(pixelPos) = vec4(sampleRadiance, 1.0f);
                                                  
                                                  vec3 sampleIrradiance = ambientDiceIrradiance.evaluateSRBF(direction);
                                                  m_irradianceImage.at(pixelPos) = vec4(sampleIrradiance, 1.0f);
                                                  
                                                  vec3 sampleSpecular = ambientDiceSpecular.evaluateSRBF(direction);
                                                  m_specularImage.at(pixelPos) = vec4(sampleSpecular, 1.0f);
                                              });
        }
    }
}
