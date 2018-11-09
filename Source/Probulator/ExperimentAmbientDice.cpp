#include <iostream>

#include "ExperimentAmbientDice.h"

#include <Eigen/Eigen>
#include <Eigen/nnls.h>

#include "ExperimentAmbientD20.h"
#include "MicrosurfaceScattering.h"

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
    
    const double AmbientDice::gramMatrixBezier[36][36] = {
        { 0.591709, -1.70158e-08, 8.35533e-08, 0, 0, 0, 0.0747991, 0.0178657, -0.0289073, 0, 0, 0, 0.0747988, 0.0323192, 0.0105012, 0, 0, 0, 0.0747987, 0.0323191, -0.0105011, 0, 0, 0, 0.074799, 0.0274924, 0.0199745, 0, 0, 0, 0.0747991, 0.0274924, -0.0199744, 0, 0, 0, },
        { -1.70158e-08, 0.0389574, -3.19622e-08, 0, 0, 0, 0.0178657, 0.00123648, -0.0081224, 0, 0, 0, -0.0314448, -0.0130972, -0.00274716, 0, 0, 0, 0.00253763, 0.00218706, 0.00325638, 0, 0, 0, -0.0219717, -0.00582474, -0.00779951, 0, 0, 0, 0.0330133, 0.0118283, -0.00748472, 0, 0, 0, },
        { 8.35533e-08, -3.19622e-08, 0.0389574, 0, 0, 0, -0.0289073, -0.0081224, 0.0093589, 0, 0, 0, 0.0128853, 0.00410339, 0.00501432, 0, 0, 0, 0.0338875, 0.0135495, -0.0046996, 0, 0, 0, -0.025924, -0.010312, -0.00446844, 0, 0, 0, 0.00805843, 0.000598114, -0.00497774, 0, 0, 0, },
        { 0, 0, 0, 0.591709, 5.35355e-08, 8.20932e-08, 0, 0, 0, 0.074799, -0.0178657, 0.0289073, 0.0747987, -0.0323191, 0.0105011, 0, 0, 0, 0.0747988, -0.0323192, -0.0105012, 0, 0, 0, 0, 0, 0, 0.0747991, -0.0274924, 0.0199744, 0, 0, 0, 0.074799, -0.0274924, -0.0199745, },
        { 0, 0, 0, 5.35355e-08, 0.0389573, 2.12357e-08, 0, 0, 0, -0.0178657, 0.00671193, -0.0047384, 0.0314448, -0.0130972, 0.00274715, 0, 0, 0, -0.0025376, 0.00218704, -0.00325638, 0, 0, 0, 0, 0, 0, 0.0219716, -0.00582473, 0.0077995, 0, 0, 0, -0.0330131, 0.0118282, 0.00748474, },
        { 0, 0, 0, 8.20932e-08, 2.12357e-08, 0.0389574, 0, 0, 0, -0.0289072, 0.00473839, -0.0114503, 0.0128852, -0.00410338, 0.00501432, 0, 0, 0, 0.0338876, -0.0135495, -0.00469961, 0, 0, 0, 0, 0, 0, -0.0259241, 0.010312, -0.00446843, 0, 0, 0, 0.0080585, -0.000598139, -0.00497775, },
        { 0.0747991, 0.0178657, -0.0289073, 0, 0, 0, 0.591709, -1.54771e-08, 8.38016e-08, 0, 0, 0, 0, 0, 0, 0.0747987, 0.0323191, -0.0105011, 0, 0, 0, 0.0747988, 0.0323192, 0.0105012, 0.0747991, 0.0274924, -0.0199744, 0, 0, 0, 0.074799, 0.0274924, 0.0199745, 0, 0, 0, },
        { 0.0178657, 0.00123648, -0.0081224, 0, 0, 0, -1.54771e-08, 0.0389574, -3.15809e-08, 0, 0, 0, 0, 0, 0, 0.00253763, 0.00218706, 0.00325638, 0, 0, 0, -0.0314448, -0.0130971, -0.00274716, 0.0330133, 0.0118283, -0.00748472, 0, 0, 0, -0.0219717, -0.00582474, -0.00779951, 0, 0, 0, },
        { -0.0289073, -0.0081224, 0.0093589, 0, 0, 0, 8.38016e-08, -3.15809e-08, 0.0389574, 0, 0, 0, 0, 0, 0, 0.0338875, 0.0135495, -0.0046996, 0, 0, 0, 0.0128853, 0.00410339, 0.00501432, 0.00805842, 0.000598114, -0.00497774, 0, 0, 0, -0.025924, -0.010312, -0.00446844, 0, 0, 0, },
        { 0, 0, 0, 0.074799, -0.0178657, -0.0289072, 0, 0, 0, 0.591709, 2.12892e-08, -9.19825e-08, 0, 0, 0, 0.0747988, -0.0323192, -0.0105012, 0, 0, 0, 0.0747987, -0.0323191, 0.0105011, 0, 0, 0, 0.074799, -0.0274924, -0.0199745, 0, 0, 0, 0.0747991, -0.0274924, 0.0199744, },
        { 0, 0, 0, -0.0178657, 0.00671193, 0.00473839, 0, 0, 0, 2.12892e-08, 0.0389574, -2.91868e-08, 0, 0, 0, 0.0314448, -0.0130972, -0.00274716, 0, 0, 0, -0.00253763, 0.00218706, 0.00325638, 0, 0, 0, 0.0219717, -0.00582474, -0.00779951, 0, 0, 0, -0.0330133, 0.0118283, -0.00748472, },
        { 0, 0, 0, 0.0289073, -0.0047384, -0.0114503, 0, 0, 0, -9.19825e-08, -2.91868e-08, 0.0389574, 0, 0, 0, -0.0128853, 0.00410339, 0.00501432, 0, 0, 0, -0.0338875, 0.0135495, -0.0046996, 0, 0, 0, 0.025924, -0.010312, -0.00446844, 0, 0, 0, -0.00805843, 0.000598114, -0.00497774, },
        { 0.0747988, -0.0314448, 0.0128853, 0.0747987, 0.0314448, 0.0128852, 0, 0, 0, 0, 0, 0, 0.591709, 5.47219e-09, 7.24456e-08, 0, 0, 0, 0.0747988, -5.29109e-09, -0.0339824, 0, 0, 0, 0.074799, -0.0105013, 0.0323193, 0.0747991, 0.0105012, 0.0323193, 0, 0, 0, 0, 0, 0, },
        { 0.0323192, -0.0130972, 0.00410339, -0.0323191, -0.0130972, -0.00410338, 0, 0, 0, 0, 0, 0, 5.47219e-09, 0.0389574, -2.67962e-09, 0, 0, 0, 4.07076e-09, 0.00378339, -9.7413e-10, 0, 0, 0, 0.0199743, 0.000299328, 0.00898386, -0.0199744, 0.000299349, -0.00898387, 0, 0, 0, 0, 0, 0, },
        { 0.0105012, -0.00274716, 0.00501432, 0.0105011, 0.00274715, 0.00501432, 0, 0, 0, 0, 0, 0, 7.24456e-08, -2.67962e-09, 0.0389574, 0, 0, 0, 0.0339824, -1.50768e-09, -0.0143788, 0, 0, 0, -0.0274925, 0.00570977, -0.0103761, -0.0274925, -0.00570977, -0.0103761, 0, 0, 0, 0, 0, 0, },
        { 0, 0, 0, 0, 0, 0, 0.0747987, 0.00253763, 0.0338875, 0.0747988, 0.0314448, -0.0128853, 0, 0, 0, 0.591709, -9.8812e-09, -7.6907e-08, 0, 0, 0, 0.0747988, 4.7105e-09, 0.0339824, 0.0747991, -0.0105012, -0.0323193, 0.074799, 0.0105013, -0.0323193, 0, 0, 0, 0, 0, 0, },
        { 0, 0, 0, 0, 0, 0, 0.0323191, 0.00218706, 0.0135495, -0.0323192, -0.0130972, 0.00410339, 0, 0, 0, -9.8812e-09, 0.0389574, -2.37552e-09, 0, 0, 0, -4.59195e-09, 0.00378339, -1.18189e-09, 0.0199744, 0.000299349, -0.00898387, -0.0199743, 0.000299328, 0.00898386, 0, 0, 0, 0, 0, 0, },
        { 0, 0, 0, 0, 0, 0, -0.0105011, 0.00325638, -0.0046996, -0.0105012, -0.00274716, 0.00501432, 0, 0, 0, -7.6907e-08, -2.37552e-09, 0.0389574, 0, 0, 0, -0.0339824, -1.34707e-09, -0.0143788, 0.0274925, -0.00570977, -0.0103761, 0.0274925, 0.00570977, -0.0103761, 0, 0, 0, 0, 0, 0, },
        { 0.0747987, 0.00253763, 0.0338875, 0.0747988, -0.0025376, 0.0338876, 0, 0, 0, 0, 0, 0, 0.0747988, 4.07076e-09, 0.0339824, 0, 0, 0, 0.591709, -1.37871e-08, -7.15453e-08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0747991, -0.0105012, -0.0323193, 0.074799, 0.0105013, -0.0323193, },
        { 0.0323191, 0.00218706, 0.0135495, -0.0323192, 0.00218704, -0.0135495, 0, 0, 0, 0, 0, 0, -5.29109e-09, 0.00378339, -1.50768e-09, 0, 0, 0, -1.37871e-08, 0.0389574, -2.42242e-09, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0199744, 0.000299349, -0.00898387, -0.0199743, 0.000299328, 0.00898386, },
        { -0.0105011, 0.00325638, -0.0046996, -0.0105012, -0.00325638, -0.00469961, 0, 0, 0, 0, 0, 0, -0.0339824, -9.7413e-10, -0.0143788, 0, 0, 0, -7.15453e-08, -2.42242e-09, 0.0389574, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0274925, -0.00570977, -0.0103761, 0.0274925, 0.00570977, -0.0103761, },
        { 0, 0, 0, 0, 0, 0, 0.0747988, -0.0314448, 0.0128853, 0.0747987, -0.00253763, -0.0338875, 0, 0, 0, 0.0747988, -4.59195e-09, -0.0339824, 0, 0, 0, 0.591709, 1.09827e-08, 7.52977e-08, 0, 0, 0, 0, 0, 0, 0.074799, -0.0105013, 0.0323193, 0.0747991, 0.0105012, 0.0323193, },
        { 0, 0, 0, 0, 0, 0, 0.0323192, -0.0130971, 0.00410339, -0.0323191, 0.00218706, 0.0135495, 0, 0, 0, 4.7105e-09, 0.00378339, -1.34707e-09, 0, 0, 0, 1.09827e-08, 0.0389574, -2.45215e-09, 0, 0, 0, 0, 0, 0, 0.0199743, 0.000299328, 0.00898386, -0.0199744, 0.000299349, -0.00898387, },
        { 0, 0, 0, 0, 0, 0, 0.0105012, -0.00274716, 0.00501432, 0.0105011, 0.00325638, -0.0046996, 0, 0, 0, 0.0339824, -1.18189e-09, -0.0143788, 0, 0, 0, 7.52977e-08, -2.45215e-09, 0.0389574, 0, 0, 0, 0, 0, 0, -0.0274925, 0.00570977, -0.0103761, -0.0274925, -0.00570977, -0.0103761, },
        { 0.074799, -0.0219717, -0.025924, 0, 0, 0, 0.0747991, 0.0330133, 0.00805842, 0, 0, 0, 0.074799, 0.0199743, -0.0274925, 0.0747991, 0.0199744, 0.0274925, 0, 0, 0, 0, 0, 0, 0.591744, -1.7741e-05, 3.51106e-07, 0.0748425, 0.0340023, 3.01532e-07, 0, 0, 0, 0, 0, 0, },
        { 0.0274924, -0.00582474, -0.010312, 0, 0, 0, 0.0274924, 0.0118283, 0.000598114, 0, 0, 0, -0.0105013, 0.000299328, 0.00570977, -0.0105012, 0.000299349, -0.00570977, 0, 0, 0, 0, 0, 0, -1.7741e-05, 0.0389661, -1.27422e-07, -0.0340038, -0.0143885, -1.37941e-07, 0, 0, 0, 0, 0, 0, },
        { 0.0199745, -0.00779951, -0.00446844, 0, 0, 0, -0.0199744, -0.00748472, -0.00497774, 0, 0, 0, 0.0323193, 0.00898386, -0.0103761, -0.0323193, -0.00898387, -0.0103761, 0, 0, 0, 0, 0, 0, 3.51106e-07, -1.27422e-07, 0.0389575, 3.05637e-07, 1.49163e-07, 0.00378353, 0, 0, 0, 0, 0, 0, },
        { 0, 0, 0, 0.0747991, 0.0219716, -0.0259241, 0, 0, 0, 0.074799, 0.0219717, 0.025924, 0.0747991, -0.0199744, -0.0274925, 0.074799, -0.0199743, 0.0274925, 0, 0, 0, 0, 0, 0, 0.0748425, -0.0340038, 3.05637e-07, 0.591758, 2.27911e-05, 2.53957e-07, 0, 0, 0, 0, 0, 0, },
        { 0, 0, 0, -0.0274924, -0.00582473, 0.010312, 0, 0, 0, -0.0274924, -0.00582474, -0.010312, 0.0105012, 0.000299349, -0.00570977, 0.0105013, 0.000299328, 0.00570977, 0, 0, 0, 0, 0, 0, 0.0340023, -0.0143885, 1.49163e-07, 2.27911e-05, 0.0389678, 1.58954e-07, 0, 0, 0, 0, 0, 0, },
        { 0, 0, 0, 0.0199744, 0.0077995, -0.00446843, 0, 0, 0, -0.0199745, -0.00779951, -0.00446844, 0.0323193, -0.00898387, -0.0103761, -0.0323193, 0.00898386, -0.0103761, 0, 0, 0, 0, 0, 0, 3.01532e-07, -1.37941e-07, 0.00378353, 2.53957e-07, 1.58954e-07, 0.0389575, 0, 0, 0, 0, 0, 0, },
        { 0.0747991, 0.0330133, 0.00805843, 0, 0, 0, 0.074799, -0.0219717, -0.025924, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0747991, 0.0199744, 0.0274925, 0.074799, 0.0199743, -0.0274925, 0, 0, 0, 0, 0, 0, 0.591661, 2.19453e-05, 5.28521e-07, 0.0747558, 0.0339617, 4.61022e-07, },
        { 0.0274924, 0.0118283, 0.000598114, 0, 0, 0, 0.0274924, -0.00582474, -0.010312, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0105012, 0.000299349, -0.00570977, -0.0105013, 0.000299328, 0.00570977, 0, 0, 0, 0, 0, 0, 2.19453e-05, 0.0389472, -2.08718e-07, -0.0339625, -0.0143692, -2.10188e-07, },
        { -0.0199744, -0.00748472, -0.00497774, 0, 0, 0, 0.0199745, -0.00779951, -0.00446844, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0323193, -0.00898387, -0.0103761, 0.0323193, 0.00898386, -0.0103761, 0, 0, 0, 0, 0, 0, 5.28521e-07, -2.08718e-07, 0.0389575, 4.58088e-07, 2.24171e-07, 0.00378353, },
        { 0, 0, 0, 0.074799, -0.0330131, 0.0080585, 0, 0, 0, 0.0747991, -0.0330133, -0.00805843, 0, 0, 0, 0, 0, 0, 0.074799, -0.0199743, 0.0274925, 0.0747991, -0.0199744, -0.0274925, 0, 0, 0, 0, 0, 0, 0.0747558, -0.0339625, 4.58088e-07, 0.591668, -1.94045e-05, 3.81176e-07, },
        { 0, 0, 0, -0.0274924, 0.0118282, -0.000598139, 0, 0, 0, -0.0274924, 0.0118283, 0.000598114, 0, 0, 0, 0, 0, 0, 0.0105013, 0.000299328, 0.00570977, 0.0105012, 0.000299349, -0.00570977, 0, 0, 0, 0, 0, 0, 0.0339617, -0.0143692, 2.24171e-07, -1.94045e-05, 0.038948, 2.2463e-07, },
        { 0, 0, 0, -0.0199745, 0.00748474, -0.00497775, 0, 0, 0, 0.0199744, -0.00748472, -0.00497774, 0, 0, 0, 0, 0, 0, -0.0323193, 0.00898386, -0.0103761, 0.0323193, -0.00898387, -0.0103761, 0, 0, 0, 0, 0, 0, 4.61022e-07, -2.10188e-07, 0.00378353, 3.81176e-07, 2.2463e-07, 0.0389575, },
    };
    
    const double AmbientDice::gramMatrixSRBF[12][12] = {
        { 0.354651, 0, 0.00790714, 0.130602, 0.130602, 0.00790713, 0.00790713, 0.130602, 0.130602, 0.00790712, 0.130602, 0.00790712, },
        { 0, 0.354651, 0.130602, 0.00790714, 0.00790714, 0.130602, 0.130602, 0.00790714, 0.00790713, 0.130602, 0.00790713, 0.130602, },
        { 0.00790714, 0.130602, 0.354651, 0, 0.130602, 0.00790714, 0.00790714, 0.130602, 0.00790713, 0.130602, 0.00790713, 0.130602, },
        { 0.130602, 0.00790714, 0, 0.354651, 0.00790713, 0.130602, 0.130602, 0.00790713, 0.130602, 0.00790712, 0.130602, 0.00790712, },
        { 0.130602, 0.00790714, 0.130602, 0.00790713, 0.354654, 0, 0.00790971, 0.130602, 0.130611, 0.00790713, 0.00790714, 0.130611, },
        { 0.00790713, 0.130602, 0.00790714, 0.130602, 0, 0.354649, 0.130602, 0.00790456, 0.00790714, 0.130594, 0.130594, 0.00790713, },
        { 0.00790713, 0.130602, 0.00790714, 0.130602, 0.00790971, 0.130602, 0.354654, 0, 0.13061, 0.00790714, 0.00790713, 0.130611, },
        { 0.130602, 0.00790714, 0.130602, 0.00790713, 0.130602, 0.00790456, 0, 0.354648, 0.00790713, 0.130594, 0.130593, 0.00790714, },
        { 0.130602, 0.00790713, 0.00790713, 0.130602, 0.130611, 0.00790714, 0.13061, 0.00790713, 0.354677, 0, 0.00790714, 0.130631, },
        { 0.00790712, 0.130602, 0.130602, 0.00790712, 0.00790713, 0.130594, 0.00790714, 0.130594, 0, 0.354624, 0.130574, 0.00790714, },
        { 0.130602, 0.00790713, 0.00790713, 0.130602, 0.00790714, 0.130594, 0.00790713, 0.130593, 0.00790714, 0.130574, 0.354621, 0, },
        { 0.00790712, 0.130602, 0.130602, 0.00790712, 0.130611, 0.00790713, 0.130611, 0.00790714, 0.130631, 0.00790714, 0, 0.354682, },
    };
    
    inline void constructOrthonormalBasis(vec3 n, vec3 *b1, vec3 *b2) {
        float sign = copysign(1.0f, n.z);
        const float a = -1.0f / (sign + n.z);
        const float b = n.x * n.y * a;
        *b1 = vec3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
        *b2 = vec3(b, sign + n.y * n.y * a, -n.y);
    }
    
    void AmbientDice::hybridCubicBezierWeights(u32 triIndex, float b0, float b1, float b2, VertexWeights *w0Out, VertexWeights *w1Out, VertexWeights *w2Out) {
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
    
    void AmbientDice::hybridCubicBezierWeights(vec3 direction, u32 *i0Out, u32 *i1Out, u32 *i2Out, VertexWeights *w0Out, VertexWeights *w1Out, VertexWeights *w2Out) {
        
        u32 triIndex, i0, i1, i2;
        float b0, b1, b2;
        AmbientDice::computeBarycentrics(direction, &triIndex, &i0, &i1, &i2, &b0, &b1, &b2);
        
        AmbientDice::hybridCubicBezierWeights(triIndex, b0, b1, b2, w0Out, w1Out, w2Out);
        
        *i0Out = i0;
        *i1Out = i1;
        *i2Out = i2;
    }
    
    void AmbientDice::srbfWeights(vec3 direction, float *weightsOut) {
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
            AmbientDice::hybridCubicBezierWeights(direction, &i0, &i1, &i2, &weights[0], &weights[1], &weights[2]);
            
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
            AmbientDice::srbfWeights(direction, weights);
            
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
    
    
    template<typename _Matrix_Type_>
    _Matrix_Type_ pseudoInverse( const _Matrix_Type_ &a, double epsilon = std::numeric_limits<double>::epsilon() )
    {
        Eigen::JacobiSVD< _Matrix_Type_ > svd( a, Eigen::ComputeThinU | Eigen::ComputeThinV );
        double tolerance = epsilon * std::max( a.cols(), a.rows() ) *svd.singularValues().array().abs()( 0 );
        return svd.matrixV() *  ( svd.singularValues().array().abs() > tolerance ).select( svd.singularValues().array().inverse(), 0 ).matrix().asDiagonal() * svd.matrixU().adjoint();
    }
    
    Eigen::MatrixXd computeCosineGramMatrixBezier() {
        using namespace Eigen;
        
        const u64 sampleCount = 16384;
        double sampleScale = 4 * M_PI / double(sampleCount);
        
        const u64 brdfSampleCount = 16384;
        double brdfSampleScale = 1.f / double(brdfSampleCount);
        
        MatrixXd gram = MatrixXd::Zero(36, 36);
        
        for (u64 sampleIt = 0; sampleIt < sampleCount; sampleIt += 1) {
            vec3 direction = sampleUniformSphere(sampleHammersley(sampleIt, sampleCount));
            mat3 tangentToWorld = makeOrthogonalBasis(direction);
            
            double bWeights[36] = { 0.f };
            
            u32 i0, i1, i2;
            AmbientDice::VertexWeights weights[3];
            AmbientDice::hybridCubicBezierWeights(direction, &i0, &i1, &i2, &weights[0], &weights[1], &weights[2]);
            
            bWeights[3 * i0 + 0] = weights[0].value;
            bWeights[3 * i0 + 1] = weights[0].directionalDerivativeU;
            bWeights[3 * i0 + 2] = weights[0].directionalDerivativeV;
            
            bWeights[3 * i1 + 0] = weights[1].value;
            bWeights[3 * i1 + 1] = weights[1].directionalDerivativeU;
            bWeights[3 * i1 + 2] = weights[1].directionalDerivativeV;
            
            bWeights[3 * i2 + 0] = weights[2].value;
            bWeights[3 * i2 + 1] = weights[2].directionalDerivativeU;
            bWeights[3 * i2 + 2] = weights[2].directionalDerivativeV;
            
            double aWeights[36] = { 0.f };
            
            for (u64 brdfSampleIt = 0; brdfSampleIt < brdfSampleCount; brdfSampleIt += 1) {
                vec3 brdfTangentDirection = sampleCosineHemisphere(sampleHammersley(brdfSampleIt, brdfSampleCount));
                vec3 brdfWorldDirection = tangentToWorld * brdfTangentDirection;

                AmbientDice::hybridCubicBezierWeights(brdfWorldDirection, &i0, &i1, &i2, &weights[0], &weights[1], &weights[2]);

                aWeights[3 * i0 + 0] += brdfSampleScale * weights[0].value;
                aWeights[3 * i0 + 1] += brdfSampleScale * weights[0].directionalDerivativeU;
                aWeights[3 * i0 + 2] += brdfSampleScale * weights[0].directionalDerivativeV;

                aWeights[3 * i1 + 0] += brdfSampleScale * weights[1].value;
                aWeights[3 * i1 + 1] += brdfSampleScale * weights[1].directionalDerivativeU;
                aWeights[3 * i1 + 2] += brdfSampleScale * weights[1].directionalDerivativeV;

                aWeights[3 * i2 + 0] += brdfSampleScale * weights[2].value;
                aWeights[3 * i2 + 1] += brdfSampleScale * weights[2].directionalDerivativeU;
                aWeights[3 * i2 + 2] += brdfSampleScale * weights[2].directionalDerivativeV;
            }
            
            for (u64 lobeAIt = 0; lobeAIt < 36; ++lobeAIt)
            {
                for (u64 lobeBIt = lobeAIt; lobeBIt < 36; ++lobeBIt)
                {
                    double delta = aWeights[lobeAIt] * bWeights[lobeBIt] * sampleScale;
                    gram(lobeAIt, lobeBIt) += delta;
                    
                    if (lobeBIt != lobeAIt) {
                        gram(lobeBIt, lobeAIt) += delta;
                    }
                }
            }
            
        }
        
        return gram;
    }
    
    Eigen::MatrixXd computeCosineGramMatrixSRBF() {
        using namespace Eigen;
        
        const u64 sampleCount = 16384;
        double sampleScale = 4 * M_PI / double(sampleCount);
        
        const u64 brdfSampleCount = 16384;
        double brdfSampleScale = 1.f / double(brdfSampleCount);
        
        MatrixXd gram = MatrixXd::Zero(12, 12);
        
        for (u64 sampleIt = 0; sampleIt < sampleCount; sampleIt += 1) {
            vec3 direction = sampleUniformSphere(sampleHammersley(sampleIt, sampleCount));
            mat3 tangentToWorld = makeOrthogonalBasis(direction);
            
            float bWeights[12] = { 0.f };
            AmbientDice::srbfWeights(direction, bWeights);
            
            double aWeights[12] = { 0.f };
            
            for (u64 brdfSampleIt = 0; brdfSampleIt < brdfSampleCount; brdfSampleIt += 1) {
                vec3 brdfTangentDirection = sampleCosineHemisphere(sampleHammersley(brdfSampleIt, brdfSampleCount));
                vec3 brdfWorldDirection = tangentToWorld * brdfTangentDirection;
                
                
                float aWeightsLocal[12] = { 0.f };
                
                AmbientDice::srbfWeights(brdfWorldDirection, aWeightsLocal);
                
                for (u64 i = 0; i < 12; i += 1) {
                    aWeights[i] += aWeightsLocal[i] * brdfSampleScale;
                }
            }
            
            for (u64 lobeAIt = 0; lobeAIt < 12; ++lobeAIt)
            {
                for (u64 lobeBIt = lobeAIt; lobeBIt < 12; ++lobeBIt)
                {
                    double delta = aWeights[lobeAIt] * bWeights[lobeBIt] * sampleScale;
                    gram(lobeAIt, lobeBIt) += delta;
                    
                    if (lobeBIt != lobeAIt) {
                        gram(lobeBIt, lobeAIt) += delta;
                    }
                }
            }
            
        }
        
        return gram;
    }
    
    Eigen::MatrixXd computeGGXGramMatrixBezier(float alpha, bool fixedNormal) {
        using namespace Eigen;
        
        const u64 sampleCount = 16384;
        double sampleScale = fixedNormal ? 2 * M_PI / double(sampleCount) : 4 * M_PI / double(sampleCount);
        
        const u64 brdfSampleCount = 16384;
        double brdfSampleScale = 1.f / double(brdfSampleCount);
        
        Microsurface *microsurface = new MicrosurfaceConductor(false, false, alpha, alpha);
        
        MatrixXd gram = MatrixXd::Zero(36, 36);
        
        for (u64 sampleIt = 0; sampleIt < sampleCount; sampleIt += 1) {
            vec2 sample = sampleHammersley(sampleIt, sampleCount);
            vec3 direction = fixedNormal ? sampleUniformHemisphere(sample.x, sample.y) : sampleUniformSphere(sample);
            
            if (fixedNormal) {
                direction = vec3(direction.x, direction.z, direction.y);
            }
            vec3 N = fixedNormal ? vec3(0, 1, 0) : direction;
            mat3 tangentToWorld = makeOrthogonalBasis(N); // local to world
            mat3 worldToTangent = transpose(tangentToWorld);
            
            double bWeights[36] = { 0.f };
            
            u32 i0, i1, i2;
            AmbientDice::VertexWeights weights[3];
            AmbientDice::hybridCubicBezierWeights(direction, &i0, &i1, &i2, &weights[0], &weights[1], &weights[2]);
            
            bWeights[3 * i0 + 0] = weights[0].value;
            bWeights[3 * i0 + 1] = weights[0].directionalDerivativeU;
            bWeights[3 * i0 + 2] = weights[0].directionalDerivativeV;
            
            bWeights[3 * i1 + 0] = weights[1].value;
            bWeights[3 * i1 + 1] = weights[1].directionalDerivativeU;
            bWeights[3 * i1 + 2] = weights[1].directionalDerivativeV;
            
            bWeights[3 * i2 + 0] = weights[2].value;
            bWeights[3 * i2 + 1] = weights[2].directionalDerivativeU;
            bWeights[3 * i2 + 2] = weights[2].directionalDerivativeV;
            
            double aWeights[36] = { 0.f };
            
            vec3 outputDirectionTangent = worldToTangent * direction;
            if (fixedNormal) {
                // Reflect the direction against the normal.
                outputDirectionTangent.x *= -1;
                outputDirectionTangent.y *= -1;
            }
            
            for (u64 brdfSampleIt = 0; brdfSampleIt < brdfSampleCount; brdfSampleIt += 1) {
                vec3 brdfTangentDirection = microsurface->sample(outputDirectionTangent);
                vec3 brdfWorldDirection = tangentToWorld * brdfTangentDirection;
                
                AmbientDice::hybridCubicBezierWeights(brdfWorldDirection, &i0, &i1, &i2, &weights[0], &weights[1], &weights[2]);
                
                aWeights[3 * i0 + 0] += brdfSampleScale * weights[0].value;
                aWeights[3 * i0 + 1] += brdfSampleScale * weights[0].directionalDerivativeU;
                aWeights[3 * i0 + 2] += brdfSampleScale * weights[0].directionalDerivativeV;
                
                aWeights[3 * i1 + 0] += brdfSampleScale * weights[1].value;
                aWeights[3 * i1 + 1] += brdfSampleScale * weights[1].directionalDerivativeU;
                aWeights[3 * i1 + 2] += brdfSampleScale * weights[1].directionalDerivativeV;
                
                aWeights[3 * i2 + 0] += brdfSampleScale * weights[2].value;
                aWeights[3 * i2 + 1] += brdfSampleScale * weights[2].directionalDerivativeU;
                aWeights[3 * i2 + 2] += brdfSampleScale * weights[2].directionalDerivativeV;
            }
            
            for (u64 lobeAIt = 0; lobeAIt < 36; ++lobeAIt)
            {
                for (u64 lobeBIt = lobeAIt; lobeBIt < 36; ++lobeBIt)
                {
                    double delta = aWeights[lobeAIt] * bWeights[lobeBIt] * sampleScale;
                    gram(lobeAIt, lobeBIt) += delta;
                    
                    if (lobeBIt != lobeAIt) {
                        gram(lobeBIt, lobeAIt) += delta;
                    }
                }
            }
            
        }
        
        delete microsurface;
        
        return gram;
    }
    
    Eigen::MatrixXd computeGGXGramMatrixSRBF(float alpha, bool fixedNormal) {
        using namespace Eigen;
        
        const u64 sampleCount = 16384;
        double sampleScale = fixedNormal ? 2 * M_PI / double(sampleCount) : 4 * M_PI / double(sampleCount);
        
        const u64 brdfSampleCount = 16384;
        double brdfSampleScale = 1.f / double(brdfSampleCount);
        
        Microsurface *microsurface = new MicrosurfaceConductor(false, false, alpha, alpha);
        
        MatrixXd gram = MatrixXd::Zero(36, 36);
        
        for (u64 sampleIt = 0; sampleIt < sampleCount; sampleIt += 1) {
            vec2 sample = sampleHammersley(sampleIt, sampleCount);
            vec3 direction = fixedNormal ? sampleUniformHemisphere(sample.x, sample.y) : sampleUniformSphere(sample);
            
            if (fixedNormal) {
                direction = vec3(direction.x, direction.z, direction.y);
            }
            vec3 N = fixedNormal ? vec3(0, 1, 0) : direction;
            mat3 tangentToWorld = makeOrthogonalBasis(N); // local to world
            mat3 worldToTangent = transpose(tangentToWorld);
            
            float bWeights[12] = { 0.f };
            AmbientDice::srbfWeights(direction, bWeights);
            
            double aWeights[12] = { 0.0 };
            
            vec3 outputDirectionTangent = worldToTangent * direction;
            if (fixedNormal) {
                // Reflect the direction against the normal.
                outputDirectionTangent.x *= -1;
                outputDirectionTangent.y *= -1;
            }
            
            for (u64 brdfSampleIt = 0; brdfSampleIt < brdfSampleCount; brdfSampleIt += 1) {
                vec3 brdfTangentDirection = microsurface->sample(outputDirectionTangent);
                vec3 brdfWorldDirection = tangentToWorld * brdfTangentDirection;
                
                float aWeightsLocal[12] = { 0.f };
                AmbientDice::srbfWeights(brdfWorldDirection, aWeightsLocal);
                
                for (u64 i = 0; i < 12; i += 1) {
                    aWeights[i] += aWeightsLocal[i] * brdfSampleScale;
                }
            }
            
            for (u64 lobeAIt = 0; lobeAIt < 12; ++lobeAIt)
            {
                for (u64 lobeBIt = lobeAIt; lobeBIt < 12; ++lobeBIt)
                {
                    double delta = aWeights[lobeAIt] * bWeights[lobeBIt] * sampleScale;
                    gram(lobeAIt, lobeBIt) += delta;
                    
                    if (lobeBIt != lobeAIt) {
                        gram(lobeBIt, lobeAIt) += delta;
                    }
                }
            }
        }
        
        delete microsurface;
        
        return gram;
    }

    AmbientDice ambientDiceConvertRadianceToSpecularBezier(const AmbientDice &ambientDiceRadiance) {
        using namespace Eigen;
        
        MatrixXd resultMatrix = pseudoInverse(AmbientDice::computeGramMatrixBezier()) * computeGGXGramMatrixBezier(ggxAlpha, specularFixedNormal);
        
        AmbientDice specular = { };
        
        for (u64 vert = 0; vert < 12; vert += 1) {
            for (u64 otherVert = 0; otherVert < 12; otherVert += 1) {
                specular.vertices[vert].value += ambientDiceRadiance.vertices[otherVert].value * (float)resultMatrix(3 * vert, 3 * otherVert);
                specular.vertices[vert].value += ambientDiceRadiance.vertices[otherVert].directionalDerivativeU * (float)resultMatrix(3 * vert, 3 * otherVert + 1);
                specular.vertices[vert].value += ambientDiceRadiance.vertices[otherVert].directionalDerivativeV * (float)resultMatrix(3 * vert, 3 * otherVert + 2);
                
                specular.vertices[vert].directionalDerivativeU += ambientDiceRadiance.vertices[otherVert].value * (float)resultMatrix(3 * vert + 1, 3 * otherVert);
                specular.vertices[vert].directionalDerivativeU += ambientDiceRadiance.vertices[otherVert].directionalDerivativeU * (float)resultMatrix(3 * vert + 1, 3 * otherVert + 1);
                specular.vertices[vert].directionalDerivativeU += ambientDiceRadiance.vertices[otherVert].directionalDerivativeV * (float)resultMatrix(3 * vert + 1, 3 * otherVert + 2);
                
                specular.vertices[vert].directionalDerivativeV += ambientDiceRadiance.vertices[otherVert].value * (float)resultMatrix(3 * vert + 2, 3 * otherVert);
                specular.vertices[vert].directionalDerivativeV += ambientDiceRadiance.vertices[otherVert].directionalDerivativeU * (float)resultMatrix(3 * vert + 2, 3 * otherVert + 1);
                specular.vertices[vert].directionalDerivativeV += ambientDiceRadiance.vertices[otherVert].directionalDerivativeV * (float)resultMatrix(3 * vert + 2, 3 * otherVert + 2);
            }
        }
        
        return specular;
    }
    
    AmbientDice ambientDiceConvertRadianceToSpecularSRBF(const AmbientDice &ambientDiceRadiance) {
        using namespace Eigen;
        
        MatrixXd resultMatrix = pseudoInverse(AmbientDice::computeGramMatrixSRBF()) * computeGGXGramMatrixSRBF(ggxAlpha, specularFixedNormal);
        
        std::cout << "Radiance to irradiance matrix: " << std::endl;
        std::cout << resultMatrix << std::endl;
        
        AmbientDice irradiance = { };
        
        for (u64 vert = 0; vert < 12; vert += 1) {
            for (u64 otherVert = 0; otherVert < 12; otherVert += 1) {
                irradiance.vertices[vert].value += ambientDiceRadiance.vertices[otherVert].value * (float)resultMatrix(vert, otherVert);
            }
        }
        
        return irradiance;
    }
    
    AmbientDice ambientDiceConvertRadianceToIrradianceBezier(const AmbientDice &ambientDiceRadiance) {
        using namespace Eigen;
        
        MatrixXd resultMatrix = pseudoInverse(AmbientDice::computeGramMatrixBezier()) * computeCosineGramMatrixBezier();
        
        std::cout << "Radiance to irradiance matrix: " << std::endl;
        std::cout << resultMatrix << std::endl;
        
        AmbientDice irradiance = { };
        
        for (u64 vert = 0; vert < 12; vert += 1) {
            for (u64 otherVert = 0; otherVert < 12; otherVert += 1) {
                irradiance.vertices[vert].value += ambientDiceRadiance.vertices[otherVert].value * (float)resultMatrix(3 * vert, 3 * otherVert);
                irradiance.vertices[vert].value += ambientDiceRadiance.vertices[otherVert].directionalDerivativeU * (float)resultMatrix(3 * vert, 3 * otherVert + 1);
                irradiance.vertices[vert].value += ambientDiceRadiance.vertices[otherVert].directionalDerivativeV * (float)resultMatrix(3 * vert, 3 * otherVert + 2);
                
                irradiance.vertices[vert].directionalDerivativeU += ambientDiceRadiance.vertices[otherVert].value * (float)resultMatrix(3 * vert + 1, 3 * otherVert);
                irradiance.vertices[vert].directionalDerivativeU += ambientDiceRadiance.vertices[otherVert].directionalDerivativeU * (float)resultMatrix(3 * vert + 1, 3 * otherVert + 1);
                irradiance.vertices[vert].directionalDerivativeU += ambientDiceRadiance.vertices[otherVert].directionalDerivativeV * (float)resultMatrix(3 * vert + 1, 3 * otherVert + 2);
                
                irradiance.vertices[vert].directionalDerivativeV += ambientDiceRadiance.vertices[otherVert].value * (float)resultMatrix(3 * vert + 2, 3 * otherVert);
                irradiance.vertices[vert].directionalDerivativeV += ambientDiceRadiance.vertices[otherVert].directionalDerivativeU * (float)resultMatrix(3 * vert + 2, 3 * otherVert + 1);
                irradiance.vertices[vert].directionalDerivativeV += ambientDiceRadiance.vertices[otherVert].directionalDerivativeV * (float)resultMatrix(3 * vert + 2, 3 * otherVert + 2);
            }
        }
        
        return irradiance;
    }
    
    AmbientDice ambientDiceConvertRadianceToIrradianceSRBF(const AmbientDice &ambientDiceRadiance) {
        using namespace Eigen;
        
        MatrixXd resultMatrix = pseudoInverse(AmbientDice::computeGramMatrixSRBF()) * computeCosineGramMatrixSRBF();
        
        std::cout << "Radiance to irradiance matrix: " << std::endl;
        std::cout << resultMatrix << std::endl;
        
        AmbientDice irradiance = { };
        
        for (u64 vert = 0; vert < 12; vert += 1) {
            for (u64 otherVert = 0; otherVert < 12; otherVert += 1) {
                irradiance.vertices[vert].value += ambientDiceRadiance.vertices[otherVert].value * (float)resultMatrix(vert, otherVert);
            }
        }
        
        return irradiance;
    }
    
    
    //    AmbientDice ExperimentAmbientDice::solveAmbientDiceLeastSquares(ImageBase<vec3>& directions, const Image& irradiance)
    //    {
    //        using namespace Eigen;
    //
    //        AmbientDice ambientDice;
    //
    //        const u64 sampleCount = directions.getPixelCount();
    //
    //        MatrixXf A = MatrixXf::Zero(sampleCount, 36);
    //
    //        for (u64 sampleIt = 0; sampleIt < sampleCount; ++sampleIt)
    //        {
    //            const vec3& direction = directions.at(sampleIt);
    //
    //            u32 i0, i1, i2;
    //            AmbientDice::VertexWeights weights[3];
    //            AmbientDice::hybridCubicBezierWeights(direction, &i0, &i1, &i2, &weights[0], &weights[1], &weights[2]);
    //
    //            A(sampleIt, 3 * i0) = weights[0].value;
    //            A(sampleIt, 3 * i0 + 1) = weights[0].directionalDerivativeU;
    //            A(sampleIt, 3 * i0 + 2) = weights[0].directionalDerivativeV;
    //
    //            A(sampleIt, 3 * i1) = weights[1].value;
    //            A(sampleIt, 3 * i1 + 1) = weights[1].directionalDerivativeU;
    //            A(sampleIt, 3 * i1 + 2) = weights[1].directionalDerivativeV;
    //
    //            A(sampleIt, 3 * i2) = weights[2].value;
    //            A(sampleIt, 3 * i2 + 1) = weights[2].directionalDerivativeU;
    //            A(sampleIt, 3 * i2 + 2) = weights[2].directionalDerivativeV;
    //
    //        }
    //
    //        auto solver = A.jacobiSvd(ComputeThinU | ComputeThinV);
    //
    //        VectorXf b;
    //        b.resize(sampleCount);
    //
    //        for (u32 channelIt = 0; channelIt < 3; ++channelIt)
    //        {
    //            for (u64 sampleIt = 0; sampleIt < sampleCount; ++sampleIt)
    //            {
    //                b[sampleIt] = irradiance.at(sampleIt)[channelIt];
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
    
    Eigen::MatrixXd AmbientDice::computeGramMatrixBezier() {
        using namespace Eigen;
        
        const u64 sampleCount = 32768;
        double sampleScale = 4 * M_PI / double(sampleCount);
        
        AmbientDice ambientDice;
        
        MatrixXd gram = MatrixXd::Zero(36, 36);
        
        for (u64 sampleIt = 0; sampleIt < sampleCount; sampleIt += 1) {
            vec3 direction = sampleUniformSphere(sampleHammersley(sampleIt, sampleCount));
            
            float allWeights[36] = { 0.f };
            
            u32 i0, i1, i2;
            AmbientDice::VertexWeights weights[3];
            AmbientDice::hybridCubicBezierWeights(direction, &i0, &i1, &i2, &weights[0], &weights[1], &weights[2]);
            
            allWeights[3 * i0 + 0] = weights[0].value;
            allWeights[3 * i0 + 1] = weights[0].directionalDerivativeU;
            allWeights[3 * i0 + 2] = weights[0].directionalDerivativeV;
            
            allWeights[3 * i1 + 0] = weights[1].value;
            allWeights[3 * i1 + 1] = weights[1].directionalDerivativeU;
            allWeights[3 * i1 + 2] = weights[1].directionalDerivativeV;
            
            allWeights[3 * i2 + 0] = weights[2].value;
            allWeights[3 * i2 + 1] = weights[2].directionalDerivativeU;
            allWeights[3 * i2 + 2] = weights[2].directionalDerivativeV;
            
            for (u64 lobeAIt = 0; lobeAIt < 36; ++lobeAIt)
            {
                for (u64 lobeBIt = lobeAIt; lobeBIt < 36; ++lobeBIt)
                {
                    double delta = allWeights[lobeAIt] * allWeights[lobeBIt] * sampleScale;
                    gram(lobeAIt, lobeBIt) += delta;
                    
                    if (lobeBIt != lobeAIt) {
                        gram(lobeBIt, lobeAIt) += delta;
                    }
                }
            }
        }
        
        return gram;
    }
    
    Eigen::MatrixXd AmbientDice::computeGramMatrixSRBF() {
        using namespace Eigen;
        
        const u64 sampleCount = 32768;
        double sampleScale = 4 * M_PI / double(sampleCount);
        
        AmbientDice ambientDice;
        
        MatrixXd gram = MatrixXd::Zero(12, 12);
        
        for (u64 sampleIt = 0; sampleIt < sampleCount; sampleIt += 1) {
            vec3 direction = sampleUniformSphere(sampleHammersley(sampleIt, sampleCount));
            
            float allWeights[12] = { 0.f };
            AmbientDice::srbfWeights(direction, allWeights);
            
            for (u64 lobeAIt = 0; lobeAIt < 12; ++lobeAIt)
            {
                for (u64 lobeBIt = lobeAIt; lobeBIt < 12; ++lobeBIt)
                {
                    double delta = allWeights[lobeAIt] * allWeights[lobeBIt] * sampleScale;
                    gram(lobeAIt, lobeBIt) += delta;
                    
                    if (lobeBIt != lobeAIt) {
                        gram(lobeBIt, lobeAIt) += delta;
                    }
                }
            }
        }
        
        return gram;
    }
    
    AmbientDice ExperimentAmbientDice::solveAmbientDiceLeastSquares(ImageBase<vec3>& directions, const Image& irradiance)
    {
        using namespace Eigen;
        
        AmbientDice ambientDice;
        
        MatrixXf moments = MatrixXf::Zero(36, 3);
        
        const ivec2 imageSize = directions.getSize();
        directions.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
                               {
                                   float texelArea = latLongTexelArea(pixelPos, imageSize);
                                   //                                              float texelArea = 1.f / directions.getPixelCount();
                                   
                                   const vec4& colour = irradiance.at(pixelPos);
                                   
                                   u32 i0, i1, i2;
                                   AmbientDice::VertexWeights weights[3];
                                   AmbientDice::hybridCubicBezierWeights(direction, &i0, &i1, &i2, &weights[0], &weights[1], &weights[2]);
                                   
                                   moments(3 * i0 + 0, 0) += weights[0].value * colour.r * texelArea;
                                   moments(3 * i0 + 1, 0) += weights[0].directionalDerivativeU * colour.r * texelArea;
                                   moments(3 * i0 + 2, 0) += weights[0].directionalDerivativeV * colour.r * texelArea;
                                   moments(3 * i1 + 0, 0) += weights[1].value * colour.r * texelArea;
                                   moments(3 * i1 + 1, 0) += weights[1].directionalDerivativeU * colour.r * texelArea;
                                   moments(3 * i1 + 2, 0) += weights[1].directionalDerivativeV * colour.r * texelArea;
                                   moments(3 * i2 + 0, 0) += weights[2].value * colour.r * texelArea;
                                   moments(3 * i2 + 1, 0) += weights[2].directionalDerivativeU * colour.r * texelArea;
                                   moments(3 * i2 + 2, 0) += weights[2].directionalDerivativeV * colour.r * texelArea;
                                   moments(3 * i0 + 0, 1) += weights[0].value * colour.g * texelArea;
                                   moments(3 * i0 + 1, 1) += weights[0].directionalDerivativeU * colour.g * texelArea;
                                   moments(3 * i0 + 2, 1) += weights[0].directionalDerivativeV * colour.g * texelArea;
                                   moments(3 * i1 + 0, 1) += weights[1].value * colour.g * texelArea;
                                   moments(3 * i1 + 1, 1) += weights[1].directionalDerivativeU * colour.g * texelArea;
                                   moments(3 * i1 + 2, 1) += weights[1].directionalDerivativeV * colour.g * texelArea;
                                   moments(3 * i2 + 0, 1) += weights[2].value * colour.g * texelArea;
                                   moments(3 * i2 + 1, 1) += weights[2].directionalDerivativeU * colour.g * texelArea;
                                   moments(3 * i2 + 2, 1) += weights[2].directionalDerivativeV * colour.g * texelArea;
                                   moments(3 * i0 + 0, 2) += weights[0].value * colour.b * texelArea;
                                   moments(3 * i0 + 1, 2) += weights[0].directionalDerivativeU * colour.b * texelArea;
                                   moments(3 * i0 + 2, 2) += weights[0].directionalDerivativeV * colour.b * texelArea;
                                   moments(3 * i1 + 0, 2) += weights[1].value * colour.b * texelArea;
                                   moments(3 * i1 + 1, 2) += weights[1].directionalDerivativeU * colour.b * texelArea;
                                   moments(3 * i1 + 2, 2) += weights[1].directionalDerivativeV * colour.b * texelArea;
                                   moments(3 * i2 + 0, 2) += weights[2].value * colour.b * texelArea;
                                   moments(3 * i2 + 1, 2) += weights[2].directionalDerivativeU * colour.b * texelArea;
                                   moments(3 * i2 + 2, 2) += weights[2].directionalDerivativeV * colour.b * texelArea;
                               });
        
        MatrixXf gram = AmbientDice::computeGramMatrixBezier().cast<float>();
        
        auto solver = gram.jacobiSvd(ComputeThinU | ComputeThinV);
        
        VectorXf b;
        b.resize(36);
        
        for (u32 channelIt = 0; channelIt < 3; ++channelIt)
        {
            for (u64 lobeIt = 0; lobeIt < 36; ++lobeIt)
            {
                b[lobeIt] = moments(lobeIt, channelIt);
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
            AmbientDice::srbfWeights(direction, weights);
            
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
    
    void ExperimentAmbientDice::run(SharedData& data)
    {
        
        using namespace Eigen;
        
//        MatrixXd bezierGram = AmbientDice::computeGramMatrixBezier();
//        MatrixXd srbfGram = AmbientDice::computeGramMatrixSRBF();
//
//        std::cout << "const float ambientDiceGramBezier[36][36] = {\n";
//        for (u64 i = 0; i < 36; i += 1) {
//            std::cout << "    { ";
//
//            for (u64 j = 0; j < 36; j += 1) {
//                std::cout << bezierGram(i, j) << ", ";
//            }
//            std::cout << "},\n";
//        }
//        std::cout << "}\n";
//
//        std::cout << "const float ambientDiceGramSRBF[12][12] = {\n";
//        for (u64 i = 0; i < 12; i += 1) {
//            std::cout << "    { ";
//
//            for (u64 j = 0; j < 12; j += 1) {
//                std::cout << srbfGram(i, j) << ", ";
//            }
//            std::cout << "},\n";
//        }
//        std::cout << "}\n";

        
        MatrixXd inverseGram = pseudoInverse(AmbientDice::computeGramMatrixBezier());
        for (u64 i = 1; i <= 20; i += 1) {
            double sqrtAlpha = double(i) / 20.0;
            double alpha = sqrtAlpha * sqrtAlpha;
            
            MatrixXd resultMatrix = inverseGram * computeGGXGramMatrixBezier(alpha, true);
            
            std::cout << "let adBezierSpecularAlpha" << alpha << " : [[Double]] = [\n";
            for (u64 i = 0; i < 36; i += 1) {
                std::cout << "    [ ";
    
                for (u64 j = 0; j < 36; j += 1) {
                    std::cout << resultMatrix(i, j) << ", ";
                }
                std::cout << "],\n";
            }
            std::cout << "]\n";
        }
        
        
        m_radianceImage = Image(data.m_outputSize);
        m_specularImage = Image(data.m_outputSize);
        m_irradianceImage = Image(data.m_outputSize);
        
        //        compareResponseImages(data);
        
        if (m_diceType == AmbientDiceTypeBezier) {
            AmbientDice ambientDiceRadiance = solveAmbientDiceLeastSquares(data.m_directionImage, m_input->m_radianceImage);
            
//            AmbientDice ambientDiceSpecular = solveAmbientDiceLeastSquares(data.m_directionImage, m_input->m_specularImage);
            AmbientDice ambientDiceSpecular = ambientDiceConvertRadianceToSpecularBezier(ambientDiceRadiance);
            
            //            AmbientDice ambientDiceIrradiance = solveAmbientDiceLeastSquares(data.m_directionImage, m_input->m_irradianceImage);
            AmbientDice ambientDiceIrradiance = ambientDiceConvertRadianceToIrradianceBezier(ambientDiceRadiance);
            
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
            
//            AmbientDice ambientDiceSpecular = solveAmbientDiceLeastSquaresSRBF(data.m_directionImage, m_input->m_specularImage);
            AmbientDice ambientDiceSpecular = ambientDiceConvertRadianceToSpecularSRBF(ambientDiceRadiance);
            
//            AmbientDice ambientDiceIrradiance = solveAmbientDiceLeastSquaresSRBF(data.m_directionImage, m_input->m_irradianceImage);
            AmbientDice ambientDiceIrradiance = ambientDiceConvertRadianceToIrradianceSRBF(ambientDiceRadiance);
            
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
