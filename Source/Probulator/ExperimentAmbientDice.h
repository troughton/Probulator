#pragma once

#include <Probulator/Experiments.h>

namespace Probulator {
    
    struct AmbientDice
    {
        static const float kT;
        static const float kT2;
        
        static const vec3 vertexPositions[12];
        static const vec3 tangents[12];
        static const vec3 bitangents[12];
        
        struct Vertex {
            vec3 value;
            vec3 directionalDerivativeU;
            vec3 directionalDerivativeV;
        };
        
        struct VertexWeights {
            float value;
            float directionalDerivativeU;
            float directionalDerivativeV;
        };
        
        Vertex vertices[12];
        
        inline void indexIcosahedron(const vec3& direction, u32 *i0, u32 *i1, u32 *i2) const
        {
            float kT = 0.618034f;
            float kT2 = kT * kT;
            
            ivec3 octantBit = ivec3(direction.x < 0 ? 1 : 0,
                                    direction.y < 0 ? 1 : 0,
                                    direction.z < 0 ? 1 : 0);
            
            ivec3 octantBitFlipped = ivec3(1) - octantBit;
            
            // Vertex indices
            u32 indexA = octantBit.y * 2 + octantBit.x + 0;
            u32 indexB = octantBit.z * 2 + octantBit.y + 4;
            u32 indexC = octantBit.z * 2 + octantBit.x + 8;
            
            u32 indexAFlipped = octantBit.z * 2 + octantBitFlipped.x + 8;
            u32 indexBFlipped = octantBitFlipped.y * 2 + octantBit.x + 0;
            u32 indexCFlipped = octantBitFlipped.y * 2 + octantBit.y + 4;
            
            // Selection
            bool vertASelect = dot(abs(direction), vec3(1.0, kT2, -kT)) > 0.0;
            bool vertBSelect = dot(abs(direction), vec3(-kT, 1.0, kT2)) > 0.0;
            bool vertCSelect = dot(abs(direction), vec3(kT2, -kT, 1.0)) > 0.0;
            
            *i0 = vertASelect ? indexA : indexAFlipped;
            *i1 = vertBSelect ? indexB : indexBFlipped;
            *i2 = vertCSelect ? indexC : indexCFlipped;
            
            if (*i1 == *i0) {
                *i1 = vertBSelect ? indexBFlipped : indexB;
            }
            
            if (*i2 == *i1) {
                *i2 = vertCSelect ? indexCFlipped : indexC;
            }
        }
        
        inline u32 indexIcosahedronTriangle(const vec3& direction) const
        {
            float kT = 0.618034f;
            float kT2 = kT * kT;
            
            ivec3 octantBit = ivec3(direction.x < 0 ? 1 : 0,
                                    direction.y < 0 ? 1 : 0,
                                    direction.z < 0 ? 1 : 0);
            
            u32 t = octantBit.x + octantBit.y * 2 + octantBit.z * 4;
            u32 tRed = 8 + octantBit.y + octantBit.z * 2;
            u32 tGreen = 12 + octantBit.x + octantBit.z * 2;
            u32 tBlue = 16 + octantBit.x + octantBit.y * 2;
            
            // Selection
            bool vertASelect = dot(abs(direction), vec3(1.0, kT2, -kT)) > 0.0;
            bool vertBSelect = dot(abs(direction), vec3(-kT, 1.0, kT2)) > 0.0;
            bool vertCSelect = dot(abs(direction), vec3(kT2, -kT, 1.0)) > 0.0;
            
            t = vertASelect ? t : tRed;
            t = vertBSelect ? t : tGreen;
            t = vertCSelect ? t : tBlue;
            
            return t;
        }
        
        vec3 hybridCubicBezier(u32 i0, u32 i1, u32 i2, float b0, float b1, float b2) const;
        void hybridCubicBezierWeights(u32 i0, u32 i1, u32 i2, float b0, float b1, float b2, VertexWeights *w0, VertexWeights *w1, VertexWeights *w2) const;
        
        inline vec3 evaluateLinear(const vec3& direction) const
        {
            u32 i0, i1, i2;
            this->indexIcosahedron(direction, &i0, &i1, &i2);
            
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
            
            return b0 * this->vertices[i0].value + b1 * this->vertices[i1].value + b2 * this->vertices[i2].value;
        }
        
        inline vec3 evaluateBezier(const vec3& direction) const
        {
            u32 i0, i1, i2;
            this->indexIcosahedron(direction, &i0, &i1, &i2);
            
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
            
            AmbientDice::VertexWeights weights[3];
            this->hybridCubicBezierWeights(i0, i1, i2, b0, b1, b2, &weights[0], &weights[1], &weights[2]);
            
            return
            weights[0].value * this->vertices[i0].value +
            weights[0].directionalDerivativeU * this->vertices[i0].directionalDerivativeU +
            weights[0].directionalDerivativeV * this->vertices[i0].directionalDerivativeV +
            weights[1].value * this->vertices[i1].value +
            weights[1].directionalDerivativeU * this->vertices[i1].directionalDerivativeU +
            weights[1].directionalDerivativeV * this->vertices[i1].directionalDerivativeV +
            weights[2].value * this->vertices[i2].value +
            weights[2].directionalDerivativeU * this->vertices[i2].directionalDerivativeU +
            weights[2].directionalDerivativeV * this->vertices[i2].directionalDerivativeV;
        }
    };

    
    class ExperimentAmbientDice : public Experiment
    {
        public:
        
        AmbientDice solveAmbientDiceRunningAverage(const ImageBase<vec3>& directions, const Image& irradiance);
        AmbientDice solveAmbientDiceRunningAverageBezier(const ImageBase<vec3>& directions, const Image& irradiance);
        
        void run(SharedData& data) override;
    };
    
}
