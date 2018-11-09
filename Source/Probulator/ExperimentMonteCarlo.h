#pragma once

#include <Probulator/Experiments.h>
#include "MicrosurfaceScattering.h"

namespace Probulator {
    
    class ExperimentMC : public Experiment
    {
    public:
        
        void run(SharedData& data) override
        {
            m_radianceImage = data.m_radianceImage;
            
            m_irradianceImage = Image(data.m_outputSize);
            data.m_directionImage.parallelForPixels2D([&](const vec3& direction, ivec2 pixelPos)
                                                      {
                                                          mat3 basis = makeOrthogonalBasis(direction);
                                                          vec3 accum = vec3(0.0f);
                                                          for (u32 sampleIt = 0; sampleIt < m_hemisphereSampleCount; ++sampleIt)
                                                          {
                                                              vec2 sampleUv = sampleHammersley(sampleIt, m_hemisphereSampleCount);
                                                              vec3 hemisphereDirection = sampleCosineHemisphere(sampleUv);
                                                              vec3 sampleDirection = basis * hemisphereDirection;
                                                              accum += (vec3)m_radianceImage.sampleNearest(cartesianToLatLongTexcoord(sampleDirection));
                                                          }
                                                          
                                                          accum /= m_hemisphereSampleCount;
                                                          
                                                          m_irradianceImage.at(pixelPos) = vec4(accum, 1.0f);
                                                      });
            
            
            m_specularImage = Image(data.m_outputSize);
            data.m_directionImage.parallelForPixels2D([&](const vec3& V, ivec2 pixelPos)
                                                      {
                                                          mat3 basis = makeOrthogonalBasis(V);
                                                          vec3 accum = vec3(0.0f);
                                                          for (u32 sampleIt = 0; sampleIt < m_hemisphereSampleCount; ++sampleIt)
                                                          {
                                                              vec2 sampleUv = sampleHammersley(sampleIt, m_hemisphereSampleCount);
                                                              
                                                              vec3 f0 = vec3(1.0f);
                                                              float f90 = 1.0;
                                                              
                                                              
                                                              //Specular
                                                              vec3 H = sampleGGXVNDF(V, ggxAlpha, ggxAlpha, sampleUv.x, sampleUv.y);
                                                              vec3 outgoingAngle = reflect(-V, H);
                                                              
                                                              float VdotH = saturate(dot(V, H)); // same as LdotH for reflection.
                                                              
                                                              //Specular
                                                              float Vis = SmithGGXMaskingShadowingG2OverG1Reflection(V, outgoingAngle, H, ggxAlpha);
                                                              vec3 F = F_Schlick(f0, f90, VdotH);
                                                              
                                                              vec3 sampleWeight = F * Vis;
                                                              
                                                              vec3 sampleDirection = basis * outgoingAngle;
                                                              accum += sampleWeight * (vec3)m_radianceImage.sampleNearest(cartesianToLatLongTexcoord(sampleDirection));
                                                          }
                                                          
                                                          accum /= m_hemisphereSampleCount;
                                                          
                                                          m_specularImage.at(pixelPos) = vec4(accum, 1.0f);
                                                      });
        }
        
        void getProperties(std::vector<Property>& outProperties) override
        {
            Experiment::getProperties(outProperties);
            outProperties.push_back(Property("Hemisphere sample count", reinterpret_cast<int*>(&m_hemisphereSampleCount)));
        }
        
        ExperimentMC& setHemisphereSampleCount(u32 v) { m_hemisphereSampleCount = v; return *this; }
        
        u32 m_hemisphereSampleCount = 1000;
    };
    
    class ExperimentMCIS : public Experiment
    {
    public:
        
        void run(SharedData& data) override
        {
            const ivec2 imageSize = data.m_outputSize;
            
            m_radianceImage = data.m_radianceImage;
            
            std::vector<float> texelWeights;
            std::vector<float> texelAreas;
            float weightSum = 0.0;
            m_radianceImage.forPixels2D([&](const vec4& p, ivec2 pixelPos)
                                        {
                                            float area = latLongTexelArea(pixelPos, imageSize);
                                            
                                            float intensity = rgbLuminance((vec3)p);
                                            float weight = intensity * area;
                                            
                                            weightSum += weight;
                                            texelWeights.push_back(weight);
                                            texelAreas.push_back(area);
                                        });
            
            DiscreteDistribution<float> discreteDistribution(texelWeights.data(), texelWeights.size(), weightSum);
            
            m_irradianceImage = Image(data.m_outputSize);
            data.m_directionImage.parallelForPixels2D([&](const vec3& normal, ivec2 pixelPos)
                                                      {
                                                          u32 pixelIndex = pixelPos.x + pixelPos.y * m_irradianceImage.getWidth();
                                                          u32 seed = m_jitterEnabled ? pixelIndex : 0;
                                                          std::mt19937 rng(seed);
                                                          
                                                          vec3 accum = vec3(0.0f);
                                                          for (u32 sampleIt = 0; sampleIt < m_sampleCount; ++sampleIt)
                                                          {
                                                              u32 sampleIndex = (u32)discreteDistribution(rng);
                                                              float sampleProbability = texelWeights[sampleIndex] / weightSum;
                                                              vec3 sampleDirection = data.m_directionImage.at(sampleIndex);
                                                              float cosTerm = dotMax0(normal, sampleDirection);
                                                              float sampleArea = (float)texelAreas[sampleIndex];
                                                              vec3 sampleRadiance = (vec3)m_radianceImage.at(sampleIndex) * sampleArea;
                                                              accum += sampleRadiance * cosTerm / sampleProbability;
                                                          }
                                                          
                                                          accum /= m_sampleCount * pi;
                                                          
                                                          m_irradianceImage.at(pixelPos) = vec4(accum, 1.0f);
                                                      });
            
            
            const u64 m_hemisphereSampleCount = 1024;
            
            Microsurface *microsurface = new MicrosurfaceConductor(false, false, ggxAlpha, ggxAlpha);
            
            m_specularImage = Image(data.m_outputSize);
            data.m_directionImage.parallelForPixels2D([&](const vec3& sampleDirection, ivec2 pixelPos)
                                                      {
                                                          if (specularFixedNormal) {
                                                              vec3 N = vec3(0, 1, 0);
                                                              mat3 basis = makeOrthogonalBasis(N);
                                                              
                                                              vec3 V = reflect(-sampleDirection, N);
                                                              
                                                              V = transpose(basis) * V;
                                                              
                                                              if (V.z < 0.f) {
                                                                  m_specularImage.at(pixelPos) = vec4(0, 0, 0, 1);
                                                                  return;
                                                              }
                                                              
                                                              vec3 accum = vec3(0.0f);
                                                              for (u32 sampleIt = 0; sampleIt < m_hemisphereSampleCount; ++sampleIt)
                                                              {
                                                                  vec2 sampleUv = sampleHammersley(sampleIt, m_hemisphereSampleCount);
                                                                  
                                                                  vec3 f0 = vec3(1.0f);
                                                                  float f90 = 1.0;
                                                                  
                                                                  //Specular
                                                                  vec3 H =  sampleGGXVNDF(V, ggxAlpha, ggxAlpha, sampleUv.x, sampleUv.y);
                                                                  vec3 outgoingAngle = reflect(-V, H);
                                                                  
                                                                  float VdotH = saturate(dot(V, H)); // same as LdotH for reflection.
                                                                  
                                                                  //Specular
                                                                  float Vis = SmithGGXMaskingShadowingG2OverG1Reflection(V, outgoingAngle, H, ggxAlpha);
                                                                  vec3 F = F_Schlick(f0, f90, VdotH);
                                                                  
                                                                  vec3 sampleWeight = F * Vis;
                                                                  
                                                                  vec3 sampleDirection = basis * microsurface->sample(V); // outgoingAngle;
                                                                  accum += /* sampleWeight * */ (vec3)m_radianceImage.sampleBilinear(cartesianToLatLongTexcoord(normalize(sampleDirection)));
                                                              }
                                                              
                                                              accum /= m_hemisphereSampleCount;
                                                              
                                                              m_specularImage.at(pixelPos) = vec4(accum, 1.0f);
                                                          } else {
                                                              mat3 basis = makeOrthogonalBasis(sampleDirection);
                                                              vec3 accum = vec3(0.0f);
                                                              for (u32 sampleIt = 0; sampleIt < m_hemisphereSampleCount; ++sampleIt)
                                                              {
                                                                  vec2 sampleUv = sampleHammersley(sampleIt, m_hemisphereSampleCount);
                                                                  
                                                                  vec3 f0 = vec3(1.0f);
                                                                  float f90 = 1.0;
                                                                  
                                                                  const vec3 V = vec3(0, 0, 1);
                                                                  
                                                                  //Specular
                                                                  vec3 H = sampleGGXVNDF(V, ggxAlpha, ggxAlpha, sampleUv.x, sampleUv.y);
                                                                  vec3 outgoingAngle = reflect(-V, H);
                                                                  
                                                                  float VdotH = saturate(dot(V, H)); // same as LdotH for reflection.
                                                                  
                                                                  //Specular
                                                                  float Vis = SmithGGXMaskingShadowingG2OverG1Reflection(V, outgoingAngle, H, ggxAlpha);
                                                                  vec3 F = F_Schlick(f0, f90, VdotH);
                                                                  
                                                                  vec3 sampleWeight = F * Vis;
                                                                  
                                                                  vec3 sampleDirection = basis * microsurface->sample(V); //outgoingAngle;
                                                                  accum += /* sampleWeight * */ (vec3)m_radianceImage.sampleBilinear(cartesianToLatLongTexcoord(normalize(sampleDirection)));
                                                              }
                                                              
                                                              accum /= m_hemisphereSampleCount;
                                                              
                                                              m_specularImage.at(pixelPos) = vec4(accum, 1.0f);
                                                          }
                                                      });
            
//            delete microsurface;
            
            data.GenerateIrradianceSamples(m_irradianceImage);
        }
        
        void getProperties(std::vector<Property>& outProperties) override
        {
            Experiment::getProperties(outProperties);
            outProperties.push_back(Property("Sample count", reinterpret_cast<int*>(&m_sampleCount)));
            outProperties.push_back(Property("Jitter enabled", &m_jitterEnabled));
        }
        
        ExperimentMCIS& setSampleCount(u32 v)
        {
            m_sampleCount = v;
            return *this;
        }
        
        ExperimentMCIS& setJitterEnabled(bool state)
        {
            m_jitterEnabled = state;
            return *this;
        }
        
        u32 m_sampleCount = 1000;
        bool m_jitterEnabled = false;
    };
    
}
