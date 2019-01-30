#include "Image.h"

#include <stb_image_write.h>
#include <stb_image.h>
#include <stb_image_resize.h>

#include "emd_hat_signatures_interface.hpp"

namespace Probulator
{
    inline vec3 rgbToXYZ(vec3 rgb) {
        float x = 0.490f * rgb.r + 0.310f * rgb.g + 0.2f * rgb.b;
        float y = 0.17697f * rgb.r + 0.8124f * rgb.g + 0.01063f * rgb.b;
        float z = 0.01f * rgb.g + 0.99f * rgb.b;
        return vec3(x, y, z);
    }
    
    inline float xyzF(float t) {
        if (t > 0.008856451679f) {
            return pow(t, 1.f / 3.f);
        } else {
            return 7.787037037f * t + 4.f / 29.f;
        }
    }
    
    inline vec3 xyzToLab(vec3 xyz) {
        const float Xn = 95.047f;
        const float Yn = 100.f;
        const float Zn = 108.883f;
        
        float L = 116.f * xyzF(xyz.y / Yn) - 16.f;
        float a = 500.f * (xyzF(xyz.x / Xn) - xyzF(xyz.y / Yn));
        float b = 200.f * (xyzF(xyz.y / Yn) - xyzF(xyz.z / Zn));
        
        return vec3(L, a, b);
    }
    
	void Image::writePng(const char* filename) const
	{
		if (m_pixels.empty()) return;
		std::vector<u32> imageLdr(m_size.x * m_size.y);
		for (size_t i = 0; i < imageLdr.size(); ++i)
		{
			u8 r = u8(saturate(at(i).x) * 255.0f);
			u8 g = u8(saturate(at(i).y) * 255.0f);
			u8 b = u8(saturate(at(i).z) * 255.0f);

			imageLdr[i] = r | (g << 8) | (b << 16) | 0xFF000000;
		}

		stbi_write_png(filename, m_size.x, m_size.y, 4, imageLdr.data(), m_size.x * 4);
	}

	void Image::paste(const Image& src, ivec2 pos)
	{
		ivec2 min = ivec2(0);
		ivec2 max = src.getSize();

		for (int y = min.y; y != max.y; ++y)
		{
			for (int x = min.x; x != max.x; ++x)
			{
				ivec2 srcPos = ivec2(x, y);
				ivec2 dstPos = srcPos + pos;
				at(dstPos) = src.at(srcPos);
			}
		}
	}

	bool Image::readHdr(const char* filename)
	{
		int w, h, comp;
		float* imageData = stbi_loadf(filename, &w, &h, &comp, 3);
		if (!imageData)
		{
			printf("ERROR: Failed to load image from file '%s'\n", filename);
			return false;
		}

		ivec2 size(w, h);
		*this = Image(size);

		vec3* inputPixels = reinterpret_cast<vec3*>(imageData);
        
        size_t pixelCount = upperHemisphereOnly ? m_pixels.size() / 2 : m_pixels.size();
		for (size_t i = 0; i < pixelCount; ++i)
		{
			m_pixels[i] = vec4(inputPixels[i], 1.0f);
		}
        
        if (upperHemisphereOnly) {
            for (size_t i = pixelCount; i < m_pixels.size(); ++i)
            {
                m_pixels[i] = vec4(0.f, 0.f, 0.f, 1.0f);
            }
        }

		free(imageData);

		return true;
	}

	void Image::writeHdr(const char* filename) const
	{
		if (m_pixels.empty()) return;
		stbi_write_hdr(filename, m_size.x, m_size.y, 4, data());
	}

	vec4 Image::sampleNearest(vec2 uv) const
	{
		ivec2 pos = floor(uv * (vec2)m_size);
		pos = clamp(pos, ivec2(0), m_size - 1);
		return at(pos);
	}
    
    vec4 Image::sampleBilinear(vec2 uv) const
    {
        vec2 pixelUV = uv * (vec2)m_size;
        vec2 fMinPixel = floor(pixelUV);
        
        ivec2 minPixel = clamp((ivec2)fMinPixel, ivec2(0), m_size - 1);
        ivec2 maxPixel = clamp((ivec2)ceil(pixelUV), ivec2(0), m_size - 1);
        
        vec2 weight = pixelUV - (vec2)minPixel;
        
        return at(minPixel) * (1 - weight.x) * (1 - weight.y) +
               at(ivec2(minPixel.x, maxPixel.y)) * (1 - weight.x) * weight.y +
               at(ivec2(maxPixel.x, minPixel.y)) * weight.x * (1 - weight.y) +
               at(maxPixel) * weight.x * weight.y;
    }

	Image imageResize(const Image& input, ivec2 newSize)
	{
		Image output(newSize);
		stbir_resize_float(input.data(), input.getWidth(), input.getHeight(), (int)input.getStrideBytes(), 
			output.data(), output.getWidth(), output.getHeight(), (int)output.getStrideBytes(), 4);
		return output;
	}

	Image imageDifference(const Image& reference, const Image& image)
	{
		ivec2 size = min(reference.getSize(), image.getSize());
        if (upperHemisphereOnly) {
            size.y /= 2;
        }

		Image result(size);

		for (int y = 0; y < size.y; ++y)
		{
			for (int x = 0; x < size.x; ++x)
			{
				vec4 error = reference.at(x, y) - image.at(x, y);
				result.at(x, y) = error;
			}
		}
		
		return result;
	}

	Image imageSymmetricAbsolutePercentageError(const Image& reference, const Image& image)
	{
		ivec2 size = min(reference.getSize(), image.getSize());
        if (upperHemisphereOnly) {
            size.y /= 2;
        }

		Image result(size);

		for (int y = 0; y < size.y; ++y)
		{
			for (int x = 0; x < size.x; ++x)
			{
				vec4 absDiff = abs(reference.at(x, y) - image.at(x, y));
				vec4 sum = reference.at(x, y) + image.at(x, y);
				result.at(x, y) = absDiff / sum;
			}
		}

		return result;
	}

	vec4 imageMeanSquareError(const Image& reference, const Image& image)
	{
		vec4 errorSquaredSum = vec4(0.0f);

		ivec2 size = min(reference.getSize(), image.getSize());
        if (upperHemisphereOnly) {
            size.y /= 2;
        }
		for (int y = 0; y < size.y; ++y)
		{
			for (int x = 0; x < size.x; ++x)
			{
				vec4 error = reference.at(x, y) - image.at(x, y);
				errorSquaredSum += error * error;
			}
		}

		errorSquaredSum /= size.x * size.y;

		return errorSquaredSum;
	}
    
    double sphericalDistance(feature_tt *a, feature_tt *b) {
        return acos(saturate(a->x * b->x + a->y * b->y + a->z * b->z));
    }
    
    vec4 imageEarthMoversDistance(const Image& reference, const Image& image)
    {
        
        ivec2 size = min(reference.getSize(), image.getSize());
        vec2 uvSize = image.getSize();
        
        if (upperHemisphereOnly) {
            size.y /= 2;
        }
        if (size.x * size.y == 0) {
            return vec4(0.f);
        }
        
        feature_tt features[size.x * size.y];
        
        double refR[size.x * size.y];
        double refG[size.x * size.y];
        double refB[size.x * size.y];
        
        double imageR[size.x * size.y];
        double imageG[size.x * size.y];
        double imageB[size.x * size.y];
        
        for (size_t y = 0; y < size.y; y += 1) {
            for (size_t x = 0; x < size.x; x += 1) {
                vec3 coord = latLongTexcoordToCartesian(vec2(x + 0.5f, y + 0.5f) / uvSize);
                features[y * size.x + x] = { coord.x, coord.y, coord.z };
                
                refR[y * size.x + x] = reference.at(x, y).r;
                refG[y * size.x + x] = reference.at(x, y).g;
                refB[y * size.x + x] = reference.at(x, y).b;
                
                imageR[y * size.x + x] = image.at(x, y).r;
                imageG[y * size.x + x] = image.at(x, y).g;
                imageB[y * size.x + x] = image.at(x, y).b;
            }
        }
        
        signature_tt<double> referenceSig;
        referenceSig.Features = features;
        referenceSig.n = size.x * size.y;
        
        signature_tt<double> imageSig = referenceSig;
        
        referenceSig.Weights = refR;
        imageSig.Weights = imageR;
        
        double errorR = emd_hat_signature_interface(&referenceSig, &imageSig, sphericalDistance, M_PI_2);
        
        referenceSig.Weights = refG;
        imageSig.Weights = imageG;
        
        double errorG = emd_hat_signature_interface(&referenceSig, &imageSig, sphericalDistance, M_PI_2);
        
        referenceSig.Weights = refB;
        imageSig.Weights = imageB;
        
        double errorB = emd_hat_signature_interface(&referenceSig, &imageSig, sphericalDistance, M_PI_2);
        
        return vec4(errorR, errorG, errorB, 0.f);
    }
}
