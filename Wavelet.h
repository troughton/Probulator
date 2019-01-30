#ifndef WAVELET_H
#define WAVELET_H

/*
FernUniversität in Hagen, Lehrgebiet Mensch-Computer-Interaktion
License: Creative Commons Attribution-Noncommercial-Share Alike 4.0 International, 
         see README file that comes with this work
Author: J. Kerdels

The use of this code for military and/or intelligence purposes is disapproved of.

This file contains the declaration of a discrete wavelet transform (class DWT)
and three classes of wavelets (Daubechies, Symlets, Coiflets) that can be
used with this transform. For information on its usage refer to the definition
in Wavelet.cpp
*/

#include <cstdlib>

typedef double Double_t;
typedef unsigned long UInt_t;
typedef long Int_t;
typedef bool Bool_t;

class Wavelet;

class DWT
{
	public:

		static void Transform(Double_t *data, UInt_t *dimensions, Bool_t inverse, Wavelet *wavelet);

		static void Transform(Double_t *data, UInt_t xCnt, Bool_t inverse, Wavelet *wavelet);

		static void Transform(Double_t *data, UInt_t xCnt, UInt_t yCnt, Bool_t inverse, Wavelet *wavelet);

		static void Transform(Double_t *data, UInt_t xCnt, UInt_t yCnt, UInt_t zCnt, Bool_t inverse, Wavelet *wavelet);

};


class Wavelet 
{
	public:
		Wavelet();
		virtual ~Wavelet(); 

		void Filter(Double_t *data, UInt_t size, Bool_t inverse);

	protected:

		void SetFilterCoefficients(const Double_t *filterCoefficients, UInt_t count);

	private:

		void FreeCoefficients();

		void FreeBuffer();

		UInt_t    fNrOfFilterCoefficients; // size of mother/wavelet filter
		Double_t *fSmoothCoefficients;     //! "smooth" filter
		Double_t *fDetailCoefficients;     //! "detail" filter
		Int_t     fCenteringOffset;        // offset used for filter

		UInt_t    fBufferSize; // size of temporary buffer
		Double_t *fBuffer;     //! temporary buffer
};


class Daubechies : public Wavelet
{
	public:
		Daubechies(UInt_t order = 2);

	private:

		static const Double_t D1[];
		static const Double_t D2[];
		static const Double_t D3[];
		static const Double_t D4[];
		static const Double_t D5[];
		static const Double_t D6[];
		static const Double_t D7[];
		static const Double_t D8[];
		static const Double_t D9[];
		static const Double_t D10[];
		static const Double_t D11[];
		static const Double_t D12[];
		static const Double_t D13[];
		static const Double_t D14[];
		static const Double_t D15[];
		static const Double_t D16[];
		static const Double_t D17[];
		static const Double_t D18[];
		static const Double_t D19[];
		static const Double_t D20[];

};



class Symlets : public Wavelet
{
	public:
		Symlets(UInt_t order = 2);

	private:

		static const Double_t S2[];
		static const Double_t S3[];
		static const Double_t S4[];
		static const Double_t S5[];
		static const Double_t S6[];
		static const Double_t S7[];
		static const Double_t S8[];
		static const Double_t S9[];
		static const Double_t S10[];
		static const Double_t S11[];
		static const Double_t S12[];
		static const Double_t S13[];
		static const Double_t S14[];
		static const Double_t S15[];
		static const Double_t S16[];
		static const Double_t S17[];
		static const Double_t S18[];
		static const Double_t S19[];
		static const Double_t S20[];

};



class Coiflets : public Wavelet
{
	public:
		Coiflets(UInt_t order = 1);

	private:

		static const Double_t C1[];
		static const Double_t C2[];
		static const Double_t C3[];
		static const Double_t C4[];
		static const Double_t C5[];

};


#endif
