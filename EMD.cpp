/*
FernUniversität in Hagen, Lehrgebiet Mensch-Computer-Interaktion
License: Creative Commons Attribution-Noncommercial-Share Alike 4.0 International, 
         see README file that comes with this work
Author: J. Kerdels

The use of this code for military and/or intelligence purposes is disapproved of.

This file contains the definition of an approximate earth movers distance based
on Shirdhonka and Jacobs 2008.

*/

#include <limits>
#include <cstdio>
#include <cmath>

#include "Wavelet.h"
#include "EMD.h"

const UInt_t EMD::Mod37BitPosition[] = // map a power of 2 bit value mod 37 to its position
{
  32, 0, 1, 26, 2, 23, 27, 0, 3, 16, 24, 30, 28, 11, 0, 13, 4,
  7, 17, 0, 25, 22, 31, 15, 29, 10, 12, 6, 0, 21, 14, 9, 5,
  20, 8, 19, 18
};

Double_t EMD::ApproxEMD(Double_t *a, Double_t *b, UInt_t *dimensions, Bool_t normalize) 
{
	// *a, *b      : data to calculate EMD for, must be powers of 2
	// *dimensions : null-terminated array of dimension sizes
	// normalize   : if true normalizes data on the fly (a,b are not modified)

	if ((a == 0) || (b == 0) || (dimensions == 0) || (dimensions[0] == 0))
        return std::numeric_limits<Double_t>::max();

	const Double_t epsilon = 0.00001;

	UInt_t size   = dimensions[0];
	UInt_t maxDim = dimensions[0];
	UInt_t dCnt = 1;
	while (dimensions[dCnt] > 0) {
		size *= dimensions[dCnt];

		if (dimensions[dCnt] > maxDim)
			maxDim = dimensions[dCnt];

		++dCnt;
	}

	// check if size is power of 2
	if ((size & (size - 1)) != 0) {
		std::printf("all dimensions have to be powers of 2\n");
        return std::numeric_limits<Double_t>::max();
	}

	Double_t sum_a = 0;
	Double_t sum_b = 0;

	if (normalize == true) {
		for (UInt_t i = 0; i < size; ++i) {
			sum_a += a[i];
			sum_b += b[i];
		}
		if (sum_a < epsilon)
			sum_a = 1;
		if (sum_b < epsilon)
			sum_b = 1;
	} else {
		sum_a = 1;
		sum_b = 1;
	}

	// transform delta of arrays
	Double_t *delta = new Double_t[size];
	for (UInt_t i = 0; i < size; ++i) {
		delta[i] = (a[i] / sum_a) - (b[i] / sum_b);		
	}

	Symlets *sym = new Symlets(5); // according to (Shirdhonka and Jacobs 2008) we use a symlet of order 5

	DWT::Transform(delta,dimensions,false,sym);

	delete sym;

	// preparing lookup table to find the biggest coordinate value of each entry
	// in the multi-dimensional arrays

	UInt_t *dMod   = new UInt_t[dCnt];
	UInt_t *dShift = new UInt_t[dCnt];

	UInt_t modVal = 1;	
	for (UInt_t i = 0; i < dCnt; ++i) {
		dShift[i]  = Mod37BitPosition[modVal % 37];
		modVal    *= dimensions[i];
		dMod[i]    = modVal-1; // works bc every dim is power of 2
	}

	// prepare weights
	UInt_t wCnt = Mod37BitPosition[maxDim % 37];
	Double_t *weight = new Double_t[wCnt-1];
	for (UInt_t i = 0; i < wCnt-1; ++i)
        weight[i] = std::pow(2.0,-(Double_t)i * (1.0 + (Double_t)dCnt / 2.0));

	// calculating the distance
	Double_t distance = 0;
	for (UInt_t i = 0; i < size; ++i) {
		// get maximum coordinate value
		UInt_t mcv = 0;
		for (UInt_t d = 0; d < dCnt; ++d) {
			UInt_t cv = (i & dMod[d]) >> dShift[d];
			if (cv > mcv)
				mcv = cv;
		}
		// round up to next power of 2
		// note: if mcv is exactly a power of 2 it should
		// get _also_ increased to the following power of 2.
		// so the typical step of --mcv at the beginning is
		// skipped.
		mcv |= mcv >>  1;
		mcv |= mcv >>  2;
		mcv |= mcv >>  4;
		mcv |= mcv >>  8;
		mcv |= mcv >> 16;
		++mcv;
		// get the exponent
		UInt_t ep = Mod37BitPosition[mcv % 37];
		// do only use wavelet coefficients and not mother function coefficients
		if (ep < 2)
			continue;

        distance += std::abs(delta[i]) * weight[ep-2];
	}

	delete[] dMod;
	delete[] dShift;
	delete[] weight;
	delete[] delta;

	return distance;
}



