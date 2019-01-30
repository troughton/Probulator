#ifndef EMD_H
#define EMD_H
/*
FernUniversität in Hagen, Lehrgebiet Mensch-Computer-Interaktion
License: Creative Commons Attribution-Noncommercial-Share Alike 4.0 International, 
         see README file that comes with this work
Author: J. Kerdels

The use of this code for military and/or intelligence purposes is disapproved of.

This file contains the declaration of an approximate earth movers distance.

*/

typedef double Double_t;
typedef unsigned long UInt_t;
typedef long Int_t;
typedef bool Bool_t;

class EMD
{
	public:

		static Double_t ApproxEMD(Double_t *a, Double_t *b, UInt_t *dimensions, Bool_t normalize);

//    Double_t EMD::ApproxEMD(TMatrixD *a, TMatrixD *b, Bool_t normalize)
//    {
//        // *a, *b      : data to calculate normalized EMD for, dimensions must be powers of 2
//
//        UInt_t nc = (UInt_t)(a->GetNcols());
//        UInt_t nr = (UInt_t)(a->GetNrows());
//        UInt_t dims[] = {nc, nr, 0};
//
//        Double_t *aData = a->GetMatrixArray();
//        Double_t *bData = b->GetMatrixArray();
//
//        return ApproxEMD(aData,bData,dims,normalize);
//    }
//
//    Double_t EMD::ApproxEMD(TVectorD *a, TVectorD *b, Bool_t normalize)
//    {
//        // *a, *b      : data to calculate normalized EMD for, dimensions must be powers of 2
//
//        UInt_t nr = (UInt_t)(a->GetNrows());
//        UInt_t dims[] = {nr, 0};
//
//        Double_t *aData = a->GetMatrixArray();
//        Double_t *bData = b->GetMatrixArray();
//
//        return ApproxEMD(aData,bData,dims,normalize);
//    }

	private:

		static const UInt_t Mod37BitPosition[];

};


#endif
