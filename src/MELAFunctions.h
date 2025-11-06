#ifndef MELA_H
#define MELA_H

/*
  Calculates the variables all at once. 
  Jets_p4[0] --> Leading jet 4Vector
  Jets_p4[1] --> Subleading jet 4Vector
  MissingE_4p --> MET 4Vector
  IsoLepton_4p --> Lepton 4Vector
  

includePaths = ["functions.h","MELAVar.h"]
df = df.Define("mela", "FCCAnalyses::MELA::MELACalculator::mela(Jets_p4[0],Jets_p4[1],MissingE_4p[0],IsoLepton_4p[0], Iso_Lepton_Charge, Jet1_charge ,Jet2_charge)")
df = df.Define("Phi", " mela.phi")
df = df.Define("Phi1", " mela.phi1")
df = df.Define("PhiStar", " mela.phiStar")
df = df.Define("CosPhiStar", " mela.CosPhiStar")

df = df.Define("ThetaStar", " mela.thetaStar")
df = df.Define("CosThetaStar", "mela.CosThetaStar")
df = df.Define("Theta1", " mela.theta1")
df = df.Define("Theta2", " mela.theta2")

df = df.Define("Boost1", " mela.boost1")
df = df.Define("Boost2", " mela.boost2")



  */

#include <TLorentzVector.h>
#include <TVector3.h>
#include <TMath.h>

namespace FCCAnalyses { namespace MELA {

struct MELACalculator {
    // Input variables
    TLorentzVector jet1, jet2, MEVector, lepVec;
    float lepCharge, jetCharge1, jetCharge2;

    // Computed values
    TLorentzVector q11, q12, q21, q22;
    TVector3 boost1, boost2;
    double phi = 0, phi1 = 0, thetaStar = 0, phiStar = 0, theta1 = 0, theta2 = 0;
    double cosPhiStar = 0, cosThetaStar = 0;
    double cosPhi = 0, cosPhi1 = 0, cosTheta1 = 0, cosTheta2 = 0;


    MELACalculator() = default;
    // Constructor
    MELACalculator(TLorentzVector jet1_, TLorentzVector jet2_, TLorentzVector MEVector_, TLorentzVector lepVec_, 
                   float lepCharge_, float jetCharge1_, float jetCharge2_)
        : jet1(jet1_), jet2(jet2_), MEVector(MEVector_), lepVec(lepVec_), 
          lepCharge(lepCharge_), jetCharge1(jetCharge1_), jetCharge2(jetCharge2_) {}

    // Compute function (modifies struct in place)
    MELACalculator& compute() {
        TLorentzVector sumjj = jet1 + jet2;
        TLorentzVector WVec = MEVector + lepVec;

        if (WVec.M() > sumjj.M()) {
            boost1 = WVec.BoostVector();
            boost2 = sumjj.BoostVector();

            if (lepCharge < 0) {
                q11 = lepVec;
                q12 = MEVector;
            } else {
                q11 = MEVector;
                q12 = lepVec;
            }

            if (jetCharge1 > jetCharge2) {
                q21 = jet1;
                q22 = jet2;
            } else {
                q21 = jet2;
                q22 = jet1;
            }
        } else {
            boost2 = WVec.BoostVector();
            boost1 = sumjj.BoostVector();

            if (lepCharge < 0) {
                q21 = lepVec;
                q22 = MEVector;
            } else {
                q21 = MEVector;
                q22 = lepVec;
            }

            if (jetCharge1 > jetCharge2) {
                q11 = jet1;
                q12 = jet2;
            } else {
                q11 = jet2;
                q12 = jet1;
            }
        }

        TLorentzVector q1 = q11 + q12;
        TLorentzVector q2 = q21 + q22;

        TVector3 n1 = q11.Vect().Cross(q12.Vect()).Unit();
        TVector3 n2 = q21.Vect().Cross(q22.Vect()).Unit();
        TVector3 nz(0., 0., 1.);
        TVector3 nsc = nz.Cross(q1.Vect()).Unit();
        
        phiStar = q1.Vect().Unit().Phi();
        cosPhiStar = TMath::Cos(phiStar);
        phi = TMath::Cos(q1.Angle(n1.Cross(n2))) * TMath::ACos(-1 * n1.Dot(n2));
        phi1 = TMath::Cos(q1.Angle(n1.Cross(nsc))) * TMath::ACos(n1.Dot(nsc));

        thetaStar = q1.Vect().Unit().Theta();
        cosThetaStar = TMath::Cos(thetaStar);
        theta1 = TMath::ACos(-1*TMath::Cos(q2.Angle(q11.Vect())));
        theta2 = TMath::ACos(-1*TMath::Cos(q1.Angle(q21.Vect())));
      
        cosPhi = TMath::Cos(phi);
        cosPhi1 = TMath::Cos(phi1);
        cosTheta1 = TMath::Cos(theta1);
        cosTheta2 = TMath::Cos(theta2);


        return *this;  // Allows method chaining
    }

    // Static function for easy one-line computation
    static MELACalculator mela(TLorentzVector jet1, TLorentzVector jet2, TLorentzVector MEVector, TLorentzVector lepVec, 
                               float lepCharge, float jetCharge1, float jetCharge2) {
        return MELACalculator(jet1, jet2, MEVector, lepVec, lepCharge, jetCharge1, jetCharge2).compute();
    }
};

}} // namespace

#endif
