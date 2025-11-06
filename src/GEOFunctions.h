#ifndef GEOFunctions_H
#define GEOFunctions_H

/*

  Based on Code from: David G. Sheffield 
  This implementation of the code is basically redundent and can be summarized to a single struct for better performance,
  but works for now.

  How to call:
  
df = df.Define("Planarity","FCCAnalyses::GEOFunctions::EventGeoFunctions::calculatePlanarity(Jet1_P3,Jet2_P3,IsoLepton_3p)")
df = df.Define("Aplanarity","FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateAplanarity(Jet1_P3,Jet2_P3,IsoLepton_3p)")
df = df.Define("Sphericity","FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateSphericity(Jet1_P3,Jet2_P3,IsoLepton_3p)")
df = df.Define("ASphericity","FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateAsphericity(Jet1_P3,Jet2_P3,IsoLepton_3p)")

  */



#include <iostream>
#include <TVector3.h>
#include <TMatrixD.h>
#include <TMatrixDEigen.h>

namespace FCCAnalyses {
    namespace GEOFunctions {
        class EventGeoFunctions {
        public:
            static double calculateSphericity(const TVector3& p1, const TVector3& p2, const TVector3& p3) {
                // Momentum tensor
                TMatrixD S(3, 3);
                TVector3 momenta[] = {p1, p2, p3};
                double sum_p2 = 0.0;
                
                for (const auto& p : momenta) {
                    sum_p2 += p.Mag2();
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            S(i, j) += p[i] * p[j];
                        }
                    }
                }
                
                // Normalize
                if (sum_p2 == 0) return 0.0;
                S *= (1.0 / sum_p2);
                
                // Get eigenvalues
                TMatrixDEigen eig(S);
                TVectorD eigenvalues = eig.GetEigenValuesRe();
                
                // Sort eigenvalues: lambda1 >= lambda2 >= lambda3
                double lambda1 = eigenvalues[2];
                double lambda2 = eigenvalues[1];
                double lambda3 = eigenvalues[0];
                
                // Sphericity calculation
                return (3.0 / 2.0) * (lambda2 + lambda3);
            }

            static double calculateAplanarity(const TVector3& p1, const TVector3& p2, const TVector3& p3) {
                // Momentum tensor
                TMatrixD S(3, 3);
                TVector3 momenta[] = {p1, p2, p3};
                double sum_p2 = 0.0;
                
                for (const auto& p : momenta) {
                    sum_p2 += p.Mag2();
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            S(i, j) += p[i] * p[j];
                        }
                    }
                }
                
                // Normalize
                if (sum_p2 == 0) return 0.0;
                S *= (1.0 / sum_p2);
                
                // Get eigenvalues
                TMatrixDEigen eig(S);
                TVectorD eigenvalues = eig.GetEigenValuesRe();
                
                // Sort eigenvalues: lambda1 >= lambda2 >= lambda3
                double lambda1 = eigenvalues[2];
                double lambda2 = eigenvalues[1];
                double lambda3 = eigenvalues[0];
                
                // Aplanarity calculation
                return (3.0 / 2.0) * lambda3;
            }

            static double calculatePlanarity(const TVector3& p1, const TVector3& p2, const TVector3& p3) {
                // Momentum tensor
                TMatrixD S(3, 3);
                TVector3 momenta[] = {p1, p2, p3};
                double sum_p2 = 0.0;
                
                for (const auto& p : momenta) {
                    sum_p2 += p.Mag2();
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            S(i, j) += p[i] * p[j];
                        }
                    }
                }
                
                // Normalize
                if (sum_p2 == 0) return 0.0;
                S *= (1.0 / sum_p2);
                
                // Get eigenvalues
                TMatrixDEigen eig(S);
                TVectorD eigenvalues = eig.GetEigenValuesRe();
                
                // Sort eigenvalues: lambda1 >= lambda2 >= lambda3
                double lambda1 = eigenvalues[2];
                double lambda2 = eigenvalues[1];
                double lambda3 = eigenvalues[0];
                
                // Planarity calculation
                return lambda2 - lambda3;
            }

            static double calculateAsphericity(const TVector3& p1, const TVector3& p2, const TVector3& p3) {
                // Momentum tensor
                TMatrixD S(3, 3);
                TVector3 momenta[] = {p1, p2, p3};
                double sum_p2 = 0.0;
                
                for (const auto& p : momenta) {
                    sum_p2 += p.Mag2();
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            S(i, j) += p[i] * p[j];
                        }
                    }
                }
                
                // Normalize
                if (sum_p2 == 0) return 0.0;
                S *= (1.0 / sum_p2);
                
                // Get eigenvalues
                TMatrixDEigen eig(S);
                TVectorD eigenvalues = eig.GetEigenValuesRe();
                
                // Sort eigenvalues: lambda1 >= lambda2 >= lambda3
                double lambda1 = eigenvalues[2];
                double lambda2 = eigenvalues[1];
                double lambda3 = eigenvalues[0];
                
                // Asphericity calculation
                return (3.0 / 2.0) * (lambda1 - lambda2);
        };
    };
}}


#endif // GEOFunctions_H
