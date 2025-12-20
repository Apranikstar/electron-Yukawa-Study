#ifndef SORTJETCONSTITUENTS_H
#define SORTJETCONSTITUENTS_H

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include "ROOT/RVec.hxx"
#include "edm4hep/ReconstructedParticleData.h"

// Type alias for jet constituents (vector of ReconstructedParticleData)
using FCCAnalysesJetConstituents = ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>;

/*
  Sorts jet constituents based on inference scores (highest to lowest).
  Example usage in RDataFrame:
  includePaths = ["functions.h","SortJetConstituents.h"]
  df = df.Define("SortedJetConstituents", "FCCAnalyses::JetUtils::JetConstituentsSorter::sort_constituents_by_score(recojet_constituents, recojet_isTAU)")
*/
namespace FCCAnalyses { namespace JetUtils {
struct JetConstituentsSorter {
  JetConstituentsSorter() = default;
  
  // Static sorting function
  static ROOT::VecOps::RVec<FCCAnalysesJetConstituents> sort_constituents_by_score(
      const ROOT::VecOps::RVec<FCCAnalysesJetConstituents> &constituents,
      const ROOT::VecOps::RVec<float> &scores)
  {
    if (constituents.size() != scores.size())
    {
      throw std::runtime_error("JetConstituentsSorter::sort_constituents_by_score: constituents and scores must have the same size");
    }
    
    // Create index vector
    std::vector<size_t> indices(constituents.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // Sort by descending score
    std::sort(indices.begin(), indices.end(),
              [&](size_t i, size_t j) { return scores[i] > scores[j]; });
    
    // Build sorted constituents
    ROOT::VecOps::RVec<FCCAnalysesJetConstituents> sortedConstituents;
    sortedConstituents.reserve(constituents.size());
    for (auto i : indices)
      sortedConstituents.push_back(constituents[i]);
    
    return sortedConstituents;
  }
};
}} // namespace FCCAnalyses::JetUtils
#endif
