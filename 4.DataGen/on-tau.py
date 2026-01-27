import json
import os
import copy

# --------------------------------
# Read config file (environment variable or default)
# --------------------------------

# Parameters with defaults
drmin = 0.01
drmax = 0.2
selection = 0.2

# -----------------------------
# Process List
# -----------------------------

processList = {

            # Semileptonic processes
            "wzp6_ee_Henueqq_ecm125": {"fraction": 1,'chunks': 1},
            "wzp6_ee_Hqqenue_ecm125": {"fraction": 1,'chunks': 1},
            "wzp6_ee_Hmunumuqq_ecm125": {"fraction": 1,'chunks': 1},
            "wzp6_ee_Hqqmunumu_ecm125": {"fraction": 1,'chunks': 1},
            "wzp6_ee_Htaunutauqq_ecm125": {"fraction": 1,'chunks': 1},
            "wzp6_ee_Hqqtaunutau_ecm125": {"fraction": 1,'chunks': 1},


            "wzp6_ee_taunutauqq_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_tautauqq_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_enueqq_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_eeqq_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_munumuqq_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_mumuqq_ecm125": {"fraction": 1,'chunks': 10},
            # Fully leptonic Processes
            #"wzp6_ee_Htautau_ecm125": {"fraction": 1},
            #"wzp6_ee_Hllnunu_ecm125": {"fraction": 1},
            "wzp6_ee_eenunu_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_mumununu_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_tautaununu_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_l1l2nunu_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_tautau_ecm125": {"fraction": 1,'chunks': 10},
            # Fully hadronic Processes
            #"wzp6_ee_Hgg_ecm125": {"fraction": 1},
            #"wzp6_ee_Hbb_ecm125": {"fraction": 1},
            "wzp6_ee_qq_ecm125": {"fraction": 1,'chunks': 10},
            "p8_ee_ZZ_4tau_ecm125": {"fraction": 1,'chunks': 10},
}

outputDir   = "/eos/experiment/fcc/ee/analyses/case-studies/higgs/electron_yukawa/DataGenReduced-CHCut/on-shell-tau"
inputDir    = "/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA"
nCPUS       = -1
includePaths = ["functions.h", "SortJets.h", "GEOFunctions.h", "MELAFunctions.h","JetConstituentsSorter.h"]

model_name = "fccee_flavtagging_edm4hep_wc"

url_model_dir = "https://fccsw.web.cern.ch/fccsw/testsamples/jet_flavour_tagging/winter2023/wc_pt_13_01_2022/"
url_preproc = "{}/{}.json".format(url_model_dir, model_name)
url_model = "{}/{}.onnx".format(url_model_dir, model_name)

model_dir = (
    "/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/wc_pt_7classes_12_04_2023/"
)
local_preproc = "{}/{}.json".format(model_dir, model_name)
local_model = "{}/{}.onnx".format(model_dir, model_name)

def get_file_path(url, filename):
    if os.path.exists(filename):
        return os.path.abspath(filename)
    else:
        urllib.request.urlretrieve(url, os.path.basename(url))
        return os.path.basename(url)

weaver_preproc = get_file_path(url_preproc, local_preproc)
weaver_model = get_file_path(url_model, local_model)

from addons.ONNXRuntime.jetFlavourHelper import JetFlavourHelper
from addons.FastJet.jetClusteringHelper import ExclusiveJetClusteringHelper

jetFlavourHelper = None
jetClusteringHelper = None

# Mandatory: RDFanalysis class where the use defines the operations on the TTree
class RDFanalysis:
    # ______________________________________________________________________________________________________
    @staticmethod
    def analysers(df):
        '''
        Analysis graph for hadronic tau Yukawa coupling measurement.
        '''
        # ===========================
        # Define aliases
        # ===========================
        df = df.Alias("Particle0", "Particle#0.index")
        df = df.Alias("Particle1", "Particle#1.index")
        df = df.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
        df = df.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")
        df = df.Alias("Electron0", "Electron#0.index")
        df = df.Alias("Muon0", "Muon#0.index")
        df = df.Alias("Photon0", "Photon#0.index")
        df = df.Alias("Jet2","Jet#2.index")
        
        # ===========================
        # Photons and charged hadrons
        # ===========================
        df = df.Define("photons_all", "FCCAnalyses::ReconstructedParticle::get(Photon0, ReconstructedParticles)")
        df = df.Define("photons", "FCCAnalyses::ReconstructedParticle::sel_p(0)(photons_all)")
        df = df.Define("ChargedHadrons", "ReconstructedParticle2MC::selRP_ChargedHadrons(MCRecoAssociations0,MCRecoAssociations1,ReconstructedParticles,Particle)")
        
        # ===========================
        # Leptons
        # ===========================
        df = df.Define("electrons_all", "FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)")
        df = df.Define("muons_all", "FCCAnalyses::ReconstructedParticle::get(Muon0, ReconstructedParticles)")
        df = df.Define("electrons", "FCCAnalyses::ReconstructedParticle::sel_p(0)(electrons_all)")
        df = df.Define("muons", "FCCAnalyses::ReconstructedParticle::sel_p(0)(muons_all)")

        # ===========================
        # Isolation using JSON-configurable parameters
        # ===========================
        df = df.Define(
            "electrons_iso",
            f"FCCAnalyses::ZHfunctions::coneIsolation({drmin}, {drmax})(electrons, ChargedHadrons)"
        )
        df = df.Define(
            "electrons_sel_iso",
            f"FCCAnalyses::ZHfunctions::sel_iso({selection})(electrons, electrons_iso)"
        )
        df = df.Define(
            "muons_iso",
            f"FCCAnalyses::ZHfunctions::coneIsolation({drmin}, {drmax})(muons, ChargedHadrons)"
        )
        df = df.Define(
            "muons_sel_iso",
            f"FCCAnalyses::ZHfunctions::sel_iso({selection})(muons, muons_iso)"
        )
        #df = df.Define("photons_iso", "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.2)(photons, ChargedHadrons)")
        #df = df.Define("photons_sel_iso","FCCAnalyses::ZHfunctions::sel_iso(0.2)(photons, photons_iso)")

        # ===========================
        # Lepton counting and vetoing
        # ===========================
        df = df.Define("IsoMuonNum", "muons_sel_iso.size()")
        df = df.Define("Iso_Electrons_No", "electrons_sel_iso.size()")
        df = df.Filter("IsoMuonNum == 0 && Iso_Electrons_No == 0")

        #df = df.Define("IsoElectron_4p", "FCCAnalyses::ReconstructedParticle::get_tlv(electrons_sel_iso)")
        #df = df.Define("IsoMuon_4p", "FCCAnalyses::ReconstructedParticle::get_tlv(muons_sel_iso)")

        # ===========================
        # Missing energy
        # ===========================
        df = df.Define("MissingE_4p", "FCCAnalyses::ReconstructedParticle::get_tlv(MissingET)")
        df = df.Define("Missing_Pt", "MissingE_4p[0].Pt()")
        df = df.Filter("Missing_Pt > 3")
        
        df = df.Define("MissingQuantities_P", "MissingE_4p[0].P()")
        df = df.Define("MissingQuantities_E", "MissingE_4p[0].E()")
        df = df.Define("MissingQuantities_M", "MissingE_4p[0].M()")
        df = df.Define("MissingQuantities_Theta", "MissingE_4p[0].Theta()")
        df = df.Define("MissingQuantities_Phi", "MissingE_4p[0].Phi()")
        df = df.Define("MissingQuantities_CosTheta", "MissingE_4p[0].CosTheta()")
        df = df.Define("MissingQuantities_CosPhi", "TMath::Cos(MissingQuantities_Phi)")

        # ===========================
        # Jet Clustering using "jetClusteringHelper": (Durham-kt)
        # ===========================
        collections = {
            "GenParticles": "Particle",
            "PFParticles": "ReconstructedParticles",
            "PFTracks": "EFlowTrack",
            "PFPhotons": "EFlowPhoton",
            "PFNeutralHadrons": "EFlowNeutralHadron",
            "TrackState": "EFlowTrack_1",
            "TrackerHits": "TrackerHits",
            "CalorimeterHits": "CalorimeterHits",
            "dNdx": "EFlowTrack_2",
            "PathLength": "EFlowTrack_L",
            "Bz": "magFieldBz",
        }

        jetClusteringHelper = ExclusiveJetClusteringHelper(collections["PFParticles"], 3)
        df = jetClusteringHelper.define(df)
        
        # ===========================
        # Jet flavour tagging
        # ===========================
        jetFlavourHelper = JetFlavourHelper(
            collections,
            jetClusteringHelper.jets,
            jetClusteringHelper.constituents,
        )
        df = jetFlavourHelper.define(df)
        df = jetFlavourHelper.inference(weaver_preproc, weaver_model, df)

        df = df.Filter("event_njet > 2")

        # ===========================
        # Sort jets by tau score
        # ===========================
        df = df.Define("SortedJets", f"FCCAnalyses::JetUtils::JetSorter::sort_jets_by_score({jetClusteringHelper.jets}, recojet_isTAU)")
        df = df.Define("SortedJetConstituents", f"FCCAnalyses::JetUtils::JetConstituentsSorter::sort_constituents_by_score({jetClusteringHelper.constituents}, recojet_isTAU)")

        #df = df.Define("SortedIndices", "FCCAnalyses::JetUtils::JetSorter::get_sorted_indices(recojet_isTAU)")

        df = df.Define("jet_nconst2", "JetConstituentsUtils::count_consts(SortedJetConstituents)") 
        df = df.Define("JetNConst1", "jet_nconst2[0]")
        df = df.Define("JetNConst2", "jet_nconst2[1]")
        df = df.Define("JetNConst3", "jet_nconst2[2]")
        df = df.Define("Jets_p4", "JetConstituentsUtils::compute_tlv_jets(SortedJets)")
        # ===========================
        # Hadronic tau kinematics (Jets_p4[0] = hadronic tau)
        # ===========================
        df = df.Define("HadronicTau_3Vec", "Jets_p4[0].Vect()")
        df = df.Define("HadronicTau_P", "Jets_p4[0].P()")
        df = df.Define("HadronicTau_M", "Jets_p4[0].M()")
        df = df.Define("HadronicTau_Phi", "Jets_p4[0].Phi()")
        df = df.Define("HadronicTau_Theta", "Jets_p4[0].Theta()")
        df = df.Define("HadronicTau_E", "Jets_p4[0].E()")
        df = df.Define("HadronicTau_CosTheta", "Jets_p4[0].CosTheta()")
        df = df.Define("HadronicTau_CosPhi", "TMath::Cos(HadronicTau_Phi)")
        
        # ===========================
        # Di-jet system (Jets 1 and 2)
        # ===========================
        df = df.Define("Jets_InMa", "(Jets_p4[1]+Jets_p4[2]).M()")
        df = df.Filter("Jets_InMa < 45.2")
        df = df.Filter("Jets_InMa > 4")

        
        # Jet 1 kinematics
        df = df.Define("Jet1_3Vec", "Jets_p4[1].Vect()")
        df = df.Define("Jet1_P", "Jets_p4[1].P()")
        df = df.Define("Jet1_Eta", "Jets_p4[1].Eta()")
        df = df.Define("Jet1_Phi", "Jets_p4[1].Phi()")
        df = df.Define("Jet1_M", "Jets_p4[1].M()")
        df = df.Define("Jet1_E", "Jets_p4[1].E()")
        df = df.Define("Jet1_Theta", "Jets_p4[1].Theta()")
        df = df.Define("Jet1_CosTheta", "Jets_p4[1].CosTheta()")
        df = df.Define("Jet1_CosPhi", "TMath::Cos(Jet1_Phi)")
        
        # Jet 2 kinematics
        df = df.Define("Jet2_3Vec", "Jets_p4[2].Vect()")
        df = df.Define("Jet2_P", "Jets_p4[2].P()")
        df = df.Define("Jet2_Eta", "Jets_p4[2].Eta()")
        df = df.Define("Jet2_Phi", "Jets_p4[2].Phi()")
        df = df.Define("Jet2_M", "Jets_p4[2].M()")
        df = df.Define("Jet2_E", "Jets_p4[2].E()")
        df = df.Define("Jet2_Theta", "Jets_p4[2].Theta()")
        df = df.Define("Jet2_CosTheta", "Jets_p4[2].CosTheta()")
        df = df.Define("Jet2_CosPhi", "TMath::Cos(Jet2_Phi)")
        
        # Jet energy extremes
        df = df.Define("Jets_MaxEnergy", "TMath::Max(Jets_p4[1].E(), Jets_p4[2].E())")
        df = df.Define("Jets_MinEnergy", "TMath::Min(Jets_p4[1].E(), Jets_p4[2].E())")
        
        # ===========================
        # Jet clustering d-parameters
        # ===========================
        jetClusteringHelper_N3 = ExclusiveJetClusteringHelper("ReconstructedParticles", 3, "N3")
        df = jetClusteringHelper_N3.define(df)
        df = df.Define("JetClustering_d23", "std::sqrt(JetClusteringUtils::get_exclusive_dmerge(_jet_N3, 2))")
        df = df.Define("JetClustering_d34", "std::sqrt(JetClusteringUtils::get_exclusive_dmerge(_jet_N3, 3))")
        
        # ===========================
        # PAIR SYSTEM DEFINITIONS
        # ===========================
        # Hadronic W Boson (Hadronic Tau + Neutrino)

        df = df.Define("HadronicW_4Vec", "Jets_p4[1] + MissingE_4p[2]")
        df = df.Define("HadronicW_P", "HadronicW_4Vec.P()")
        df = df.Define("HadronicW_Phi", "HadronicW_4Vec.Phi()")
        df = df.Define("HadronicW_CosPhi", "TMath::Cos(HadronicW_Phi)")
        df = df.Define("HadronicW_Theta", "HadronicW_4Vec.Theta()")
        df = df.Define("HadronicW_CosTheta", "HadronicW_4Vec.CosTheta()")
        df = df.Define("HadronicW_InvariantMass", "HadronicW_4Vec.M()")
        
        # Leptonic W Boson (Jet1 + Jet2)
        df = df.Define("LeptonicW_4Vec", "Jets_p4[0] + MissingE_4p[0]")
        df = df.Define("LeptonicW_P", "LeptonicW_4Vec.P()")
        df = df.Define("LeptonicW_Phi", "LeptonicW_4Vec.Phi()")
        df = df.Define("LeptonicW_CosPhi", "TMath::Cos(LeptonicW_Phi)")
        df = df.Define("LeptonicW_Theta", "LeptonicW_4Vec.Theta()")
        df = df.Define("LeptonicW_CosTheta", "LeptonicW_4Vec.CosTheta()")
        df = df.Define("LeptonicW_InvariantMass", "LeptonicW_4Vec.M()")
        
        # Angular separations between the two W bosons
        df = df.Define("WW_DeltaR", "HadronicW_4Vec.DeltaR(LeptonicW_4Vec)")
        df = df.Define("WW_DeltaPhi", "HadronicW_4Vec.DeltaPhi(LeptonicW_4Vec)")
        df = df.Define("WW_DeltaTheta", "HadronicW_Theta - LeptonicW_Theta")
        df = df.Define("WW_DeltaInvariantMass", "HadronicW_InvariantMass - LeptonicW_InvariantMass")
        df = df.Define("WW_Angle", "HadronicW_4Vec.Angle(LeptonicW_4Vec.Vect())")
        df = df.Define("WW_CosAngle", "TMath::Cos(WW_Angle)")

        # ===========================
        # Event-level invariant mass
        # ===========================
        df = df.Define("Event_InvariantMass", "(Jets_p4[0] + Jets_p4[1] + Jets_p4[2] + MissingE_4p[0]).M()")

        # ===========================
        # On-shell and off-shell mass variables
        # ===========================
        df = df.Define("System_OnShellMass", "TMath::Max(HadronicW_InvariantMass, LeptonicW_InvariantMass)")
        df = df.Define("System_OffShellMass", "TMath::Min(HadronicW_InvariantMass, LeptonicW_InvariantMass)")
        df = df.Define("System_MaxCosTheta", "TMath::Cos(TMath::Max(HadronicW_Theta, LeptonicW_Theta))")
        df = df.Define("System_MinCosTheta", "TMath::Cos(TMath::Min(HadronicW_Theta, LeptonicW_Theta))")
        df = df.Define("System_MaxCosPhi", "TMath::Cos(TMath::Max(HadronicW_Phi, LeptonicW_Phi))")
        df = df.Define("System_MinCosPhi", "TMath::Cos(TMath::Min(HadronicW_Phi, LeptonicW_Phi))")

        # ===========================
        # Event-shape variables
        # ===========================
        df = df.Define("EventShape_Planarity", "FCCAnalyses::GEOFunctions::EventGeoFunctions::calculatePlanarity(Jet1_3Vec, Jet2_3Vec, HadronicTau_3Vec)")
        df = df.Define("EventShape_Aplanarity", "FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateAplanarity(Jet1_3Vec, Jet2_3Vec, HadronicTau_3Vec)")
        df = df.Define("EventShape_Sphericity", "FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateSphericity(Jet1_3Vec, Jet2_3Vec, HadronicTau_3Vec)")
        df = df.Define("EventShape_Asphericity", "FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateAsphericity(Jet1_3Vec, Jet2_3Vec, HadronicTau_3Vec)")

        # ===========================
        # Jet flavor tagging scores - Products and Logits
        # ===========================
        epsilon = 1e-3
        
        df = df.Define("SortedIndices", "FCCAnalyses::JetUtils::JetSorter::get_sorted_indices(recojet_isTAU)")

        df = df.Define("JetFlavor_UpQuarkProduct", "recojet_isU[SortedIndices[1]] * recojet_isU[SortedIndices[2]]")
        df = df.Define("JetFlavor_UpQuarkProduct_Logit", f"TMath::Log((JetFlavor_UpQuarkProduct + {epsilon}) / (1.0 - JetFlavor_UpQuarkProduct + {epsilon}))")

        df = df.Define("JetFlavor_DownQuarkProduct", "recojet_isD[SortedIndices[1]] * recojet_isD[SortedIndices[2]]")
        df = df.Define("JetFlavor_DownQuarkProduct_Logit", f"TMath::Log((JetFlavor_DownQuarkProduct + {epsilon}) / (1.0 - JetFlavor_DownQuarkProduct + {epsilon}))")

        df = df.Define("JetFlavor_StrangeQuarkProduct", "recojet_isS[SortedIndices[1]] * recojet_isS[SortedIndices[2]]")
        df = df.Define("JetFlavor_StrangeQuarkProduct_Logit", f"TMath::Log((JetFlavor_StrangeQuarkProduct + {epsilon}) / (1.0 - JetFlavor_StrangeQuarkProduct + {epsilon}))")

        df = df.Define("JetFlavor_CharmQuarkProduct", "recojet_isC[SortedIndices[1]] * recojet_isC[SortedIndices[2]]")
        df = df.Define("JetFlavor_CharmQuarkProduct_Logit", f"TMath::Log((JetFlavor_CharmQuarkProduct + {epsilon}) / (1.0 - JetFlavor_CharmQuarkProduct + {epsilon}))")

        df = df.Define("JetFlavor_BottomQuarkProduct", "recojet_isB[SortedIndices[1]] * recojet_isB[SortedIndices[2]]")
        df = df.Define("JetFlavor_BottomQuarkProduct_Logit", f"TMath::Log((JetFlavor_BottomQuarkProduct + {epsilon}) / (1.0 - JetFlavor_BottomQuarkProduct + {epsilon}))")

        df = df.Define("JetFlavor_TauProduct", "recojet_isTAU[SortedIndices[1]] * recojet_isTAU[SortedIndices[2]]")
        df = df.Define("JetFlavor_TauProduct_Logit", f"TMath::Log((JetFlavor_TauProduct + {epsilon}) / (1.0 - JetFlavor_TauProduct + {epsilon}))")

        df = df.Define("JetFlavor_UpDownQuarkProduct", "recojet_isU[SortedIndices[1]] * recojet_isD[SortedIndices[2]]")
        df = df.Define("JetFlavor_UpDownQuarkProduct_Logit", f"TMath::Log((JetFlavor_UpDownQuarkProduct + {epsilon}) / (1.0 - JetFlavor_UpDownQuarkProduct + {epsilon}))")

        df = df.Define("JetFlavor_DownUpQuarkProduct", "recojet_isD[SortedIndices[1]] * recojet_isU[SortedIndices[2]]")
        df = df.Define("JetFlavor_DownUpQuarkProduct_Logit", f"TMath::Log((JetFlavor_DownUpQuarkProduct + {epsilon}) / (1.0 - JetFlavor_DownUpQuarkProduct + {epsilon}))")

        df = df.Define("JetFlavor_StrangeCharmQuarkProduct", "recojet_isS[SortedIndices[1]] * recojet_isC[SortedIndices[2]]")
        df = df.Define("JetFlavor_StrangeCharmQuarkProduct_Logit", f"TMath::Log((JetFlavor_StrangeCharmQuarkProduct + {epsilon}) / (1.0 - JetFlavor_StrangeCharmQuarkProduct + {epsilon}))")

        df = df.Define("JetFlavor_CharmStrangeQuarkProduct", "recojet_isC[SortedIndices[1]] * recojet_isS[SortedIndices[2]]")
        df = df.Define("JetFlavor_CharmStrangeQuarkProduct_Logit", f"TMath::Log((JetFlavor_CharmStrangeQuarkProduct + {epsilon}) / (1.0 - JetFlavor_CharmStrangeQuarkProduct + {epsilon}))")
                
        return df

    @staticmethod
    def output():
        '''
        Output variables which will be saved to output root file.
        '''
        branchList = [
            # Hadronic tau variables (equivalent to isolated electron)
            "HadronicTau_P",
            "HadronicTau_Phi",
            "HadronicTau_Theta",
            "HadronicTau_E",
            "HadronicTau_CosTheta",
            "HadronicTau_CosPhi",
            "HadronicTau_M",

            # Missing quantities variables (Neutrino)
            "MissingQuantities_P",
            "MissingQuantities_E",
            "Missing_Pt",
            "MissingQuantities_Theta",
            "MissingQuantities_Phi",
            "MissingQuantities_CosTheta",
            "MissingQuantities_CosPhi",
            "MissingQuantities_M",

            # Jet clustering variables
            "JetClustering_d23",
            "JetClustering_d34",

            "JetNConst1", "JetNConst2","JetNConst3",
            
            # Jet 1 kinematics
            "Jet1_P",
            "Jet1_Phi",
            "Jet1_M",
            "Jet1_E",
            "Jet1_Theta",
            "Jet1_CosTheta",
            "Jet1_CosPhi",

            # Jet 2 kinematics
            "Jet2_P",
            "Jet2_Phi",
            "Jet2_M",
            "Jet2_E",
            "Jet2_Theta",
            "Jet2_CosTheta",
            "Jet2_CosPhi",

            # Jet energy
            "Jets_MaxEnergy",
            "Jets_MinEnergy",

            # Hadronic W Boson (Hadronic Tau + Neutrino) kinematics
            "HadronicW_P",
            "HadronicW_Phi",
            "HadronicW_CosPhi",
            "HadronicW_Theta",
            "HadronicW_CosTheta",
            "HadronicW_InvariantMass",
            
            # Leptonic W Boson (Jet1 + Jet2) kinematics
            "LeptonicW_P",
            "LeptonicW_Phi",
            "LeptonicW_CosPhi",
            "LeptonicW_Theta",
            "LeptonicW_CosTheta",
            "LeptonicW_InvariantMass",
            
            # Angular separations between the two W bosons
            "WW_DeltaR",
            "WW_DeltaPhi",
            "WW_DeltaTheta",
            "WW_DeltaInvariantMass",
            "WW_Angle",
            "WW_CosAngle",

            # Event-level masses
            "Event_InvariantMass",
            "Jets_InMa",

            # On-shell and off-shell mass variables
            "System_OnShellMass",
            "System_OffShellMass",
            "System_MaxCosTheta",
            "System_MinCosTheta",
            "System_MaxCosPhi",
            "System_MinCosPhi",

            # Event shape variables
            "EventShape_Planarity",
            "EventShape_Aplanarity",
            "EventShape_Sphericity",
            "EventShape_Asphericity",

            # Jet flavor tagging scores - Logits
            "JetFlavor_UpQuarkProduct_Logit",
            "JetFlavor_DownQuarkProduct_Logit",
            "JetFlavor_StrangeQuarkProduct_Logit",
            "JetFlavor_CharmQuarkProduct_Logit",
            "JetFlavor_BottomQuarkProduct_Logit",
            "JetFlavor_TauProduct_Logit",
            "JetFlavor_UpDownQuarkProduct_Logit",
            "JetFlavor_DownUpQuarkProduct_Logit",
            "JetFlavor_StrangeCharmQuarkProduct_Logit",
            "JetFlavor_CharmStrangeQuarkProduct_Logit",
        ]
        return branchList
